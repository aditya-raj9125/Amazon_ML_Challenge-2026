# =============================================================
# train_eval.py — LightGBM training, threshold sweep, F0.5 evaluation
# =============================================================
# Responsibilities
# ────────────────
# 1. Build training pairs from blocking candidates + ground truth
# 2. TF-IDF fitting on training corpus (serialised for inference reuse)
# 3. LightGBM training with early stopping on val log-loss
# 4. Macro-averaged F0.5 threshold sweep on held-out val set
# 5. Serialise model + best threshold for inference
#
# IMPROVEMENTS:
# - tqdm progress bars on ALL long-running loops
# - Proper error handling with try/except
# - Memory-efficient batch processing
# - NaN handling in feature computation

import os
import gc
import pickle
import random
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from tqdm import tqdm

from features import build_feature_vector, FEATURE_NAMES
from normalize import normalize_name, normalize_address, normalize_name_no_sort, normalize_address_no_sort
from config import (
    LGBM_PARAMS, LGBM_EARLY_STOPPING_ROUNDS,
    NEG_TO_POS_RATIO, RANDOM_SEED,
    TFIDF_NAME_PATH, TFIDF_ADDR_PATH,
    MODEL_PATH, ARTIFACTS_DIR,
    THRESHOLD_LOW, THRESHOLD_HIGH, THRESHOLD_STEP,
)


# ─── F0.5 evaluation helpers ─────────────────────────────────────────────────

def f05_score(precision: float, recall: float) -> float:
    """F0.5 for a single entity. Returns 0 if both P and R are 0."""
    denom = 0.25 * precision + recall
    return (1.25 * precision * recall / denom) if denom > 1e-9 else 0.0


def macro_f05(predictions: dict[str, set[str]],
              ground_truth: dict[str, set[str]]) -> float:
    """
    Compute macro-averaged F0.5 score exactly as the challenge does.

    predictions  : { s1_id : set of predicted match ids }
    ground_truth : { s1_id : set of true match ids }

    All S1 entities from GT must be in predictions (singletons included).
    A singleton predicted correctly (both empty) scores 1.0.
    A singleton predicted as any match scores 0.0.
    """
    scores = []
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())

        if not true_set and not pred_set:
            scores.append(1.0)
            continue
        if not true_set and pred_set:
            scores.append(0.0)
            continue
        if true_set and not pred_set:
            # Precision = 1 (no predictions), Recall = 0
            scores.append(f05_score(1.0, 0.0))
            continue

        inter = len(true_set & pred_set)
        prec  = inter / len(pred_set)
        rec   = inter / len(true_set)
        scores.append(f05_score(prec, rec))

    return float(np.mean(scores)) if scores else 0.0


# ─── Train/Validation split ───────────────────────────────────────────────────

def make_train_val_split(gt: pl.DataFrame,
                          val_fraction: float = 0.20,
                          seed: int = RANDOM_SEED):
    """
    Group-split by source1_entity_id, stratified by match-count bucket.
    Returns (train_s1_ids, val_s1_ids) as Python sets.
    """
    rows = []
    for row in tqdm(gt.iter_rows(named=True), total=len(gt), desc="Parsing splits"):
        s1_id  = row["source1_entity_id"]
        raw    = row.get("matched_entity_ids", "") or ""
        n_match = len([m for m in raw.split(",") if m.strip()])
        bucket  = min(n_match, 5)
        rows.append((s1_id, bucket))

    s1_ids_arr  = [r[0] for r in rows]
    buckets_arr = [r[1] for r in rows]

    train_ids, val_ids = train_test_split(
        s1_ids_arr,
        test_size=val_fraction,
        stratify=buckets_arr,
        random_state=seed,
    )
    return set(train_ids), set(val_ids)


# ─── Record lookup (polars → python dict) ─────────────────────────────────────

def build_record_lookup(df: pl.DataFrame) -> dict[str, dict]:
    """
    Convert a Polars DataFrame into a dict {entity_id → record_dict}.
    Reuses pre-computed normalised columns from Polars if present.
    """
    lookup = {}
    has_norm_name = "norm_name" in df.columns
    has_norm_name_ns = "norm_name_ns" in df.columns
    has_norm_addr = "norm_addr" in df.columns
    has_norm_addr_ns = "norm_addr_ns" in df.columns

    for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Building lookup"):
        eid  = row["entity_id"]
        name = row.get("business_name", "") or ""
        addr = row.get("business_address", "") or ""
        try:
            lookup[eid] = {
                **row,
                "norm_name":    row["norm_name"] if has_norm_name else normalize_name(name),
                "norm_name_ns": row["norm_name_ns"] if has_norm_name_ns else normalize_name_no_sort(name),
                "norm_addr":    row["norm_addr"] if has_norm_addr else normalize_address(addr),
                "norm_addr_ns": row["norm_addr_ns"] if has_norm_addr_ns else normalize_address_no_sort(addr),
                "embed_vec":    None,
            }
        except Exception as e:
            # Skip corrupt records gracefully
            lookup[eid] = {
                "entity_id": eid,
                "business_name": name,
                "business_address": addr,
                "country": row.get("country", ""),
                "norm_name": "",
                "norm_name_ns": "",
                "norm_addr": "",
                "norm_addr_ns": "",
                "embed_vec": None,
            }
    return lookup


def attach_embeddings(lookup: dict[str, dict],
                       entity_ids: list[str],
                       embeddings: np.ndarray) -> None:
    """In-place: attach embedding vectors to the record lookup dict."""
    for eid, emb in tqdm(zip(entity_ids, embeddings), total=len(entity_ids),
                         desc="Attaching embeddings"):
        if eid in lookup:
            lookup[eid]["embed_vec"] = emb


# ─── Pair dataset builder ─────────────────────────────────────────────────────

def build_pair_dataset(
    train_s1_ids: set[str],
    candidates: dict[str, set[str]],
    gt_dict: dict[str, set[str]],
    lookup_all: dict[str, dict],
    neg_ratio: int = NEG_TO_POS_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build the feature matrix X and label vector y for LightGBM training.
    Optimized: downsample negatives BEFORE feature computation.
    """
    rng = random.Random(seed)

    pos_pairs: list[tuple[str, str]] = []
    neg_pairs: list[tuple[str, str]] = []

    print("Selecting candidate pairs for training...")
    for s1_id in tqdm(train_s1_ids, desc="Selecting pairs"):
        if s1_id not in lookup_all:
            continue
        true_m  = gt_dict.get(s1_id, set())
        cand_m  = candidates.get(s1_id, set())

        for s23_id in cand_m:
            if s23_id not in lookup_all:
                continue
            if s23_id in true_m:
                pos_pairs.append((s1_id, s23_id))
            else:
                neg_pairs.append((s1_id, s23_id))

    print(f"  Raw candidate positives : {len(pos_pairs):,}")
    print(f"  Raw candidate negatives : {len(neg_pairs):,}")

    # Downsample negatives BEFORE feature computation
    target_neg = min(len(neg_pairs), len(pos_pairs) * neg_ratio)
    rng.shuffle(neg_pairs)
    neg_pairs = neg_pairs[:target_neg]
    print(f"  Selected negatives       : {len(neg_pairs):,}")

    selected_pairs = pos_pairs + neg_pairs
    y = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs), dtype=np.int8)

    # Compute features ONLY on the selected pairs
    print(f"Computing features for {len(selected_pairs):,} selected pairs...")
    rows = []
    errors = 0
    for s1_id, s23_id in tqdm(selected_pairs, desc="Extracting features"):
        try:
            rec_a = lookup_all[s1_id]
            rec_b = lookup_all[s23_id]
            fv = build_feature_vector(rec_a, rec_b)
            rows.append([fv.get(f, 0.0) for f in FEATURE_NAMES])
        except Exception as e:
            rows.append([0.0] * len(FEATURE_NAMES))
            errors += 1

    if errors > 0:
        print(f"  [WARNING] {errors:,} pairs had feature computation errors (defaulted to zeros)")

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    # Replace infinity with NaN (LightGBM handles NaN natively)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y


# ─── TF-IDF fitting ───────────────────────────────────────────────────────────

def fit_tfidf(texts_name: list[str],
              texts_addr: list[str],
              analyzer: str = "char_wb",
              ngram_range: tuple = (2, 4),
              max_features: int = 200_000):
    """
    Fit two TF-IDF vectorisers (name and address) on the training corpus.
    Serialises to disk for reuse during inference.
    """
    print("Fitting TF-IDF on names ...")
    tfidf_name = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
    )
    tfidf_name.fit(texts_name)

    print("Fitting TF-IDF on addresses ...")
    tfidf_addr = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
    )
    tfidf_addr.fit(texts_addr)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(TFIDF_NAME_PATH, "wb") as f:
        pickle.dump(tfidf_name, f)
    with open(TFIDF_ADDR_PATH, "wb") as f:
        pickle.dump(tfidf_addr, f)

    print(f"TF-IDF models saved to {ARTIFACTS_DIR}")
    return tfidf_name, tfidf_addr


# ─── LightGBM training ───────────────────────────────────────────────────────

def train_lightgbm(X_train: pd.DataFrame, y_train: np.ndarray,
                   X_val:   pd.DataFrame, y_val:   np.ndarray) -> lgb.LGBMClassifier:
    """
    Train LightGBM binary classifier with early stopping.
    Handles NaN values natively (no fillna needed).
    """
    model = lgb.LGBMClassifier(**LGBM_PARAMS)

    print("Training LightGBM ...")
    print(f"  Train set: {len(X_train):,} pairs ({y_train.sum():,} pos)")
    print(f"  Val set:   {len(X_val):,} pairs ({y_val.sum():,} pos)")
    print(f"  Features:  {X_train.shape[1]}")
    print(f"  Params:    num_leaves={LGBM_PARAMS['num_leaves']}, "
          f"n_estimators={LGBM_PARAMS['n_estimators']}, "
          f"lr={LGBM_PARAMS['learning_rate']}")

    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOPPING_ROUNDS,
                                    verbose=True),
                lgb.log_evaluation(period=50),
            ],
        )
    except Exception as e:
        print(f"  [ERROR] LightGBM training failed: {e}")
        traceback.print_exc()
        raise

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Best iteration: {model.best_iteration_}")

    # Print feature importance
    try:
        importances = model.feature_importances_
        feat_imp = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
        print("\n  Top 15 features by importance:")
        for fname, imp in feat_imp[:15]:
            print(f"    {fname:<30}: {imp:>6}")
    except Exception:
        pass

    return model


# ─── Threshold sweep on validation set ───────────────────────────────────────

def sweep_threshold(model: lgb.LGBMClassifier,
                    val_s1_ids: set[str],
                    candidates: dict[str, set[str]],
                    gt_dict: dict[str, set[str]],
                    lookup_all: dict[str, dict]) -> float:
    """
    Score every candidate pair in the val split, then sweep thresholds.
    Returns the optimal threshold maximizing MACRO-averaged F0.5.
    """
    # Score all val pairs in batches
    val_scores: dict[str, dict[str, float]] = {}

    rows = []
    index_map = []

    for s1_id in tqdm(val_s1_ids, desc="Scoring val pairs"):
        cands = candidates.get(s1_id, set())
        if s1_id not in lookup_all:
            continue
        rec_a = lookup_all[s1_id]
        for s23_id in cands:
            if s23_id not in lookup_all:
                continue
            rec_b = lookup_all[s23_id]
            try:
                fv = build_feature_vector(rec_a, rec_b)
                rows.append([fv.get(f, 0.0) for f in FEATURE_NAMES])
                index_map.append((s1_id, s23_id))
            except Exception:
                continue

    if not rows:
        print("[Threshold] No val pairs to score.")
        return 0.5

    X_val_infer = pd.DataFrame(rows, columns=FEATURE_NAMES)
    X_val_infer = X_val_infer.replace([np.inf, -np.inf], np.nan)
    del rows
    gc.collect()

    print(f"  Scoring {len(X_val_infer):,} val pairs ...")
    probs = model.predict_proba(X_val_infer)[:, 1]
    del X_val_infer
    gc.collect()

    # Populate score dict
    for (s1_id, s23_id), prob in zip(index_map, probs):
        if s1_id not in val_scores:
            val_scores[s1_id] = {}
        val_scores[s1_id][s23_id] = float(prob)

    # Ensure all val S1 entities appear
    for s1_id in val_s1_ids:
        if s1_id not in val_scores:
            val_scores[s1_id] = {}

    best_t   = 0.5
    best_f05 = 0.0

    thresholds = np.arange(THRESHOLD_LOW, THRESHOLD_HIGH + 1e-9, THRESHOLD_STEP)
    print(f"  Sweeping {len(thresholds)} thresholds from {THRESHOLD_LOW} to {THRESHOLD_HIGH} ...")

    for t in tqdm(thresholds, desc="Threshold sweep"):
        preds = {
            s1_id: {s23_id for s23_id, prob in scores.items() if prob >= t}
            for s1_id, scores in val_scores.items()
        }
        score = macro_f05(preds, gt_dict)
        if score > best_f05:
            best_f05 = score
            best_t   = float(t)

    print(f"\n[Threshold] Best threshold: {best_t:.4f}  →  Val F0.5 = {best_f05:.5f}")
    return best_t


# ─── Full evaluation report ───────────────────────────────────────────────────

def evaluate_on_val(model: lgb.LGBMClassifier,
                    threshold: float,
                    val_s1_ids: set[str],
                    candidates: dict[str, set[str]],
                    gt_dict: dict[str, set[str]],
                    lookup_all: dict[str, dict]) -> dict:
    """
    Produce a full evaluation report on the val split at the given threshold.
    """
    rows      = []
    index_map = []

    for s1_id in tqdm(val_s1_ids, desc="Eval scoring"):
        cands = candidates.get(s1_id, set())
        if s1_id not in lookup_all:
            continue
        rec_a = lookup_all[s1_id]
        for s23_id in cands:
            if s23_id not in lookup_all:
                continue
            rec_b = lookup_all[s23_id]
            try:
                fv = build_feature_vector(rec_a, rec_b)
                rows.append([fv.get(f, 0.0) for f in FEATURE_NAMES])
                index_map.append((s1_id, s23_id))
            except Exception:
                continue

    val_scores: dict[str, dict[str, float]] = {}
    if rows:
        X_infer = pd.DataFrame(rows, columns=FEATURE_NAMES)
        X_infer = X_infer.replace([np.inf, -np.inf], np.nan)
        probs   = model.predict_proba(X_infer)[:, 1]
        for (s1_id, s23_id), prob in zip(index_map, probs):
            if s1_id not in val_scores:
                val_scores[s1_id] = {}
            val_scores[s1_id][s23_id] = float(prob)
        del X_infer, rows
        gc.collect()

    for s1_id in val_s1_ids:
        if s1_id not in val_scores:
            val_scores[s1_id] = {}

    preds = {
        s1_id: {s23_id for s23_id, prob in scores.items() if prob >= threshold}
        for s1_id, scores in val_scores.items()
    }

    # Restrict GT to val entities
    val_gt = {s1_id: gt_dict.get(s1_id, set()) for s1_id in val_s1_ids}

    f05 = macro_f05(preds, val_gt)

    # Aggregate precision / recall
    all_prec, all_rec = [], []
    singleton_correct = 0
    singleton_total   = 0
    false_positives   = 0
    false_negatives   = 0

    for s1_id, true_set in val_gt.items():
        pred_set = preds.get(s1_id, set())
        if not true_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
            else:
                false_positives += len(pred_set)
            continue
        inter = len(true_set & pred_set)
        fp = len(pred_set) - inter
        fn = len(true_set) - inter
        false_positives += fp
        false_negatives += fn
        all_prec.append(inter / len(pred_set) if pred_set else 1.0)
        all_rec.append(inter / len(true_set))

    results = {
        "macro_f05":          f05,
        "macro_precision":    float(np.mean(all_prec)) if all_prec else 0.0,
        "macro_recall":       float(np.mean(all_rec)) if all_rec else 0.0,
        "singleton_accuracy": singleton_correct / singleton_total if singleton_total else 1.0,
        "false_positives":    false_positives,
        "false_negatives":    false_negatives,
    }

    print("\n" + "="*55)
    print("  VALIDATION EVALUATION REPORT")
    print("="*55)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<30}: {v:.5f}")
        else:
            print(f"  {k:<30}: {v:,}")
    print("="*55 + "\n")

    return results
