# =============================================================
# train_eval.py — LightGBM training, threshold sweep, F0.5 evaluation
# =============================================================
# Responsibilities
# ────────────────
# 1. Build training pairs from blocking candidates + ground truth
#    (positive labels from GT, negative labels = hard negatives
#     from same-block non-matches, downsampled NEG_TO_POS_RATIO:1)
# 2. TF-IDF fitting on training corpus (serialised for inference reuse)
# 3. LightGBM training with early stopping on val log-loss
# 4. Macro-averaged F0.5 threshold sweep on held-out val set
# 5. Serialise model + best threshold for inference

import os
import pickle
import random
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

    Design
    ------
    - Never split an S1 entity's matches across train/val: that would be
      label leakage because the model would see the true S2/S3 match of an
      S1 entity while training and then "validate" on it.
    - Stratify by match-count bucket (0,1,2,3,4,5+) so the val set has
      proportionate singleton representation (5.58% in train data) and
      multi-match entities.
    - Country distribution is roughly preserved because match-count
      stratification implicitly controls for it (India/US have similar
      match-count profiles in the dataset).
    """
    # Parse match counts
    rows = []
    for row in gt.iter_rows(named=True):
        s1_id  = row["source1_entity_id"]
        raw    = row.get("matched_entity_ids", "") or ""
        n_match = len([m for m in raw.split(",") if m.strip()])
        bucket  = min(n_match, 5)   # cap at 5 for stratification
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
    Adds pre-computed normalised fields so we don't recompute per pair.
    """
    lookup = {}
    for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Building lookup"):
        eid  = row["entity_id"]
        name = row.get("business_name", "") or ""
        addr = row.get("business_address", "") or ""
        lookup[eid] = {
            **row,
            "norm_name":    normalize_name(name),
            "norm_name_ns": normalize_name_no_sort(name),
            "norm_addr":    normalize_address(addr),
            "norm_addr_ns": normalize_address_no_sort(addr),
            "embed_vec":    None,   # filled in after encoding
        }
    return lookup


def attach_embeddings(lookup: dict[str, dict],
                       entity_ids: list[str],
                       embeddings: np.ndarray) -> None:
    """In-place: attach embedding vectors to the record lookup dict."""
    for eid, emb in zip(entity_ids, embeddings):
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

    Positive pairs  : all (s1_id, s23_id) in GT intersect candidates.
    Hard negatives  : same-block (s1_id, s23_id) pairs that are NOT in GT,
                      downsampled to neg_ratio positives.
    Easy negatives  : random cross-country or random pairs are deliberately
                      excluded — they are too easy and waste label budget.

    Returns
    -------
    X : pd.DataFrame of shape (N, |FEATURE_NAMES|)
    y : np.ndarray of shape (N,), dtype int8, values {0, 1}
    """
    rng = random.Random(seed)

    positives = []   # list of feature dicts
    negatives = []   # list of feature dicts

    for s1_id in tqdm(train_s1_ids, desc="Building pairs"):
        if s1_id not in lookup_all:
            continue
        rec_a   = lookup_all[s1_id]
        true_m  = gt_dict.get(s1_id, set())
        cand_m  = candidates.get(s1_id, set())

        for s23_id in cand_m:
            if s23_id not in lookup_all:
                continue
            rec_b = lookup_all[s23_id]
            fv    = build_feature_vector(rec_a, rec_b)

            if s23_id in true_m:
                positives.append(fv)
            else:
                negatives.append(fv)

    print(f"  Raw positives : {len(positives):,}")
    print(f"  Raw negatives : {len(negatives):,}")

    # Downsample negatives (keep hard ones — already from same block)
    target_neg = min(len(negatives), len(positives) * neg_ratio)
    rng.shuffle(negatives)
    negatives = negatives[:target_neg]
    print(f"  After downsampling negatives: {len(negatives):,}")

    pos_df = pd.DataFrame(positives, columns=FEATURE_NAMES).fillna(0.0)
    neg_df = pd.DataFrame(negatives, columns=FEATURE_NAMES).fillna(0.0)

    X = pd.concat([pos_df, neg_df], ignore_index=True)
    y = np.array([1] * len(positives) + [0] * len(negatives), dtype=np.int8)

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

    These are used to compute cosine similarity between pairs — the
    TF-IDF model is fit only on training data (no external corpus).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

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

    scale_pos_weight is set BELOW 1.0 (see config) because F0.5 penalises
    false positives 2x more than false negatives — we want the model to be
    conservative, not recall-maximising.

    Early stopping uses val binary_logloss.
    Best iteration is saved automatically by LightGBM.
    """
    model = lgb.LGBMClassifier(**LGBM_PARAMS)

    print("Training LightGBM ...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOPPING_ROUNDS,
                                verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Best iteration: {model.best_iteration_}")

    return model


# ─── Threshold sweep on validation set ───────────────────────────────────────

def sweep_threshold(model: lgb.LGBMClassifier,
                    val_s1_ids: set[str],
                    candidates: dict[str, set[str]],
                    gt_dict: dict[str, set[str]],
                    lookup_all: dict[str, dict]) -> float:
    """
    Score every candidate pair in the val split, then sweep thresholds
    from THRESHOLD_LOW to THRESHOLD_HIGH in THRESHOLD_STEP steps.

    The threshold is chosen to maximise MACRO-averaged F0.5 —
    exactly the way the challenge scores it.

    Returns the optimal threshold.
    """
    import numpy as np

    # Score all val pairs
    val_scores: dict[str, dict[str, float]] = {}   # s1_id → {s23_id: score}

    rows = []
    index_map = []   # (s1_id, s23_id) per row

    for s1_id in tqdm(val_s1_ids, desc="Scoring val pairs"):
        cands = candidates.get(s1_id, set())
        if s1_id not in lookup_all:
            continue
        rec_a = lookup_all[s1_id]
        for s23_id in cands:
            if s23_id not in lookup_all:
                continue
            rec_b = lookup_all[s23_id]
            fv = build_feature_vector(rec_a, rec_b)
            rows.append([fv.get(f, 0.0) for f in FEATURE_NAMES])
            index_map.append((s1_id, s23_id))

    if not rows:
        print("[Threshold] No val pairs to score.")
        return 0.5

    X_val_infer = pd.DataFrame(rows, columns=FEATURE_NAMES).fillna(0.0)
    probs = model.predict_proba(X_val_infer)[:, 1]

    # Populate score dict
    for (s1_id, s23_id), prob in zip(index_map, probs):
        if s1_id not in val_scores:
            val_scores[s1_id] = {}
        val_scores[s1_id][s23_id] = float(prob)

    # Ensure all val S1 entities appear (singletons with no candidates)
    for s1_id in val_s1_ids:
        if s1_id not in val_scores:
            val_scores[s1_id] = {}

    best_t   = 0.5
    best_f05 = 0.0

    thresholds = np.arange(THRESHOLD_LOW, THRESHOLD_HIGH + 1e-9, THRESHOLD_STEP)
    for t in thresholds:
        preds = {
            s1_id: {s23_id for s23_id, prob in scores.items() if prob >= t}
            for s1_id, scores in val_scores.items()
        }
        score = macro_f05(preds, gt_dict)
        if score > best_f05:
            best_f05 = score
            best_t   = float(t)

    print(f"[Threshold] Best threshold: {best_t:.3f}  ->  Val F0.5 = {best_f05:.5f}")
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

    Prints and returns a dict with:
        macro_f05, macro_precision, macro_recall,
        singleton_accuracy, blocking_recall
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
            fv = build_feature_vector(rec_a, rec_b)
            rows.append([fv.get(f, 0.0) for f in FEATURE_NAMES])
            index_map.append((s1_id, s23_id))

    val_scores: dict[str, dict[str, float]] = {}
    if rows:
        X_infer = pd.DataFrame(rows, columns=FEATURE_NAMES).fillna(0.0)
        probs   = model.predict_proba(X_infer)[:, 1]
        for (s1_id, s23_id), prob in zip(index_map, probs):
            if s1_id not in val_scores:
                val_scores[s1_id] = {}
            val_scores[s1_id][s23_id] = float(prob)

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

    for s1_id, true_set in val_gt.items():
        pred_set = preds.get(s1_id, set())
        if not true_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
            continue
        inter = len(true_set & pred_set)
        all_prec.append(inter / len(pred_set) if pred_set else 1.0)
        all_rec.append(inter / len(true_set))

    results = {
        "macro_f05":          f05,
        "macro_precision":    float(np.mean(all_prec)) if all_prec else 0.0,
        "macro_recall":       float(np.mean(all_rec)) if all_rec else 0.0,
        "singleton_accuracy": singleton_correct / singleton_total if singleton_total else 1.0,
    }

    print("\n" + "="*55)
    print("  VALIDATION EVALUATION REPORT")
    print("="*55)
    for k, v in results.items():
        print(f"  {k:<30}: {v:.5f}")
    print("="*55 + "\n")

    return results
