# =============================================================
# pipeline.py — Full end-to-end orchestrator
# =============================================================
# This module ties every stage together in the correct order.
# It is called from the single Colab notebook cell-by-cell.
#
# Stage order
# ───────────
#  0. Setup & install
#  1. Load data (Polars, memory-efficient)
#  2. Normalise text fields
#  3. Encode with multilingual sentence encoder (GPU)
#  4. Generate candidates (blocking union)
#  5. Measure blocking recall on train GT
#  6. Train / validation split (group-split by S1 entity_id)
#  7. Build pair feature matrix for training
#  8. Fit TF-IDF (serialised for reuse)
#  9. Train LightGBM with early stopping
# 10. Threshold sweep → optimal threshold for F0.5
# 11. Full validation evaluation report
# 12. Run inference on test set
# 13. Post-process (one-to-one dedup + graph pruning)
# 14. Write output TSVs
# 15. Validate submission with utils/validate_submission.py

import os
import sys
import pickle

import numpy as np
import polars as pl
from tqdm import tqdm

# ─── Ensure src/ is on sys.path when called from notebook ────────────────────
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1,  TEST_S2,  TEST_S3,
    ARTIFACTS_DIR, OUTPUT_DIR,
    VAL_FRACTION, RANDOM_SEED,
    EMBED_MODEL_NAME, EMBED_BATCH_SIZE,
    EMBED_S1_TRAIN_PATH, EMBED_S2_TRAIN_PATH, EMBED_S3_TRAIN_PATH,
    EMBED_S1_TEST_PATH,  EMBED_S2_TEST_PATH,  EMBED_S3_TEST_PATH,
    PARQUET_CACHE_DIR,
    ANN_TOP_K, SNM_WINDOW,
    LGBM_PARAMS, LGBM_EARLY_STOPPING_ROUNDS, NEG_TO_POS_RATIO,
    ENABLE_ONE_TO_ONE_DEDUP, ENABLE_GRAPH_PRUNING,
)
from normalize import (
    LEGAL_SUFFIXES, ADDR_ABBREV,
    normalize_name, normalize_address, normalize_name_no_sort, normalize_address_no_sort,
    build_text_for_embedding,
)
from blocking import generate_candidates, compute_blocking_recall, encode_texts
from train_eval import (
    build_record_lookup, attach_embeddings,
    make_train_val_split, build_pair_dataset,
    fit_tfidf, train_lightgbm,
    sweep_threshold, evaluate_on_val,
)
from predict import load_model, run_inference


# ─── Data loading ─────────────────────────────────────────────────────────────

def _get_parquet_path(tsv_path: str) -> str:
    """Return the parquet cache path corresponding to a TSV file."""
    if PARQUET_CACHE_DIR:
        os.makedirs(PARQUET_CACHE_DIR, exist_ok=True)
        fname = os.path.splitext(os.path.basename(tsv_path))[0] + ".parquet"
        return os.path.join(PARQUET_CACHE_DIR, fname)
    return os.path.splitext(tsv_path)[0] + ".parquet"


def load_source(path: str) -> pl.DataFrame:
    """
    Load a TSV source file with Polars.
    Checks for a cached .parquet version first. If missing, loads TSV,
    cleans nulls, caches as zstd-compressed parquet for fast subsequent loads,
    and returns the DataFrame.
    """
    parquet_path = _get_parquet_path(path)
    if os.path.exists(parquet_path):
        print(f"Loading {os.path.basename(parquet_path)} (from parquet cache) ...")
        df = pl.read_parquet(parquet_path)
        print(f"  Rows: {len(df):,}")
        return df

    print(f"Loading {os.path.basename(path)} (TSV) ...")
    df = pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=10_000,
        null_values=["", "NULL", "null", "N/A"]
    )
    # Fill nulls with empty string for text columns
    for col in ["business_name", "business_address", "country"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null(""))

    print(f"  Rows: {len(df):,}")

    # Convert and cache to parquet for fast subsequent loads
    try:
        df.write_parquet(parquet_path, compression="zstd")
        print(f"  Saved parquet cache to {parquet_path}")
    except Exception as e:
        print(f"  [Notice] Could not write parquet cache ({e}). Continuing with in-memory DataFrame.")

    return df


def load_ground_truth(path: str) -> pl.DataFrame:
    """
    Load ground truth TSV with Polars.
    Checks for a cached .parquet version first. If missing, loads TSV,
    cleans nulls, caches to parquet, and returns the DataFrame.
    """
    parquet_path = _get_parquet_path(path)
    if os.path.exists(parquet_path):
        print(f"Loading {os.path.basename(parquet_path)} (from parquet cache) ...")
        df = pl.read_parquet(parquet_path)
        print(f"  GT rows: {len(df):,}")
        return df

    print(f"Loading ground truth (TSV) ...")
    df = pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=1000,
        null_values=["", "NULL", "null"]
    )
    if "matched_entity_ids" in df.columns:
        df = df.with_columns(pl.col("matched_entity_ids").fill_null(""))
    print(f"  GT rows: {len(df):,}")

    try:
        df.write_parquet(parquet_path, compression="zstd")
        print(f"  Saved ground truth parquet cache to {parquet_path}")
    except Exception as e:
        print(f"  [Notice] Could not write parquet cache ({e}). Continuing with in-memory DataFrame.")

    return df


def gt_to_dict(gt_df: pl.DataFrame) -> dict[str, set[str]]:
    """Parse ground truth DataFrame to { s1_id : set of matched ids }."""
    gt_dict = {}
    for row in gt_df.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        raw   = row.get("matched_entity_ids", "") or ""
        matches = {m.strip() for m in raw.split(",") if m.strip()}
        gt_dict[s1_id] = matches
    return gt_dict


# ─── Embedding helpers ────────────────────────────────────────────────────────

def get_or_compute_embeddings(df: pl.DataFrame,
                               save_path: str,
                               force_recompute: bool = False) -> np.ndarray:
    """
    Load pre-computed embeddings from disk or compute and cache them.

    Embeddings are the most expensive step (~1-2 hrs on T4 for 5M rows).
    Caching ensures you don't redo this if you restart the notebook.

    The text fed to the encoder is: raw business_name + " " + raw business_address
    (un-normalised, so the multilingual model can use its full vocabulary).
    """
    if not force_recompute and os.path.exists(save_path):
        print(f"Loading cached embeddings from {save_path} ...")
        return np.load(save_path)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    texts = [
        build_text_for_embedding(
            row.get("business_name", "") or "",
            row.get("business_address", "") or "",
        )
        for row in df.iter_rows(named=True)
    ]
    print(f"Encoding {len(texts):,} texts with {EMBED_MODEL_NAME} ...")
    embeds = encode_texts(texts)
    np.save(save_path, embeds)
    print(f"Embeddings saved to {save_path}  shape={embeds.shape}")
    return embeds


# ─── Stage functions (called from notebook in sequence) ──────────────────────

def stage_load_train():
    """
    Stage 1 — Load all training source files and ground truth.
    Returns (s1, s2, s3, gt) as Polars DataFrames.
    """
    s1 = load_source(TRAIN_S1)
    s2 = load_source(TRAIN_S2)
    s3 = load_source(TRAIN_S3)
    gt = load_ground_truth(TRAIN_GT)
    return s1, s2, s3, gt


def stage_load_test():
    """
    Stage 2 — Load all test source files.
    Returns (ts1, ts2, ts3) as Polars DataFrames.
    """
    ts1 = load_source(TEST_S1)
    ts2 = load_source(TEST_S2)
    ts3 = load_source(TEST_S3)
    return ts1, ts2, ts3


def stage_encode_train(s1, s2, s3, force=False):
    """
    Stage 3 — Compute (or load) multilingual embeddings for train sources.
    Returns (e1, e2, e3) as float32 numpy arrays.
    """
    e1 = get_or_compute_embeddings(s1, EMBED_S1_TRAIN_PATH, force_recompute=force)
    e2 = get_or_compute_embeddings(s2, EMBED_S2_TRAIN_PATH, force_recompute=force)
    e3 = get_or_compute_embeddings(s3, EMBED_S3_TRAIN_PATH, force_recompute=force)
    return e1, e2, e3


def stage_encode_test(ts1, ts2, ts3, force=False):
    """
    Stage 4 — Compute (or load) embeddings for test sources.
    Returns (te1, te2, te3) as float32 numpy arrays.
    """
    te1 = get_or_compute_embeddings(ts1, EMBED_S1_TEST_PATH, force_recompute=force)
    te2 = get_or_compute_embeddings(ts2, EMBED_S2_TEST_PATH, force_recompute=force)
    te3 = get_or_compute_embeddings(ts3, EMBED_S3_TEST_PATH, force_recompute=force)
    return te1, te2, te3


def stage_add_norm_cols(df: pl.DataFrame) -> pl.DataFrame:
    """
    Stage helper — add norm_name, norm_name_ns, norm_addr, and norm_addr_ns columns.
    Uses Polars native string expressions for high-throughput vectorized execution
    (~10x faster than row-by-row map_elements).
    """
    if "norm_name" in df.columns and "norm_addr" in df.columns:
        print(f"  Columns norm_name and norm_addr already exist for {len(df):,} rows.")
        return df

    print(f"  Normalising {len(df):,} rows (Polars native expressions) ...")

    # Legal suffix removal regex pattern (case-insensitive word boundary match)
    legal_pattern = r"(?i)\b(" + "|".join(sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b"

    # Name normalisation:
    # 1. Base clean: lower, '&' -> 'and', remove non-alphanumeric, strip legal suffixes, collapse spaces
    base_name = (
        pl.col("business_name")
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"&", " and ")
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(legal_pattern, " ")
        .str.replace_all(r"\s+", " ")
        .str.replace_all(r"^\s+|\s+$", "")
    )

    # Address normalisation:
    base_addr = (
        pl.col("business_address")
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"&", " and ")
        .str.replace_all(r"[^\w\s]", " ")
    )
    for k, v in ADDR_ABBREV.items():
        base_addr = base_addr.str.replace_all(rf"\b{k}\b", v)

    base_addr = (
        base_addr
        .str.replace_all(r"\s+", " ")
        .str.replace_all(r"^\s+|\s+$", "")
    )

    # Token-sorted name: split on space, sort tokens, join with space
    sorted_name = base_name.str.split(" ").list.sort().list.join(" ")

    df = df.with_columns([
        sorted_name.alias("norm_name"),
        base_name.alias("norm_name_ns"),
        base_addr.alias("norm_addr"),
        base_addr.alias("norm_addr_ns"),
    ])
    return df


def stage_blocking_train(s1, s2, s3, e1, e2, e3):
    """
    Stage 5 — Generate candidate pairs for the TRAINING set.
    Returns candidates dict: { s1_id : set of s23_ids }
    """
    print("\n[Stage] Blocking on TRAIN ...")
    candidates = generate_candidates(s1, s2, s3, e1, e2, e3,
                                      top_k=ANN_TOP_K, snm_window=SNM_WINDOW)
    return candidates


def stage_blocking_test(ts1, ts2, ts3, te1, te2, te3):
    """
    Stage 6 — Generate candidate pairs for the TEST set.
    Returns test_candidates dict: { s1_id : set of s23_ids }
    """
    print("\n[Stage] Blocking on TEST ...")
    test_candidates = generate_candidates(ts1, ts2, ts3, te1, te2, te3,
                                           top_k=ANN_TOP_K, snm_window=SNM_WINDOW)
    return test_candidates


def stage_blocking_recall(candidates, gt_df):
    """
    Stage 7 — Measure blocking recall against training ground truth.
    Prints and returns the float recall value.
    A value < 0.99 is a warning — investigate which blocking strategy is missing pairs.
    """
    print("\n[Stage] Measuring blocking recall ...")
    recall = compute_blocking_recall(candidates, gt_df)
    if recall < 0.97:
        print(f"WARNING: Blocking recall = {recall:.4f} < 0.97. "
              "Consider increasing ANN_TOP_K or SNM_WINDOW in config.py.")
    return recall


def stage_build_lookups(s1, s2, s3, e1, e2, e3):
    """
    Stage 8 — Build record lookup dicts for all train sources,
    attach embedding vectors for use as features.

    Returns lookup_all dict merging S1 + S2 + S3.
    """
    print("\n[Stage] Building record lookups ...")
    lookup_s1 = build_record_lookup(s1)
    lookup_s2 = build_record_lookup(s2)
    lookup_s3 = build_record_lookup(s3)

    print("  Attaching embeddings ...")
    attach_embeddings(lookup_s1, s1["entity_id"].to_list(), e1)
    attach_embeddings(lookup_s2, s2["entity_id"].to_list(), e2)
    attach_embeddings(lookup_s3, s3["entity_id"].to_list(), e3)

    lookup_all = {**lookup_s1, **lookup_s2, **lookup_s3}
    print(f"  Total records in lookup: {len(lookup_all):,}")
    return lookup_all


def stage_build_test_lookups(ts1, ts2, ts3, te1, te2, te3):
    """
    Stage 9 — Build record lookup dicts for all test sources.
    Returns test_lookup_all dict.
    """
    print("\n[Stage] Building test record lookups ...")
    lu1 = build_record_lookup(ts1)
    lu2 = build_record_lookup(ts2)
    lu3 = build_record_lookup(ts3)

    attach_embeddings(lu1, ts1["entity_id"].to_list(), te1)
    attach_embeddings(lu2, ts2["entity_id"].to_list(), te2)
    attach_embeddings(lu3, ts3["entity_id"].to_list(), te3)

    test_lookup_all = {**lu1, **lu2, **lu3}
    print(f"  Total test records in lookup: {len(test_lookup_all):,}")
    return test_lookup_all


def stage_train_val_split(gt_df):
    """
    Stage 10 — Group-split by source1_entity_id, stratified by match_count bucket.
    Returns (train_s1_ids, val_s1_ids) as sets.
    """
    print("\n[Stage] Train/val split ...")
    train_ids, val_ids = make_train_val_split(gt_df, VAL_FRACTION, RANDOM_SEED)
    print(f"  Train S1 entities: {len(train_ids):,}")
    print(f"  Val   S1 entities: {len(val_ids):,}")
    return train_ids, val_ids


def stage_build_training_pairs(train_s1_ids, candidates, gt_dict, lookup_all):
    """
    Stage 11 — Build (X_train, y_train) from training candidates.
    Positives from GT, hard negatives from same-block non-matches.
    Returns (X_train, y_train, X_val_ids)
    NOTE: val pairs are scored separately during threshold sweep.
    """
    print("\n[Stage] Building training pairs ...")
    X_train, y_train = build_pair_dataset(
        train_s1_ids=train_s1_ids,
        candidates=candidates,
        gt_dict=gt_dict,
        lookup_all=lookup_all,
        neg_ratio=NEG_TO_POS_RATIO,
        seed=RANDOM_SEED,
    )
    print(f"  X_train shape: {X_train.shape}  positives: {y_train.sum():,}")
    return X_train, y_train


def stage_train_model(X_train, y_train, val_s1_ids, candidates, gt_dict, lookup_all):
    """
    Stage 12 — Train LightGBM. Uses a small val pair sample for early stopping.

    We build a lightweight val feature matrix (not the full threshold-sweep set)
    to drive early stopping — this is log-loss based, not F0.5, which is fine
    because we tune F0.5 separately via threshold sweep.
    """
    import pandas as pd
    from features import FEATURE_NAMES
    from features import build_feature_vector
    import random

    print("\n[Stage] Building val pairs for early stopping ...")
    rng = random.Random(RANDOM_SEED)
    val_sample = list(val_s1_ids)
    rng.shuffle(val_sample)
    val_sample = val_sample[:min(50_000, len(val_sample))]  # cap for speed

    val_rows   = []
    val_labels = []
    for s1_id in tqdm(val_sample, desc="Val pairs"):
        cands = candidates.get(s1_id, set())
        if s1_id not in lookup_all:
            continue
        rec_a = lookup_all[s1_id]
        true_m = gt_dict.get(s1_id, set())
        for s23_id in cands:
            if s23_id not in lookup_all:
                continue
            rec_b = lookup_all[s23_id]
            fv = build_feature_vector(rec_a, rec_b)
            val_rows.append([fv.get(f, 0.0) for f in FEATURE_NAMES])
            val_labels.append(1 if s23_id in true_m else 0)

    X_val_es = pd.DataFrame(val_rows, columns=FEATURE_NAMES).fillna(0.0)
    y_val_es = np.array(val_labels, dtype=np.int8)
    print(f"  Early-stopping val set: {len(X_val_es):,} pairs  "
          f"(positives: {y_val_es.sum():,})")

    model = train_lightgbm(X_train, y_train, X_val_es, y_val_es)
    return model


def stage_threshold_sweep(model, val_s1_ids, candidates, gt_dict, lookup_all):
    """
    Stage 13 — Sweep thresholds on val set to maximise macro F0.5.
    Returns optimal threshold float.
    """
    print("\n[Stage] Threshold sweep ...")
    threshold = sweep_threshold(model, val_s1_ids, candidates, gt_dict, lookup_all)
    return threshold


def stage_evaluate(model, threshold, val_s1_ids, candidates, gt_dict, lookup_all):
    """
    Stage 14 — Full validation evaluation report at optimal threshold.
    Prints macro F0.5, precision, recall, singleton accuracy.
    Returns results dict.
    """
    print("\n[Stage] Full validation evaluation ...")
    results = evaluate_on_val(model, threshold, val_s1_ids, candidates, gt_dict, lookup_all)
    return results


def stage_inference(model, threshold, ts1, test_candidates, test_lookup_all):
    """
    Stage 15 — Run inference on test set.
    Writes matching_results.tsv and candidate_pairs.tsv to OUTPUT_DIR.
    """
    print("\n[Stage] Test set inference ...")
    test_s1_ids = ts1["entity_id"].to_list()
    predictions = run_inference(
        model=model,
        threshold=threshold,
        test_s1_ids=test_s1_ids,
        candidates=test_candidates,
        lookup_all=test_lookup_all,
        output_dir=OUTPUT_DIR,
    )
    return predictions


def stage_validate_submission():
    """
    Stage 16 — Run the stdlib-only validator on the output files.
    Import and call validate() directly (no subprocess needed in Colab).
    """
    import importlib.util, sys
    validator_path = os.path.join(
        os.path.dirname(_SRC_DIR), "..", "..", "utils", "validate_submission.py"
    )
    validator_path = os.path.abspath(validator_path)

    spec = importlib.util.spec_from_file_location("validate_submission", validator_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from config import OUTPUT_DIR, TEST_S1
    test_dir = os.path.dirname(TEST_S1)
    matching  = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    try:
        mod.validate(matching, candidate, test_dir)
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError("Submission validation FAILED. Check output above.")
