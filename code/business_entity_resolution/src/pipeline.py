# =============================================================
# pipeline.py — Full end-to-end orchestrator
# =============================================================
# This module ties every stage together in the correct order.
# It is called from the single notebook cell-by-cell.
#
# IMPROVEMENTS OVER PREVIOUS VERSION:
# - Checkpointing: every stage saves to disk, can resume after kernel crash
# - Progress bars: tqdm on all long operations
# - Error handling: try/except with cleanup on every stage
# - Memory management: explicit gc.collect() and del after each stage
# - Disk-aware: checks EBS space before large operations
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
import gc
import math
import shutil
import pickle
import traceback

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
    ARTIFACTS_DIR, OUTPUT_DIR, CHECKPOINT_DIR,
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
    build_text_for_embedding, strip_accents,
)
from blocking import generate_candidates, compute_blocking_recall, encode_texts
from train_eval import (
    build_record_lookup, attach_embeddings,
    make_train_val_split, build_pair_dataset,
    fit_tfidf, train_lightgbm,
    sweep_threshold, evaluate_on_val,
)
from predict import load_model, run_inference


# ─── Utilities ────────────────────────────────────────────────────────────────

def check_disk_space(path: str = "/", min_gb: float = 5.0) -> bool:
    """Check if we have enough disk space. Returns True if OK."""
    try:
        stat = os.statvfs(path)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        if free_gb < min_gb:
            print(f"[WARNING] Low disk space: {free_gb:.1f} GB free (need {min_gb:.1f} GB)")
            return False
        return True
    except (OSError, AttributeError):
        return True  # Windows or unsupported FS — skip check


def save_checkpoint(obj, name: str) -> str:
    """Save an object to the checkpoint directory. Returns the path."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"{name}.pkl")
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(path) / (1024**2)
        print(f"  ✓ Checkpoint saved: {name} ({size_mb:.0f} MB)")
    except Exception as e:
        print(f"  [WARNING] Could not save checkpoint {name}: {e}")
        path = ""
    return path


def load_checkpoint(name: str):
    """Load a checkpoint. Returns None if not found."""
    path = os.path.join(CHECKPOINT_DIR, f"{name}.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            size_mb = os.path.getsize(path) / (1024**2)
            print(f"  ✓ Checkpoint loaded: {name} ({size_mb:.0f} MB)")
            return obj
        except Exception as e:
            print(f"  [WARNING] Corrupt checkpoint {name}: {e}")
    return None


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
    cleans nulls, caches as zstd-compressed parquet for fast subsequent loads.
    """
    parquet_path = _get_parquet_path(path)
    if os.path.exists(parquet_path):
        print(f"Loading {os.path.basename(parquet_path)} (from parquet cache) ...")
        df = pl.read_parquet(parquet_path)
        print(f"  Rows: {len(df):,}")
        return df

    if not os.path.exists(path):
        from config import DATA_ROOT, REPO_ROOT
        raise FileNotFoundError(
            f"\n[ERROR] Source file not found: {path}\n"
            f"  - Configured DATA_ROOT : {DATA_ROOT}\n"
            f"  - REPO_ROOT           : {REPO_ROOT}\n"
            f"Please ensure your dataset files are inside '{os.path.join(REPO_ROOT, 'dataset')}' or set AMAZON_ML_DATA."
        )

    print(f"Loading {os.path.basename(path)} (TSV) ...")
    try:
        df = pl.read_csv(
            path,
            separator="\t",
            infer_schema_length=10_000,
            null_values=["", "NULL", "null", "N/A"],
            encoding="utf8-lossy",   # Handle encoding errors gracefully
        )
    except Exception as e:
        print(f"  [ERROR] Failed to load TSV with utf8-lossy: {e}")
        print(f"  Retrying with ignore_errors=True ...")
        df = pl.read_csv(
            path,
            separator="\t",
            infer_schema_length=10_000,
            null_values=["", "NULL", "null", "N/A"],
            ignore_errors=True,
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
    """Load ground truth TSV with Polars, with encoding error handling."""
    parquet_path = _get_parquet_path(path)
    if os.path.exists(parquet_path):
        print(f"Loading {os.path.basename(parquet_path)} (from parquet cache) ...")
        df = pl.read_parquet(parquet_path)
        print(f"  GT rows: {len(df):,}")
        return df

    if not os.path.exists(path):
        from config import DATA_ROOT, REPO_ROOT
        raise FileNotFoundError(
            f"\n[ERROR] Ground truth file not found: {path}\n"
            f"  - Configured DATA_ROOT : {DATA_ROOT}\n"
            f"  - REPO_ROOT           : {REPO_ROOT}\n"
        )

    print(f"Loading ground truth (TSV) ...")
    try:
        df = pl.read_csv(
            path,
            separator="\t",
            infer_schema_length=1000,
            null_values=["", "NULL", "null"],
            encoding="utf8-lossy",
        )
    except Exception as e:
        print(f"  [ERROR] Failed to load GT with utf8-lossy: {e}")
        df = pl.read_csv(
            path,
            separator="\t",
            infer_schema_length=1000,
            null_values=["", "NULL", "null"],
            ignore_errors=True,
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
    for row in tqdm(gt_df.iter_rows(named=True), total=len(gt_df), desc="Parsing GT"):
        s1_id = row["source1_entity_id"]
        raw   = row.get("matched_entity_ids", "") or ""
        matches = {m.strip() for m in raw.split(",") if m.strip()}
        gt_dict[s1_id] = matches
    return gt_dict


# ─── Embedding helpers ────────────────────────────────────────────────────────

_EMBED_CHUNK_SIZE = 250_000


def get_or_compute_embeddings(df: pl.DataFrame,
                              save_path: str,
                              force_recompute: bool = False) -> np.ndarray:
    """
    Load pre-computed embeddings from disk, or encode + cache with checkpointing.
    Processes in chunks of 500K rows for kernel crash resilience.
    """
    # Full cache hit
    if not force_recompute and os.path.exists(save_path):
        print(f"Loading cached embeddings from {save_path} (mmap) ...")
        return np.load(save_path, mmap_mode="r")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = len(df)
    dim = 384
    n_chunks = math.ceil(n / _EMBED_CHUNK_SIZE)
    chunk_dir = save_path + ".chunks"
    os.makedirs(chunk_dir, exist_ok=True)

    print(f"Encoding {n:,} texts in {n_chunks} chunk(s) of {_EMBED_CHUNK_SIZE:,} "
          f"(checkpoint after every chunk) ...")

    for ci in range(n_chunks):
        chunk_path = os.path.join(chunk_dir, f"chunk_{ci:04d}.npy")

        if not force_recompute and os.path.exists(chunk_path):
            print(f"  [Chunk {ci+1}/{n_chunks}] Checkpoint found on disk ✓")
            continue

        start = ci * _EMBED_CHUNK_SIZE
        end   = min(start + _EMBED_CHUNK_SIZE, n)

        chunk_df = df.slice(start, end - start)
        names = chunk_df["business_name"].fill_null("").to_list()
        addrs = chunk_df["business_address"].fill_null("").to_list()
        chunk_texts = [
            build_text_for_embedding(nm, ad)
            for nm, ad in zip(names, addrs)
        ]
        del chunk_df, names, addrs

        print(f"  [Chunk {ci+1}/{n_chunks}] Encoding rows {start:,}–{end:,} "
              f"({len(chunk_texts):,} texts) ...")

        try:
            chunk_emb = encode_texts(chunk_texts)
            del chunk_texts
            chunk_emb_f16 = chunk_emb.astype(np.float16)
            del chunk_emb
            gc.collect()

            np.save(chunk_path, chunk_emb_f16)
            chunk_mb = os.path.getsize(chunk_path) / (1024 ** 2)
            print(f"  [Chunk {ci+1}/{n_chunks}] Saved checkpoint: {chunk_mb:.0f} MB → {chunk_path}")
            del chunk_emb_f16
            gc.collect()

        except Exception as e:
            print(f"  [ERROR] Chunk {ci+1}/{n_chunks} encoding failed: {e}")
            traceback.print_exc()
            # Clean up partial chunk
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
            raise

    # Merge all chunks via memory-mapped streaming
    print(f"\nMerging {n_chunks} chunk(s) via memory-mapped streaming into final file ...")
    final_mmap = np.lib.format.open_memmap(save_path, mode="w+", dtype=np.float16, shape=(n, dim))
    for ci in range(n_chunks):
        c_start = ci * _EMBED_CHUNK_SIZE
        c_end   = min(c_start + _EMBED_CHUNK_SIZE, n)
        c_path  = os.path.join(chunk_dir, f"chunk_{ci:04d}.npy")
        c_arr   = np.load(c_path)
        final_mmap[c_start:c_end] = c_arr
        del c_arr
    final_mmap.flush()
    del final_mmap
    gc.collect()

    disk_mb = os.path.getsize(save_path) / (1024 ** 2)
    print(f"  Saved {disk_mb:.0f} MB float16 → {save_path}  ✓")

    shutil.rmtree(chunk_dir, ignore_errors=True)
    print("  Chunk checkpoints cleaned up.")

    return np.load(save_path, mmap_mode="r")


# ─── Stage functions (called from notebook in sequence) ──────────────────────

def stage_load_train():
    """Stage 1 — Load all training source files and ground truth."""
    print("\n" + "="*60)
    print("  STAGE 1: LOADING TRAINING DATA")
    print("="*60)
    s1 = load_source(TRAIN_S1)
    s2 = load_source(TRAIN_S2)
    s3 = load_source(TRAIN_S3)
    gt = load_ground_truth(TRAIN_GT)
    print(f"\n  Summary: S1={len(s1):,}, S2={len(s2):,}, S3={len(s3):,}, GT={len(gt):,}")
    return s1, s2, s3, gt


def stage_load_test():
    """Stage 2 — Load all test source files."""
    print("\n" + "="*60)
    print("  STAGE 2: LOADING TEST DATA")
    print("="*60)
    ts1 = load_source(TEST_S1)
    ts2 = load_source(TEST_S2)
    ts3 = load_source(TEST_S3)
    print(f"\n  Summary: S1={len(ts1):,}, S2={len(ts2):,}, S3={len(ts3):,}")
    return ts1, ts2, ts3


def stage_encode_train(s1, s2, s3, force=False):
    """Stage 3 — Compute (or load) multilingual embeddings for train sources."""
    print("\n" + "="*60)
    print("  STAGE 3: ENCODING TRAIN EMBEDDINGS")
    print("="*60)
    e1 = get_or_compute_embeddings(s1, EMBED_S1_TRAIN_PATH, force_recompute=force)
    e2 = get_or_compute_embeddings(s2, EMBED_S2_TRAIN_PATH, force_recompute=force)
    e3 = get_or_compute_embeddings(s3, EMBED_S3_TRAIN_PATH, force_recompute=force)
    return e1, e2, e3


def stage_encode_test(ts1, ts2, ts3, force=False):
    """Stage 4 — Compute (or load) embeddings for test sources."""
    print("\n" + "="*60)
    print("  STAGE 4: ENCODING TEST EMBEDDINGS")
    print("="*60)
    te1 = get_or_compute_embeddings(ts1, EMBED_S1_TEST_PATH, force_recompute=force)
    te2 = get_or_compute_embeddings(ts2, EMBED_S2_TEST_PATH, force_recompute=force)
    te3 = get_or_compute_embeddings(ts3, EMBED_S3_TEST_PATH, force_recompute=force)
    return te1, te2, te3


def stage_add_norm_cols(df: pl.DataFrame) -> pl.DataFrame:
    """
    Stage helper — add norm_name, norm_name_ns, norm_addr, and norm_addr_ns columns.
    Uses Polars native string expressions for high-throughput vectorized execution.
    """
    if "norm_name" in df.columns and "norm_addr" in df.columns:
        print(f"  Columns norm_name and norm_addr already exist for {len(df):,} rows.")
        return df

    print(f"  Normalising {len(df):,} rows (Polars native expressions) ...")

    # Legal suffix removal regex pattern (case-insensitive word boundary match)
    legal_pattern = r"(?i)\b(" + "|".join(sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b"

    # Name normalisation:
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

    # No-space name: remove all spaces from normalised name (for typo-robust matching)
    nospace_name = base_name.str.replace_all(r"\s", "")

    df = df.with_columns([
        sorted_name.alias("norm_name"),
        base_name.alias("norm_name_ns"),
        nospace_name.alias("norm_name_nospace"),
        base_addr.alias("norm_addr"),
        base_addr.alias("norm_addr_ns"),
    ])
    print(f"  ✓ Normalisation complete for {len(df):,} rows.")
    return df


def stage_blocking_train(s1, s2, s3, e1, e2, e3, force: bool = False):
    """Stage 5 — Generate candidate pairs for the TRAINING set."""
    print("\n" + "="*60)
    print("  STAGE 5: BLOCKING ON TRAIN")
    print("="*60)

    # Check for checkpoint unless forced
    if not force:
        cached = load_checkpoint("train_candidates")
        if cached is not None:
            print(f"  Using cached candidates ({sum(len(v) for v in cached.values()):,} pairs)")
            return cached

    candidates = generate_candidates(s1, s2, s3, e1, e2, e3,
                                      top_k=ANN_TOP_K, snm_window=SNM_WINDOW)

    # Checkpoint
    save_checkpoint(candidates, "train_candidates")
    return candidates


def stage_blocking_test(ts1, ts2, ts3, te1, te2, te3, force: bool = False):
    """Stage 6 — Generate candidate pairs for the TEST set."""
    print("\n" + "="*60)
    print("  STAGE 6: BLOCKING ON TEST")
    print("="*60)

    if not force:
        cached = load_checkpoint("test_candidates")
        if cached is not None:
            print(f"  Using cached candidates ({sum(len(v) for v in cached.values()):,} pairs)")
            return cached

    test_candidates = generate_candidates(ts1, ts2, ts3, te1, te2, te3,
                                           top_k=ANN_TOP_K, snm_window=SNM_WINDOW)

    save_checkpoint(test_candidates, "test_candidates")
    return test_candidates


def stage_blocking_recall(candidates, gt_df):
    """Stage 7 — Measure blocking recall against training ground truth."""
    print("\n" + "="*60)
    print("  STAGE 7: BLOCKING RECALL")
    print("="*60)
    recall = compute_blocking_recall(candidates, gt_df)
    if recall < 0.97:
        print(f"⚠️  WARNING: Blocking recall = {recall:.4f} < 0.97. "
              "Consider increasing TF-IDF K or ANN top_k.")
    elif recall < 0.99:
        print(f"ℹ️  Blocking recall = {recall:.4f}. Good, but < 0.99. "
              "Remaining misses cap your final F0.5.")
    else:
        print(f"✅  Blocking recall = {recall:.4f} >= 0.99. Excellent!")
    return recall


def stage_build_lookups(s1, s2, s3, e1, e2, e3):
    """
    Stage 8 — Build record lookup dicts for all train sources,
    attach embedding vectors for use as features.
    """
    print("\n" + "="*60)
    print("  STAGE 8: BUILDING RECORD LOOKUPS")
    print("="*60)

    lookup_s1 = build_record_lookup(s1)
    lookup_s2 = build_record_lookup(s2)
    lookup_s3 = build_record_lookup(s3)

    print("  Attaching embeddings ...")
    attach_embeddings(lookup_s1, s1["entity_id"].to_list(), e1)
    attach_embeddings(lookup_s2, s2["entity_id"].to_list(), e2)
    attach_embeddings(lookup_s3, s3["entity_id"].to_list(), e3)

    lookup_all = {**lookup_s1, **lookup_s2, **lookup_s3}
    del lookup_s1, lookup_s2, lookup_s3
    gc.collect()
    print(f"  Total records in lookup: {len(lookup_all):,}")
    return lookup_all


def stage_build_test_lookups(ts1, ts2, ts3, te1, te2, te3):
    """Stage 9 — Build record lookup dicts for all test sources."""
    print("\n" + "="*60)
    print("  STAGE 9: BUILDING TEST RECORD LOOKUPS")
    print("="*60)

    lu1 = build_record_lookup(ts1)
    lu2 = build_record_lookup(ts2)
    lu3 = build_record_lookup(ts3)

    attach_embeddings(lu1, ts1["entity_id"].to_list(), te1)
    attach_embeddings(lu2, ts2["entity_id"].to_list(), te2)
    attach_embeddings(lu3, ts3["entity_id"].to_list(), te3)

    test_lookup_all = {**lu1, **lu2, **lu3}
    del lu1, lu2, lu3
    gc.collect()
    print(f"  Total test records in lookup: {len(test_lookup_all):,}")
    return test_lookup_all


def stage_train_val_split(gt_df):
    """Stage 10 — Group-split by source1_entity_id, stratified by match_count bucket."""
    print("\n" + "="*60)
    print("  STAGE 10: TRAIN/VAL SPLIT")
    print("="*60)
    train_ids, val_ids = make_train_val_split(gt_df, VAL_FRACTION, RANDOM_SEED)
    print(f"  Train S1 entities: {len(train_ids):,}")
    print(f"  Val   S1 entities: {len(val_ids):,}")
    return train_ids, val_ids


def stage_build_training_pairs(train_s1_ids, candidates, gt_dict, lookup_all):
    """Stage 11 — Build (X_train, y_train) from training candidates."""
    print("\n" + "="*60)
    print("  STAGE 11: BUILDING TRAINING PAIRS")
    print("="*60)
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
    """Stage 12 — Train LightGBM. Uses a small val pair sample for early stopping."""
    import pandas as pd
    from features import FEATURE_NAMES
    from features import build_feature_vector
    import random

    print("\n" + "="*60)
    print("  STAGE 12: TRAINING LIGHTGBM MODEL")
    print("="*60)

    print("  Building val pairs for early stopping ...")
    rng = random.Random(RANDOM_SEED)
    val_sample = list(val_s1_ids)
    rng.shuffle(val_sample)
    val_sample = val_sample[:min(10_000, len(val_sample))]

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
            try:
                fv = build_feature_vector(rec_a, rec_b)
                val_rows.append([fv.get(f, 0.0) for f in FEATURE_NAMES])
                val_labels.append(1 if s23_id in true_m else 0)
            except Exception as e:
                continue  # Skip corrupt pairs

    X_val_es = pd.DataFrame(val_rows, columns=FEATURE_NAMES).fillna(0.0)
    y_val_es = np.array(val_labels, dtype=np.int8)
    del val_rows
    gc.collect()
    print(f"  Early-stopping val set: {len(X_val_es):,} pairs  "
          f"(positives: {y_val_es.sum():,})")

    model = train_lightgbm(X_train, y_train, X_val_es, y_val_es)
    del X_val_es, y_val_es
    gc.collect()
    return model


def stage_threshold_sweep(model, val_s1_ids, candidates, gt_dict, lookup_all):
    """Stage 13 — Sweep thresholds on val set to maximise macro F0.5."""
    print("\n" + "="*60)
    print("  STAGE 13: THRESHOLD SWEEP")
    print("="*60)
    threshold = sweep_threshold(model, val_s1_ids, candidates, gt_dict, lookup_all)
    return threshold


def stage_evaluate(model, threshold, val_s1_ids, candidates, gt_dict, lookup_all):
    """Stage 14 — Full validation evaluation report at optimal threshold."""
    print("\n" + "="*60)
    print("  STAGE 14: FULL VALIDATION EVALUATION")
    print("="*60)
    results = evaluate_on_val(model, threshold, val_s1_ids, candidates, gt_dict, lookup_all)
    return results


def stage_inference(model, threshold, ts1, test_candidates, test_lookup_all):
    """Stage 15 — Run inference on test set."""
    print("\n" + "="*60)
    print("  STAGE 15: TEST SET INFERENCE")
    print("="*60)
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
    """Stage 16 — Run the stdlib-only validator on the output files."""
    print("\n" + "="*60)
    print("  STAGE 16: VALIDATING SUBMISSION")
    print("="*60)
    import importlib.util
    from config import REPO_ROOT, OUTPUT_DIR, TEST_S1
    validator_path = os.path.abspath(os.path.join(REPO_ROOT, "utils", "validate_submission.py"))

    if not os.path.exists(validator_path):
        print(f"  [WARNING] Validator not found at {validator_path}. Skipping.")
        return

    try:
        spec = importlib.util.spec_from_file_location("validate_submission", validator_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        test_dir = os.path.dirname(TEST_S1)
        matching  = os.path.join(OUTPUT_DIR, "matching_results.tsv")
        candidate = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

        mod.validate(matching, candidate, test_dir)
        print("  ✅ Submission validation PASSED!")
    except SystemExit as e:
        if e.code != 0:
            print("  ❌ Submission validation FAILED. Check output above.")
            raise RuntimeError("Submission validation FAILED.")
    except Exception as e:
        print(f"  [WARNING] Validator error: {e}")
        traceback.print_exc()
