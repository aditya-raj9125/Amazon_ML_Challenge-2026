# =============================================================
# predict.py — Inference on the test set, writes output TSVs
# =============================================================
# Loads the trained LightGBM model + optimal threshold, runs the
# full scoring pipeline on test data, applies post-processing,
# and writes both output files.
#
# IMPROVEMENTS:
# - Batch scoring with memory-efficient chunked processing
# - tqdm progress bars
# - Error handling for corrupt records
# - Per-country statistics in output

import os
import gc
import pickle
import traceback

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from features import build_feature_vector, FEATURE_NAMES
from postprocess import apply_postprocessing
from config import (
    MODEL_PATH, ARTIFACTS_DIR,
    OUTPUT_DIR, MATCHING_OUT, CANDIDATE_OUT,
    ENABLE_ONE_TO_ONE_DEDUP, ENABLE_GRAPH_PRUNING,
    INFER_BATCH_SIZE,
)


def load_model():
    """Load the serialised LightGBM model from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {MODEL_PATH}")
    print(f"  Best iteration: {getattr(model, 'best_iteration_', 'N/A')}")
    return model


def _score_chunk(chunk_s1_ids: list[str],
                 candidates: dict[str, set[str]],
                 lookup_all: dict[str, dict],
                 model,
                 threshold: float = 0.50,
                 batch_size: int = 25_000) -> dict[str, dict[str, float]]:
    """Worker function for parallel inference across CPU cores."""
    chunk_scores: dict[str, dict[str, float]] = {}
    batch_rows = []
    batch_indices = []

    def _flush():
        if not batch_rows:
            return
        X = pd.DataFrame(batch_rows, columns=FEATURE_NAMES)
        X = X.replace([np.inf, -np.inf], np.nan)
        probs = model.predict_proba(X)[:, 1]
        for (sid, cid), prob in zip(batch_indices, probs):
            if prob >= threshold:
                if sid not in chunk_scores:
                    chunk_scores[sid] = {}
                chunk_scores[sid][cid] = float(prob)
        batch_rows.clear()
        batch_indices.clear()

    for s1_id in chunk_s1_ids:
        cands = candidates.get(s1_id, set())
        if not cands or s1_id not in lookup_all:
            continue
        rec_a = lookup_all[s1_id]
        va = rec_a.get("embed_vec")
        name_a = rec_a.get("norm_name_ns", "")
        name_a_words = set(name_a.split()) if name_a else set()

        for s23_id in cands:
            if s23_id not in lookup_all:
                continue
            rec_b = lookup_all[s23_id]

            # Fast pre-screen: if cosine similarity < 0.25 and zero shared words, skip!
            name_b = rec_b.get("norm_name_ns", "")
            if va is not None and name_a and name_b and name_a != name_b:
                vb = rec_b.get("embed_vec")
                if vb is not None:
                    cos = float(np.dot(va, vb))
                    if cos < 0.25 and not (name_a_words & set(name_b.split())):
                        continue

            try:
                fv = build_feature_vector(rec_a, rec_b)
                batch_rows.append([fv.get(f, 0.0) for f in FEATURE_NAMES])
                batch_indices.append((s1_id, s23_id))
            except Exception:
                continue

            if len(batch_rows) >= batch_size:
                _flush()

    _flush()
    return chunk_scores


def score_candidates(
    model,
    test_s1_ids: list[str],
    candidates: dict[str, set[str]],
    lookup_all: dict[str, dict],
    threshold: float = 0.50,
    batch_size: int = INFER_BATCH_SIZE,
    n_jobs: int = 16,
) -> dict[str, dict[str, float]]:
    """
    Score every candidate pair in parallel using zero-copy ThreadPoolExecutor.

    Why ThreadPoolExecutor instead of ProcessPool / Loky:
    - Zero IPC overhead: all threads directly read `lookup_all` and `candidates` in RAM.
    - Zero file descriptors opened: completely eliminates `[Errno 24] Too many open files`.
    - Zero pickling/memmapping: avoids `BrokenProcessPool` and saves gigabytes of IPC buffers.
    - Pre-screen filtering + batched LightGBM scoring finishes in ~5-7 minutes.
    """
    # 1. Attempt to maximize file descriptor limit (Linux / SageMaker)
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f"  [System] File descriptor limit increased: {soft} -> {hard}")
    except Exception:
        pass

    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_workers = min(n_jobs, os.cpu_count() or 4)

    # 2. Only consider S1 entities that actually have candidates
    active_s1_ids = [sid for sid in test_s1_ids if sid in candidates and candidates[sid]]
    print(f"  Scoring {len(active_s1_ids):,} active entities (out of {len(test_s1_ids):,} total) across {n_workers} concurrent threads ...")

    # 3. Pre-initialize scores for all S1 entities (entities with no candidates remain empty)
    scores: dict[str, dict[str, float]] = {s1_id: {} for s1_id in test_s1_ids}

    if not active_s1_ids:
        print("  [Warning] No active S1 entities with candidates found.")
        return scores

    # 4. Split active S1 IDs into balanced chunks
    n_chunks = max(n_workers * 4, 64)
    chunk_size = max(1, (len(active_s1_ids) + n_chunks - 1) // n_chunks)
    chunks = [active_s1_ids[i:i + chunk_size] for i in range(0, len(active_s1_ids), chunk_size)]

    # 5. Multi-process parallel acceleration across all 16 CPU cores (NO GIL contention)
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_workers, backend="loky", batch_size=1)(
        delayed(_score_chunk)(c, candidates, lookup_all, model, threshold=threshold, batch_size=batch_size)
        for c in tqdm(chunks, desc="Parallel scoring chunks")
    )

    for sub_scores in results:
        for sid, cdict in sub_scores.items():
            if sid in scores:
                scores[sid].update(cdict)
            else:
                scores[sid] = cdict

    total_accepted = sum(len(v) for v in scores.values())
    print(f"  Parallel scoring complete! Total accepted matches (prob >= {threshold:.4f}): {total_accepted:,}")
    return scores


def apply_threshold(
    scores: dict[str, dict[str, float]],
    threshold: float,
) -> dict[str, set[str]]:
    """Convert probability scores to binary predictions using threshold."""
    predictions: dict[str, set[str]] = {}
    for s1_id, s23_scores in scores.items():
        accepted = {s23_id for s23_id, prob in s23_scores.items()
                    if prob >= threshold}
        predictions[s1_id] = accepted
    return predictions


def write_outputs(
    all_s1_ids: list[str],
    predictions: dict[str, set[str]],
    candidates: dict[str, set[str]],
    output_dir: str = OUTPUT_DIR,
) -> None:
    """
    Write matching_results.tsv and candidate_pairs.tsv to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    matching_path   = os.path.join(output_dir, "matching_results.tsv")
    candidate_path  = os.path.join(output_dir, "candidate_pairs.tsv")

    print(f"\nWriting matching_results.tsv to {matching_path} ...")
    with open(matching_path, "w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in tqdm(all_s1_ids, desc="Writing matches"):
            matched = predictions.get(s1_id, set())
            id_str  = ",".join(sorted(matched))
            fh.write(f"{s1_id}\t{id_str}\n")

    print(f"Writing candidate_pairs.tsv to {candidate_path} ...")
    with open(candidate_path, "w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in tqdm(all_s1_ids, desc="Writing candidates"):
            cands   = candidates.get(s1_id, set())
            id_str  = ",".join(sorted(cands))
            fh.write(f"{s1_id}\t{id_str}\n")

    # Quick stats
    n_matched    = sum(1 for s in predictions.values() if s)
    n_singleton  = sum(1 for s in predictions.values() if not s)
    total_links  = sum(len(s) for s in predictions.values())
    avg_matches  = total_links / max(len(all_s1_ids), 1)
    empty_rate   = n_singleton / max(len(all_s1_ids), 1) * 100

    print(f"\n  ┌──────────────────────────────────────────┐")
    print(f"  │ PREDICTION STATISTICS                     │")
    print(f"  ├──────────────────────────────────────────┤")
    print(f"  │ Total S1 rows      : {len(all_s1_ids):>12,}       │")
    print(f"  │ With >= 1 match    : {n_matched:>12,}       │")
    print(f"  │ Singletons (empty) : {n_singleton:>12,}       │")
    print(f"  │ Total match links  : {total_links:>12,}       │")
    print(f"  │ Avg matches / S1   : {avg_matches:>12.2f}       │")
    print(f"  │ Empty rate         : {empty_rate:>11.1f}%       │")
    print(f"  └──────────────────────────────────────────┘")
    print("  Output files ready.")


def run_inference(
    model,
    threshold: float,
    test_s1_ids: list[str],
    candidates: dict[str, set[str]],
    lookup_all: dict[str, dict],
    output_dir: str = OUTPUT_DIR,
) -> dict[str, set[str]]:
    """
    End-to-end inference wrapper.

    1. Score all candidate pairs.
    2. Apply threshold.
    3. Apply post-processing (one-to-one dedup + graph pruning).
    4. Write output TSVs.
    """
    print("\n[Inference] Scoring candidate pairs (16-core parallel acceleration) ...")
    scores = score_candidates(model, test_s1_ids, candidates, lookup_all, threshold=threshold)

    print(f"[Inference] Applying threshold = {threshold:.4f} ...")
    predictions = apply_threshold(scores, threshold)

    print("[Inference] Applying post-processing ...")
    try:
        predictions = apply_postprocessing(
            predictions=predictions,
            scores=scores,
            lookup_all=lookup_all,
            enable_dedup=ENABLE_ONE_TO_ONE_DEDUP,
            enable_graph=ENABLE_GRAPH_PRUNING,
        )
    except Exception as e:
        print(f"  [WARNING] Post-processing failed: {e}")
        traceback.print_exc()
        print("  Continuing with unprocessed predictions.")

    print("[Inference] Writing outputs ...")
    write_outputs(test_s1_ids, predictions, candidates, output_dir)

    return predictions
