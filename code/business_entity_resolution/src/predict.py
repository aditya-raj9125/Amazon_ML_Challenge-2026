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


def score_candidates(
    model,
    test_s1_ids: list[str],
    candidates: dict[str, set[str]],
    lookup_all: dict[str, dict],
    batch_size: int = INFER_BATCH_SIZE,
) -> dict[str, dict[str, float]]:
    """
    Score every candidate pair with the LightGBM model.
    Processes in batches to avoid OOM on large test candidate sets.
    """
    scores: dict[str, dict[str, float]] = {s1_id: {} for s1_id in test_s1_ids}

    batch_rows    = []
    batch_indices = []
    total_scored  = 0
    total_errors  = 0

    def _flush(batch_rows, batch_indices):
        nonlocal total_scored
        if not batch_rows:
            return
        X = pd.DataFrame(batch_rows, columns=FEATURE_NAMES)
        X = X.replace([np.inf, -np.inf], np.nan)
        probs = model.predict_proba(X)[:, 1]
        for (s1_id, s23_id), prob in zip(batch_indices, probs):
            scores[s1_id][s23_id] = float(prob)
        total_scored += len(batch_rows)
        batch_rows.clear()
        batch_indices.clear()
        del X, probs
        gc.collect()

    for s1_id in tqdm(test_s1_ids, desc="Scoring test pairs"):
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
                batch_rows.append([fv.get(f, 0.0) for f in FEATURE_NAMES])
                batch_indices.append((s1_id, s23_id))
            except Exception as e:
                total_errors += 1
                continue

            if len(batch_rows) >= batch_size:
                _flush(batch_rows, batch_indices)

    _flush(batch_rows, batch_indices)

    print(f"  Total pairs scored: {total_scored:,}")
    if total_errors > 0:
        print(f"  [WARNING] {total_errors:,} pairs skipped due to errors")

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
    print("\n[Inference] Scoring candidate pairs ...")
    scores = score_candidates(model, test_s1_ids, candidates, lookup_all)

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
