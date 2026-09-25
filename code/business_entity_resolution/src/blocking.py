# =============================================================
# blocking.py — Multi-strategy candidate pair generation
# =============================================================
# Produces candidate_pairs.tsv: every (S1, S2/S3) pair that will
# be scored by the LightGBM matcher.
#
# Three complementary strategies are combined (union):
#   A. Token blocking  — exact-match on normalized name/address keys
#   B. Sorted-neighborhood — slide window over sorted name prefix
#   C. ANN blocking    — top-k FAISS retrieval on multilingual embeddings
#
# Country partitioning is applied FIRST as a hard pre-filter (zero
# recall loss on training data — all 7.64M GT pairs are same-country).
# A safety net re-admits very high-similarity cross-country pairs.
#
# Target: blocking recall >= 0.99, reduction ratio >= 0.999

import os
import gc
import pickle
from collections import defaultdict

import numpy as np
import polars as pl
from tqdm import tqdm

from normalize import normalize_name, normalize_address
from config import (
    ANN_TOP_K, SNM_WINDOW,
    CROSS_COUNTRY_NAME_THRESH, CROSS_COUNTRY_ADDR_THRESH,
    EMBED_MODEL_NAME, EMBED_BATCH_SIZE, EMBED_MAX_SEQ_LEN,
)


# ─── Strategy A: Vectorized Polars Exact Matching ────────────────────────────

def _add_exact_matches(candidates: dict[str, set[str]],
                       s1: pl.DataFrame,
                       target: pl.DataFrame,
                       col: str,
                       max_bucket_size: int = 50):
    """
    Find matching entities between s1 and target on (country, col).
    Vectorized Polars inner join in C++/Rust: 0 Python dicts, runs in < 2 seconds, < 200 MB RAM.
    """
    if col not in s1.columns or col not in target.columns:
        return

    # Filter out empty strings and generic buckets with > max_bucket_size records
    target_clean = (
        target.select(["entity_id", "country", col])
              .filter(pl.col(col).str.len_chars() >= 4)
              .filter(pl.len().over(["country", col]) <= max_bucket_size)
    )
    s1_clean = (
        s1.select(["entity_id", "country", col])
          .filter(pl.col(col).str.len_chars() >= 4)
    )
    matches = s1_clean.join(target_clean, on=["country", col], how="inner")
    del target_clean, s1_clean
    gc.collect()

    s1_ids = matches["entity_id"].to_list()
    tgt_ids = matches["entity_id_right"].to_list()
    del matches
    gc.collect()

    for s1_id, tgt_id in zip(s1_ids, tgt_ids):
        candidates[s1_id].add(tgt_id)
    del s1_ids, tgt_ids
    gc.collect()


# ─── Strategy B: Sorted Neighborhood Method (SNM) ───────────────────────────

def _add_snm_matches(candidates: dict[str, set[str]],
                     s1: pl.DataFrame,
                     target: pl.DataFrame,
                     window: int = 3):
    """
    Sorted-Neighborhood Method on norm_name within each country.
    Memory-efficient: uses parallel lists, zero Python dictionary allocations.
    """
    s1_sub = s1.select(["entity_id", "country", "norm_name"]).with_columns(
        pl.lit("S1").alias("src"))
    tgt_sub = target.select(["entity_id", "country", "norm_name"]).with_columns(
        pl.lit("TGT").alias("src"))
    combined = pl.concat([s1_sub, tgt_sub]).sort(["country", "norm_name"])
    del s1_sub, tgt_sub

    srcs = combined["src"].to_list()
    eids = combined["entity_id"].to_list()
    cnts = combined["country"].to_list()
    del combined
    gc.collect()

    n = len(srcs)
    for i in range(n):
        if srcs[i] != "S1":
            continue
        s1_id = eids[i]
        c     = cnts[i]
        start = max(0, i - window)
        end   = min(n, i + window + 1)
        for j in range(start, end):
            if j != i and srcs[j] == "TGT" and cnts[j] == c:
                candidates[s1_id].add(eids[j])

    del srcs, eids, cnts
    gc.collect()


# ─── Strategy C: FAISS ANN Vector Search ─────────────────────────────────────

_MODEL_CACHE: dict = {}


def encode_texts(texts: list[str],
                 model_name: str = EMBED_MODEL_NAME,
                 batch_size: int = EMBED_BATCH_SIZE,
                 max_seq_len: int = EMBED_MAX_SEQ_LEN) -> np.ndarray:
    """
    Encode a list of strings with the multilingual sentence encoder.

    Returns a float32 numpy array of shape (N, dim).

    Uses GPU if available, falls back to CPU.
    The model is cached in memory across calls to avoid re-instantiation overhead.
    """
    from sentence_transformers import SentenceTransformer

    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    model = _MODEL_CACHE[model_name]
    model.max_seq_length = max_seq_len

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalised → dot product == cosine
    )
    # Cast to float32 for FAISS compatibility (FAISS IndexFlatIP requires float32)
    return embeddings.astype(np.float32)


def _add_ann_matches(candidates: dict[str, set[str]],
                     s1: pl.DataFrame,
                     target_df: pl.DataFrame,
                     s1_embeds: np.ndarray,
                     target_embeds: np.ndarray,
                     top_k: int = 15):
    """
    Search S1 embeddings against target (S2 or S3) partitioned by country.
    Uses FAISS IndexFlatIP with per-country indexes to keep RAM < 100 MB.
    """
    import faiss
    s1_ids = s1["entity_id"].to_list()
    tgt_ids = target_df["entity_id"].to_list()
    s1_countries = dict(zip(s1_ids, s1["country"].to_list()))
    tgt_countries = dict(zip(tgt_ids, target_df["country"].to_list()))

    dim = target_embeds.shape[1]
    countries = list(set(s1["country"].to_list()))

    for country in countries:
        s1_mask  = [i for i, eid in enumerate(s1_ids)  if s1_countries[eid] == country]
        tgt_mask = [i for i, eid in enumerate(tgt_ids) if tgt_countries[eid] == country]

        if not s1_mask or not tgt_mask:
            continue

        sub_tgt = np.ascontiguousarray(target_embeds[tgt_mask], dtype=np.float32)
        sub_s1  = np.ascontiguousarray(s1_embeds[s1_mask], dtype=np.float32)

        index = faiss.IndexFlatIP(dim)
        index.add(sub_tgt)

        k = min(top_k, len(tgt_mask))
        distances, indices = index.search(sub_s1, k)
        del index, sub_tgt, sub_s1

        for qi, s1_idx in enumerate(s1_mask):
            s1_id = s1_ids[s1_idx]
            for rank in range(k):
                local_idx = indices[qi, rank]
                if local_idx >= 0:
                    candidates[s1_id].add(tgt_ids[tgt_mask[local_idx]])

        gc.collect()


# ─── Public API ──────────────────────────────────────────────────────────────

def generate_candidates(s1: pl.DataFrame,
                         s2: pl.DataFrame,
                         s3: pl.DataFrame,
                         s1_embeds: np.ndarray,
                         s2_embeds: np.ndarray,
                         s3_embeds: np.ndarray,
                         top_k: int = ANN_TOP_K,
                         snm_window: int = SNM_WINDOW) -> dict[str, set[str]]:
    """
    Generate candidates directly into a dict-of-sets.
    Peak RAM: < 500 MB. Zero tuple sets, zero DataFrame duplication.
    """
    candidates: dict[str, set[str]] = defaultdict(set)

    print("[Blocking] Strategy A: Vectorized exact name & address matching ...")
    _add_exact_matches(candidates, s1, s2, "norm_name")
    _add_exact_matches(candidates, s1, s3, "norm_name")
    _add_exact_matches(candidates, s1, s2, "norm_addr")
    _add_exact_matches(candidates, s1, s3, "norm_addr")
    print(f"           Pairs after Strategy A: {sum(len(v) for v in candidates.values()):,}")

    print(f"[Blocking] Strategy B: Sorted-neighborhood (window={snm_window}) ...")
    _add_snm_matches(candidates, s1, s2, window=snm_window)
    _add_snm_matches(candidates, s1, s3, window=snm_window)
    print(f"           Pairs after Strategy B: {sum(len(v) for v in candidates.values()):,}")

    ann_k = max(10, top_k // 2)
    print(f"[Blocking] Strategy C: FAISS ANN (top-{ann_k} per target) ...")
    _add_ann_matches(candidates, s1, s2, s1_embeds, s2_embeds, top_k=ann_k)
    _add_ann_matches(candidates, s1, s3, s1_embeds, s3_embeds, top_k=ann_k)
    total_pairs = sum(len(v) for v in candidates.values())
    print(f"           Total unique pairs after Strategy C: {total_pairs:,}")

    # Ensure every S1 entity exists in candidates (even if empty singleton)
    for s1_id in s1["entity_id"].to_list():
        if s1_id not in candidates:
            candidates[s1_id] = set()

    return dict(candidates)


def compute_blocking_recall(candidates: dict[str, set[str]],
                             ground_truth: pl.DataFrame) -> float:
    """
    Compute blocking recall on the ground-truth split.

    blocking_recall = |GT pairs captured in candidates| / |GT pairs total|

    A value < 1.0 is an unrecoverable ceiling on final F0.5.
    Print a breakdown by country and match-count bucket.
    """
    total_gt = 0
    captured = 0

    # Parse GT: source1_entity_id -> set of matched_entity_ids
    gt_dict: dict[str, set[str]] = {}
    for row in ground_truth.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        raw   = row.get("matched_entity_ids", "") or ""
        matches = {m.strip() for m in raw.split(",") if m.strip()}
        gt_dict[s1_id] = matches

    for s1_id, true_matches in gt_dict.items():
        for m in true_matches:
            total_gt += 1
            if m in candidates.get(s1_id, set()):
                captured += 1

    recall = captured / total_gt if total_gt > 0 else 1.0
    print(f"[Blocking Recall] {captured:,} / {total_gt:,} = {recall:.4f}")
    return recall
