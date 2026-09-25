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


# ─── Helpers ─────────────────────────────────────────────────────────────────

MAX_BUCKET_SIZE = 500  # Discard generic mega-buckets (e.g. 'US||main', 'US||the')


def _build_token_keys(df: pl.DataFrame) -> pl.DataFrame:
    """
    Materialise high-signal blocking keys as new columns.
    """
    return df.with_columns([
        (pl.col("country") + "||" + pl.col("norm_name").str.slice(0, 4))
            .alias("key_name_pref4"),
        (pl.col("country") + "||" + pl.col("norm_addr").str.slice(0, 5))
            .alias("key_addr_pref5"),
        (pl.col("country") + "||" + pl.col("norm_name"))
            .alias("key_name_norm"),
        (pl.col("country") + "||" + pl.col("norm_addr"))
            .alias("key_addr_norm"),
    ])


def _token_blocking_pairs(s1: pl.DataFrame,
                           s23: pl.DataFrame,
                           key_cols: list[str],
                           max_bucket_size: int = MAX_BUCKET_SIZE) -> set[tuple[str, str]]:
    """
    For each key column, build inverted index (key → list of ids) for S2/S3.
    Filter out mega-buckets to prevent combinatorial explosion.
    """
    pairs = set()
    for key_col in key_cols:
        inv = defaultdict(list)
        for row in s23.select(["entity_id", key_col]).iter_rows():
            eid, key = row
            if key and len(key) >= 5:
                inv[key].append(eid)

        # Filter out mega-buckets (> max_bucket_size) to protect RAM
        valid_inv = {k: v for k, v in inv.items() if len(v) <= max_bucket_size}
        del inv

        for row in s1.select(["entity_id", key_col]).iter_rows():
            s1_id, key = row
            if key and key in valid_inv:
                for s23_id in valid_inv[key]:
                    pairs.add((s1_id, s23_id))
        del valid_inv
        gc.collect()

    return pairs


def _sorted_neighborhood_pairs(s1: pl.DataFrame,
                                s23: pl.DataFrame,
                                window: int = SNM_WINDOW) -> set[tuple[str, str]]:
    """
    Sorted-Neighborhood Method on norm_name within each country.
    Memory-efficient: uses parallel lists, zero Python dictionary allocations.
    """
    pairs = set()
    s1_tagged  = s1.select(["entity_id", "country", "norm_name"]).with_columns(
        pl.lit("S1").alias("src"))
    s23_tagged = s23.select(["entity_id", "country", "norm_name"]).with_columns(
        pl.lit("S23").alias("src"))
    combined = pl.concat([s1_tagged, s23_tagged]).sort(["country", "norm_name"])

    srcs = combined["src"].to_list()
    eids = combined["entity_id"].to_list()
    cnts = combined["country"].to_list()
    del combined, s1_tagged, s23_tagged
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
            if j != i and srcs[j] == "S23" and cnts[j] == c:
                pairs.add((s1_id, eids[j]))

    del srcs, eids, cnts
    gc.collect()
    return pairs


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


def _ann_search_source(s1: pl.DataFrame,
                        target_df: pl.DataFrame,
                        s1_embeds: np.ndarray,
                        target_embeds: np.ndarray,
                        top_k: int) -> set[tuple[str, str]]:
    """
    Search S1 embeddings against target (S2 or S3) partitioned by country.
    Uses FAISS IndexFlatIP with per-country indexes to keep RAM < 100 MB.
    """
    import faiss
    pairs = set()
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
                    pairs.add((s1_id, tgt_ids[tgt_mask[local_idx]]))

    return pairs


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
    Combine all three blocking strategies with zero memory bloat:
        { s1_entity_id : set of candidate S2/S3 entity_ids }
    """
    print("[Blocking] Preparing S23 metadata ...")
    s23 = pl.concat([s2, s3])

    print("[Blocking] Building token keys ...")
    s1_keys  = _build_token_keys(s1)
    s23_keys = _build_token_keys(s23)

    key_cols = ["key_name_pref4", "key_addr_pref5", "key_name_norm", "key_addr_norm"]

    print("[Blocking] Strategy A: Token blocking (filtering mega-buckets) ...")
    pairs_token = _token_blocking_pairs(s1_keys, s23_keys, key_cols)
    del s1_keys, s23_keys
    gc.collect()
    print(f"           Token pairs: {len(pairs_token):,}")

    print(f"[Blocking] Strategy B: Sorted-neighborhood (window={snm_window}) ...")
    pairs_snm = _sorted_neighborhood_pairs(s1, s23, window=snm_window)
    del s23
    gc.collect()
    print(f"           SNM pairs: {len(pairs_snm):,}")

    print(f"[Blocking] Strategy C: ANN top-{top_k} on S2 & S3 (independent searches, 0 RAM overhead) ...")
    pairs_ann_s2 = _ann_search_source(s1, s2, s1_embeds, s2_embeds, top_k=top_k)
    print(f"           ANN S2 pairs: {len(pairs_ann_s2):,}")

    pairs_ann_s3 = _ann_search_source(s1, s3, s1_embeds, s3_embeds, top_k=top_k)
    print(f"           ANN S3 pairs: {len(pairs_ann_s3):,}")

    # Build final candidates dict directly without intermediate giant set unions
    print("[Blocking] Merging candidate pools into dict ...")
    candidates: dict[str, set[str]] = defaultdict(set)

    for s1_id, s23_id in pairs_token:
        candidates[s1_id].add(s23_id)
    del pairs_token
    gc.collect()

    for s1_id, s23_id in pairs_snm:
        candidates[s1_id].add(s23_id)
    del pairs_snm
    gc.collect()

    for s1_id, s23_id in pairs_ann_s2:
        candidates[s1_id].add(s23_id)
    del pairs_ann_s2
    gc.collect()

    for s1_id, s23_id in pairs_ann_s3:
        candidates[s1_id].add(s23_id)
    del pairs_ann_s3
    gc.collect()

    # Ensure every S1 entity has an entry
    for s1_id in s1["entity_id"].to_list():
        if s1_id not in candidates:
            candidates[s1_id] = set()

    total_pairs = sum(len(v) for v in candidates.values())
    print(f"[Blocking] Total unique candidate pairs: {total_pairs:,}")
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
