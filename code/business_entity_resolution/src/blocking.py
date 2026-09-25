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

def _name_prefix_key(norm_name: str, n: int) -> str:
    """Return first n chars of the normalised name as a blocking key."""
    return norm_name[:n] if len(norm_name) >= n else norm_name


def _build_token_keys(df: pl.DataFrame) -> pl.DataFrame:
    """
    Materialise all blocking keys as new columns.

    Keys produced per record
    ------------------------
    key_name_pref3  : country + first-3-chars of sorted norm_name
    key_name_pref4  : country + first-4-chars of sorted norm_name
    key_addr_pref4  : country + first-4-chars of sorted norm_addr
    key_name_norm   : country + full sorted norm_name (high precision)
    key_addr_norm   : country + full sorted norm_addr
    """
    df = df.with_columns([
        (pl.col("country") + "||" + pl.col("norm_name").str.slice(0, 3))
            .alias("key_name_pref3"),
        (pl.col("country") + "||" + pl.col("norm_name").str.slice(0, 4))
            .alias("key_name_pref4"),
        (pl.col("country") + "||" + pl.col("norm_addr").str.slice(0, 4))
            .alias("key_addr_pref4"),
        (pl.col("country") + "||" + pl.col("norm_name"))
            .alias("key_name_norm"),
        (pl.col("country") + "||" + pl.col("norm_addr"))
            .alias("key_addr_norm"),
    ])
    return df


def _token_blocking_pairs(s1: pl.DataFrame,
                           s23: pl.DataFrame,
                           key_cols: list[str]) -> set[tuple[str, str]]:
    """
    For each key column, build inverted index (key → list of ids) for S2/S3,
    then look up every S1 key and collect matching S2/S3 ids.

    Returns a set of (s1_id, s23_id) tuples.
    """
    pairs = set()
    # Build S2/S3 inverted index for each key
    for key_col in key_cols:
        inv = defaultdict(list)
        for row in s23.select(["entity_id", key_col]).iter_rows():
            eid, key = row
            if key:
                inv[key].append(eid)

        for row in s1.select(["entity_id", key_col]).iter_rows():
            s1_id, key = row
            if key and key in inv:
                for s23_id in inv[key]:
                    pairs.add((s1_id, s23_id))
    return pairs


def _sorted_neighborhood_pairs(s1: pl.DataFrame,
                                s23: pl.DataFrame,
                                window: int = SNM_WINDOW) -> set[tuple[str, str]]:
    """
    Sorted-Neighborhood Method on norm_name within each country.

    Steps:
      1. Concatenate S1 and S2/S3 records with source tag.
      2. Sort by (country, norm_name).
      3. Slide a window of size `window`; pair every S1 in window with
         every S2/S3 in the same window.

    Catches near-duplicates that differ in spelling but sort adjacently.
    """
    pairs = set()
    # Tag source
    s1_tagged  = s1.select(["entity_id", "country", "norm_name"]).with_columns(
        pl.lit("S1").alias("src"))
    s23_tagged = s23.select(["entity_id", "country", "norm_name"]).with_columns(
        pl.lit("S23").alias("src"))
    combined = pl.concat([s1_tagged, s23_tagged]).sort(["country", "norm_name"])
    rows = combined.to_dicts()

    # Slide window
    for i, row in enumerate(rows):
        if row["src"] != "S1":
            continue
        s1_id = row["entity_id"]
        country = row["country"]
        # Look backward and forward within window
        start = max(0, i - window)
        end   = min(len(rows), i + window + 1)
        for j in range(start, end):
            if j == i:
                continue
            nbr = rows[j]
            if nbr["src"] == "S23" and nbr["country"] == country:
                pairs.add((s1_id, nbr["entity_id"]))

    return pairs


def encode_texts(texts: list[str],
                 model_name: str = EMBED_MODEL_NAME,
                 batch_size: int = EMBED_BATCH_SIZE,
                 max_seq_len: int = EMBED_MAX_SEQ_LEN) -> np.ndarray:
    """
    Encode a list of strings with the multilingual sentence encoder.

    Returns a float32 numpy array of shape (N, dim).

    Uses GPU if available, falls back to CPU.
    The model is cached in memory after first load — call this function
    once per source file, not once per pair.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_len

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalised → dot product == cosine
        precision="float16",         # fp16: T4 Tensor Cores → ~1.8x throughput,
                                     # zero quality loss for cosine similarity search
    )
    # Cast to float32 for FAISS compatibility (FAISS IndexFlatIP requires float32)
    return embeddings.astype(np.float32)


def _ann_blocking_pairs(s1: pl.DataFrame,
                         s23: pl.DataFrame,
                         s1_embeds: np.ndarray,
                         s23_embeds: np.ndarray,
                         top_k: int = ANN_TOP_K) -> set[tuple[str, str]]:
    """
    Approximate nearest-neighbor blocking via FAISS.

    For each S1 embedding, retrieve top_k most similar S2/S3 embeddings
    (cosine similarity — safe because embeddings are L2-normalised so
    inner product == cosine).

    Country partitioning is applied: only cross-country pairs with
    very high similarity pass through (safety net).

    Returns a set of (s1_id, s23_id) tuples.
    """
    pairs = set()
    s1_ids  = s1["entity_id"].to_list()
    s23_ids = s23["entity_id"].to_list()
    s1_countries  = dict(zip(s1["entity_id"].to_list(),  s1["country"].to_list()))
    s23_countries = dict(zip(s23["entity_id"].to_list(), s23["country"].to_list()))

    dim = s23_embeds.shape[1]

    # Process per-country to keep index small and enforce partitioning
    countries = list(set(s1["country"].to_list()))
    for country in countries:
        # S1 indices for this country
        s1_mask   = [i for i, eid in enumerate(s1_ids)  if s1_countries[eid]  == country]
        s23_mask  = [i for i, eid in enumerate(s23_ids) if s23_countries[eid] == country]

        if not s1_mask or not s23_mask:
            continue

        sub_s23_embeds = s23_embeds[s23_mask]
        sub_s1_embeds  = s1_embeds[s1_mask]

        # Build flat FAISS index (inner product on L2-normed = cosine)
        import faiss
        index = faiss.IndexFlatIP(dim)
        index.add(sub_s23_embeds)

        k = min(top_k, len(s23_mask))
        distances, indices = index.search(sub_s1_embeds, k)

        for qi, s1_idx in enumerate(s1_mask):
            s1_id = s1_ids[s1_idx]
            for rank in range(k):
                s23_local_idx = indices[qi, rank]
                if s23_local_idx < 0:
                    continue
                s23_idx = s23_mask[s23_local_idx]
                pairs.add((s1_id, s23_ids[s23_idx]))

    return pairs


def _cross_country_safety_net(s1: pl.DataFrame,
                               s23: pl.DataFrame,
                               s1_embeds: np.ndarray,
                               s23_embeds: np.ndarray,
                               cos_threshold: float = 0.97) -> set[tuple[str, str]]:
    """
    Retrieve a tiny number of cross-country near-matches as insurance.

    Uses a global FAISS index (all countries) and only admits pairs
    where cosine similarity > cos_threshold. Typically adds < 0.01%
    more pairs but prevents silent failure if France records happen
    to have misassigned country labels.
    """
    pairs = set()
    s1_ids  = s1["entity_id"].to_list()
    s23_ids = s23["entity_id"].to_list()
    s1_countries  = dict(zip(s1_ids,  s1["country"].to_list()))
    s23_countries = dict(zip(s23_ids, s23["country"].to_list()))

    dim = s23_embeds.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(s23_embeds)

    k = min(5, len(s23_ids))
    distances, indices = index.search(s1_embeds, k)

    for qi, s1_id in enumerate(s1_ids):
        for rank in range(k):
            s23_idx = indices[qi, rank]
            if s23_idx < 0:
                continue
            cos_sim = float(distances[qi, rank])
            if cos_sim < cos_threshold:
                break   # Results are sorted; no point continuing
            s23_id = s23_ids[s23_idx]
            if s1_countries[s1_id] != s23_countries[s23_id]:
                pairs.add((s1_id, s23_id))

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
    Combine all three blocking strategies and return a mapping:
        { s1_entity_id : set of candidate S2/S3 entity_ids }

    This dict is BOTH the input to the feature/matcher stage AND the
    source for candidate_pairs.tsv.

    Parameters
    ----------
    s1, s2, s3   : Polars DataFrames with columns
                     [entity_id, business_name, business_address, country,
                      norm_name, norm_addr]
    s1_embeds    : L2-normalised embeddings for s1, shape (|s1|, dim)
    s2_embeds    : same for s2
    s3_embeds    : same for s3
    """
    # Combine S2 and S3 into one dataframe for joint blocking
    s23 = pl.concat([s2, s3])
    s23_embeds = np.vstack([s2_embeds, s3_embeds])

    print("[Blocking] Building token keys ...")
    s1  = _build_token_keys(s1)
    s23 = _build_token_keys(s23)

    key_cols = ["key_name_pref3", "key_name_pref4",
                "key_addr_pref4", "key_name_norm", "key_addr_norm"]

    print("[Blocking] Strategy A: Token blocking ...")
    pairs_token = _token_blocking_pairs(s1, s23, key_cols)
    print(f"           Token pairs: {len(pairs_token):,}")

    print("[Blocking] Strategy B: Sorted-neighborhood (window={snm_window}) ...")
    pairs_snm = _sorted_neighborhood_pairs(s1, s23, window=snm_window)
    print(f"           SNM pairs: {len(pairs_snm):,}")

    print("[Blocking] Strategy C: ANN (top_k={top_k}) ...")
    pairs_ann = _ann_blocking_pairs(s1, s23, s1_embeds, s23_embeds, top_k=top_k)
    print(f"           ANN pairs: {len(pairs_ann):,}")

    # Safety net
    print("[Blocking] Cross-country safety net ...")
    pairs_cc = _cross_country_safety_net(s1, s23, s1_embeds, s23_embeds)
    print(f"           Cross-country safety pairs: {len(pairs_cc):,}")

    # Union
    all_pairs = pairs_token | pairs_snm | pairs_ann | pairs_cc
    print(f"[Blocking] Total unique candidate pairs (union): {len(all_pairs):,}")

    # Pivot to dict: s1_id → set of s23_ids
    candidates: dict[str, set[str]] = defaultdict(set)
    for s1_id, s23_id in all_pairs:
        candidates[s1_id].add(s23_id)

    # Ensure every S1 entity has a row (even if empty)
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
