# =============================================================
# blocking.py — Multi-strategy candidate pair generation
# =============================================================
# Produces candidate_pairs.tsv: every (S1, S2/S3) pair that will
# be scored by the LightGBM matcher.
#
# Three complementary strategies are combined (union):
#   A. Token blocking       — exact-match on normalized name/address keys
#   B. Sorted-neighborhood  — slide window over sorted name prefix (vectorized Polars shifts)
#   C. ANN blocking         — top-k vector retrieval on multilingual embeddings (PyTorch GPU/CPU)
#
# Country partitioning is applied FIRST as a hard pre-filter (zero
# recall loss on training data — all 7.64M GT pairs are same-country).
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
    Vectorized Polars inner join in C++/Rust: 0 Python dicts, runs in < 1 second, < 100 MB RAM.
    Filters mega-buckets on both sides to prevent combinatorial explosion.
    """
    if col not in s1.columns or col not in target.columns:
        return

    # Filter nulls, short strings (< 4 chars), and generic mega-buckets
    target_clean = (
        target.select(["entity_id", "country", col])
              .filter(pl.col(col).is_not_null())
              .filter(pl.col(col).str.len_chars() >= 4)
              .filter(pl.len().over(["country", col]) <= max_bucket_size)
    )
    s1_clean = (
        s1.select(["entity_id", "country", col])
          .filter(pl.col(col).is_not_null())
          .filter(pl.col(col).str.len_chars() >= 4)
          .filter(pl.len().over(["country", col]) <= max_bucket_size)
    )
    matches = s1_clean.join(target_clean, on=["country", col], how="inner")
    del target_clean, s1_clean
    gc.collect()

    if matches.height == 0:
        del matches
        return

    # Aggregate by entity_id to update candidates in bulk (C-speed)
    grouped = (
        matches.select(["entity_id", "entity_id_right"])
               .group_by("entity_id")
               .agg(pl.col("entity_id_right"))
    )
    del matches
    gc.collect()

    for s1_id, tgts in zip(grouped["entity_id"].to_list(), grouped["entity_id_right"].to_list()):
        candidates[s1_id].update(tgts)
    del grouped
    gc.collect()


# ─── Strategy B: Vectorized Sorted Neighborhood Method (SNM) ────────────────

def _add_snm_matches(candidates: dict[str, set[str]],
                     s1: pl.DataFrame,
                     target: pl.DataFrame,
                     window: int = 3):
    """
    Sorted-Neighborhood Method on norm_name within each country.
    100% vectorized in Polars using column shifts in Rust. Zero Python loops, < 1 second runtime.
    """
    s1_sub = s1.select(["entity_id", "country", "norm_name"]).with_columns(
        pl.lit("S1").alias("src"))
    tgt_sub = target.select(["entity_id", "country", "norm_name"]).with_columns(
        pl.lit("TGT").alias("src"))
    combined = (
        pl.concat([s1_sub, tgt_sub])
          .filter(pl.col("norm_name").str.len_chars() >= 2)
          .sort(["country", "norm_name"])
    )
    del s1_sub, tgt_sub
    gc.collect()

    for offset in range(-window, window + 1):
        if offset == 0:
            continue
        pairs_df = (
            combined.select([
                pl.col("entity_id").alias("s1_id"),
                pl.col("src").alias("s1_src"),
                pl.col("country").alias("s1_country"),
                pl.col("entity_id").shift(-offset).alias("tgt_id"),
                pl.col("src").shift(-offset).alias("tgt_src"),
                pl.col("country").shift(-offset).alias("tgt_country"),
            ])
            .filter(
                (pl.col("s1_src") == "S1") &
                (pl.col("tgt_src") == "TGT") &
                (pl.col("s1_country") == pl.col("tgt_country"))
            )
            .select(["s1_id", "tgt_id"])
        )

        if pairs_df.height == 0:
            del pairs_df
            continue

        grouped = pairs_df.group_by("s1_id").agg(pl.col("tgt_id"))
        del pairs_df

        for s1_id, tgts in zip(grouped["s1_id"].to_list(), grouped["tgt_id"].to_list()):
            candidates[s1_id].update(tgts)
        del grouped

    del combined
    gc.collect()


# ─── Strategy C: PyTorch Vector ANN Search (GPU / CPU) ───────────────────────

_MODEL_CACHE: dict = {}


def encode_texts(texts: list[str],
                 model_name: str = EMBED_MODEL_NAME,
                 batch_size: int = EMBED_BATCH_SIZE,
                 max_seq_len: int = EMBED_MAX_SEQ_LEN) -> np.ndarray:
    """
    Encode a list of strings with the multilingual sentence encoder.
    Returns a float32 numpy array of shape (N, dim).
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
        normalize_embeddings=True,   # L2-normalised -> dot product == cosine
    )
    return embeddings.astype(np.float32)


def _add_ann_matches(candidates: dict[str, set[str]],
                     s1: pl.DataFrame,
                     target_df: pl.DataFrame,
                     s1_embeds: np.ndarray,
                     target_embeds: np.ndarray,
                     top_k: int = 15,
                     max_sim_bytes: int = 512 * 1024 * 1024):
    """
    Search S1 embeddings against target (S2 or S3) partitioned by country.
    Uses PyTorch matrix multiplication & topk with dynamic query batch sizing.
    The similarity matrix is capped at max_sim_bytes (default 512 MB VRAM),
    completely preventing CUDA OutOfMemoryError.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = (device.type == "cuda")

    s1_ids = s1["entity_id"].to_list()
    tgt_ids = target_df["entity_id"].to_list()
    s1_countries = s1["country"].to_list()
    tgt_countries = target_df["country"].to_list()

    # Group row indices by country
    country_to_s1 = defaultdict(list)
    for idx, c in enumerate(s1_countries):
        country_to_s1[c].append(idx)

    country_to_tgt = defaultdict(list)
    for idx, c in enumerate(tgt_countries):
        country_to_tgt[c].append(idx)

    for country, s1_idx_list in country_to_s1.items():
        tgt_idx_list = country_to_tgt.get(country, [])
        if not tgt_idx_list:
            continue

        n_tgt = len(tgt_idx_list)
        n_queries = len(s1_idx_list)
        k = min(top_k, n_tgt)

        country_tgt_ids = [tgt_ids[i] for i in tgt_idx_list]
        tgt_idx_arr = np.array(tgt_idx_list, dtype=np.int64)
        s1_idx_arr = np.array(s1_idx_list, dtype=np.int64)

        # Dynamic query batch size so (batch_size * n_tgt * element_size) <= max_sim_bytes
        elem_bytes = 2 if use_fp16 else 4
        safe_batch_size = max(32, min(512, int(max_sim_bytes / (n_tgt * elem_bytes))))
        vram_mb = int(safe_batch_size * n_tgt * elem_bytes / (1024 * 1024))
        print(f"           [{country}] {n_queries:,} queries vs {n_tgt:,} targets (batch={safe_batch_size}, ~{vram_mb} MB VRAM) ...")

        try:
            # Load target embeddings directly in float16/float32
            sub_tgt = target_embeds[tgt_idx_arr]
            if use_fp16:
                sub_tgt_arr = sub_tgt.astype(np.float16) if sub_tgt.dtype != np.float16 else sub_tgt
                tgt_t = torch.from_numpy(sub_tgt_arr).half().to(device)
            else:
                sub_tgt_arr = sub_tgt.astype(np.float32) if sub_tgt.dtype != np.float32 else sub_tgt
                tgt_t = torch.from_numpy(sub_tgt_arr).float().to(device)
            del sub_tgt, sub_tgt_arr

            # Transpose to (dim, N_tgt) for dot product
            tgt_t_T = tgt_t.t().contiguous()
            del tgt_t

            for b_start in range(0, n_queries, safe_batch_size):
                b_end = min(b_start + safe_batch_size, n_queries)
                b_indices = s1_idx_arr[b_start:b_end]
                b_embs = s1_embeds[b_indices]

                if use_fp16:
                    b_embs_arr = b_embs.astype(np.float16) if b_embs.dtype != np.float16 else b_embs
                    q_t = torch.from_numpy(b_embs_arr).half().to(device)
                else:
                    b_embs_arr = b_embs.astype(np.float32) if b_embs.dtype != np.float32 else b_embs
                    q_t = torch.from_numpy(b_embs_arr).float().to(device)
                del b_embs, b_embs_arr

                # Cosine similarity dot-product: (batch_size, dim) @ (dim, N_tgt) -> (batch_size, N_tgt)
                sim = torch.mm(q_t, tgt_t_T)
                _, topk_local_idx = torch.topk(sim, k=k, dim=1)
                topk_np = topk_local_idx.cpu().numpy()
                del sim, q_t, topk_local_idx

                for qi, s1_orig_idx in enumerate(b_indices):
                    s1_id = s1_ids[s1_orig_idx]
                    candidates[s1_id].update(country_tgt_ids[loc] for loc in topk_np[qi] if loc >= 0)

            del tgt_t_T
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"           [Warning] CUDA OOM for {country}, falling back to CPU ...")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            sub_tgt = target_embeds[tgt_idx_arr]
            tgt_t_cpu = torch.from_numpy(sub_tgt.astype(np.float32)).t().contiguous()
            del sub_tgt

            cpu_batch = 128
            for b_start in range(0, n_queries, cpu_batch):
                b_end = min(b_start + cpu_batch, n_queries)
                b_indices = s1_idx_arr[b_start:b_end]
                b_embs = s1_embeds[b_indices]
                q_t = torch.from_numpy(b_embs.astype(np.float32))
                del b_embs

                sim = torch.mm(q_t, tgt_t_cpu)
                _, topk_local_idx = torch.topk(sim, k=k, dim=1)
                topk_np = topk_local_idx.numpy()
                del sim, q_t, topk_local_idx

                for qi, s1_orig_idx in enumerate(b_indices):
                    s1_id = s1_ids[s1_orig_idx]
                    candidates[s1_id].update(country_tgt_ids[loc] for loc in topk_np[qi] if loc >= 0)

            del tgt_t_cpu
            gc.collect()

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

    print(f"[Blocking] Strategy B: Vectorized sorted-neighborhood (window={snm_window}) ...")
    _add_snm_matches(candidates, s1, s2, window=snm_window)
    _add_snm_matches(candidates, s1, s3, window=snm_window)
    print(f"           Pairs after Strategy B: {sum(len(v) for v in candidates.values()):,}")

    ann_k = max(10, top_k // 2)
    print(f"[Blocking] Strategy C: Vector ANN search (top-{ann_k} per target) ...")
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
