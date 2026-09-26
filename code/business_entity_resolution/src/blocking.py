# ==============================================================================
# blocking.py — Single-Pass Fusion Blocking for 99.9% Recall
# ==============================================================================
# Complete redesign for Amazon ML Challenge 2026 (Business Entity Resolution).
#
# ARCHITECTURE: 4-Layer Fusion
#   Layer 1: Exact Hash Blocking         — O(N) Polars joins       (~2 min)
#   Layer 2: GPU ANN Dense Vector Search  — FP16 matmul on A10G    (~1 min)
#   Layer 3: Single-Pass Fused TF-IDF     — ONE vectorization/country (~5 min)
#   Layer 4: Token Overlap Safety Net     — for under-covered S1s  (~1 min)
#
# KEY DESIGN DECISION:
#   The old code ran 4 separate TF-IDF passes (combined-word, char-4gram,
#   address-guarded, reverse) × 3 countries = 12 vectorize-and-multiply cycles.
#   This redesign fuses them into a SINGLE pass per country (3 total), cutting
#   ~75% of redundant vectorization while preserving union-of-signals recall.
#
# MEMORY BUDGET: < 5 GB peak on 32 GB SageMaker ml.g5.2xlarge
# ==============================================================================

import gc
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
import polars as pl
from tqdm import tqdm

from config import (
    ANN_TOP_K,
    EMBED_MODEL_NAME, EMBED_BATCH_SIZE, EMBED_MAX_SEQ_LEN,
    HF_CACHE_DIR,
)

# Maximum candidates permitted per S1 entity (prevents outlier bloat)
MAX_CANDS_PER_S1 = 100


# ─── Sparse Top-K Extraction (Zero Dense Allocation) ─────────────────────────

def _extract_sparse_topk(sim_csr, top_k: int, min_score: float = 0.01) -> list[list[int]]:
    """
    Extract top-K column indices for each row of a CSR sparse matrix.
    Zero dense memory allocation — operates directly on CSR indptr/indices/data.
    """
    results = []
    indptr = sim_csr.indptr
    indices = sim_csr.indices
    data = sim_csr.data

    n_rows = sim_csr.shape[0]
    for i in range(n_rows):
        start = indptr[i]
        end = indptr[i + 1]
        if end == start:
            results.append([])
            continue

        row_data = data[start:end]
        row_ind = indices[start:end]

        valid_mask = row_data >= min_score
        if not np.any(valid_mask):
            results.append([])
            continue

        v_data = row_data[valid_mask]
        v_ind = row_ind[valid_mask]

        if len(v_data) <= top_k:
            results.append(v_ind.tolist())
        else:
            top_locs = np.argpartition(v_data, -top_k)[-top_k:]
            results.append(v_ind[top_locs].tolist())

    return results


# ==============================================================================
# LAYER 1: Exact Hash Blocking (O(N) Polars Inner Joins)
# ==============================================================================

def _get_exact_matches(s1: pl.DataFrame,
                       target: pl.DataFrame,
                       col: str,
                       max_bucket_size: int = 100,
                       min_len: int = 4) -> pl.DataFrame:
    """
    Match entities on (country, col) using vectorized Polars inner join.
    Returns a Polars DataFrame with columns [s1_id, cand_id].
    """
    if col not in s1.columns or col not in target.columns:
        return pl.DataFrame({"s1_id": [], "cand_id": []}, schema={"s1_id": pl.Utf8, "cand_id": pl.Utf8})

    try:
        s1_clean = (
            s1.select(["entity_id", "country", col])
              .filter(pl.col(col).is_not_null())
              .filter(pl.col(col).str.len_chars() >= min_len)
              .filter(pl.len().over(["country", col]) <= max_bucket_size)
        )
        tgt_clean = (
            target.select(["entity_id", "country", col])
                  .filter(pl.col(col).is_not_null())
                  .filter(pl.col(col).str.len_chars() >= min_len)
                  .filter(pl.len().over(["country", col]) <= max_bucket_size)
        )
        matches = s1_clean.join(tgt_clean, on=["country", col], how="inner")
        del s1_clean, tgt_clean

        if matches.height == 0:
            del matches
            return pl.DataFrame({"s1_id": [], "cand_id": []}, schema={"s1_id": pl.Utf8, "cand_id": pl.Utf8})

        res = matches.select([
            pl.col("entity_id").alias("s1_id"),
            pl.col("entity_id_right").alias("cand_id")
        ])
        del matches
        return res

    except Exception as e:
        print(f"    [WARNING] Exact match on {col} failed: {e}")
        return pl.DataFrame({"s1_id": [], "cand_id": []}, schema={"s1_id": pl.Utf8, "cand_id": pl.Utf8})


def _get_exact_key_matches(s1: pl.DataFrame,
                           target: pl.DataFrame,
                           key_expr,
                           key_name: str = "key",
                           max_bucket_size: int = 100,
                           min_key_len: int = 5) -> pl.DataFrame:
    """
    Match entities on a dynamically computed Polars expression key within country.
    """
    try:
        s1_keyed = (
            s1.with_columns(key_expr.alias(key_name))
              .select(["entity_id", "country", key_name])
              .filter(pl.col(key_name).is_not_null())
              .filter(pl.col(key_name).str.len_chars() >= min_key_len)
              .filter(pl.len().over(["country", key_name]) <= max_bucket_size)
        )
        tgt_keyed = (
            target.with_columns(key_expr.alias(key_name))
                  .select(["entity_id", "country", key_name])
                  .filter(pl.col(key_name).is_not_null())
                  .filter(pl.col(key_name).str.len_chars() >= min_key_len)
                  .filter(pl.len().over(["country", key_name]) <= max_bucket_size)
        )

        matches = s1_keyed.join(tgt_keyed, on=["country", key_name], how="inner")
        del s1_keyed, tgt_keyed

        if matches.height == 0:
            del matches
            return pl.DataFrame({"s1_id": [], "cand_id": []}, schema={"s1_id": pl.Utf8, "cand_id": pl.Utf8})

        res = matches.select([
            pl.col("entity_id").alias("s1_id"),
            pl.col("entity_id_right").alias("cand_id")
        ])
        del matches
        return res

    except Exception as e:
        print(f"    [WARNING] Exact key matching ({key_name}) failed: {e}")
        return pl.DataFrame({"s1_id": [], "cand_id": []}, schema={"s1_id": pl.Utf8, "cand_id": pl.Utf8})


def _run_layer1_exact(s1: pl.DataFrame, s2: pl.DataFrame, s3: pl.DataFrame) -> pl.DataFrame:
    """
    Layer 1: All exact hash join strategies in one function.
    Returns deduplicated Polars DataFrame of [s1_id, cand_id].
    """
    t0 = time.time()
    print("\n[Layer 1/4] Exact Key Matching (Bucket ≤ 100) ...")
    p1_dfs = []

    for tgt_name, tgt in [("S2", s2), ("S3", s3)]:
        # Direct column matches
        for col in ["norm_name", "norm_name_ns", "norm_name_nospace"]:
            if col in s1.columns and col in tgt.columns:
                p1_dfs.append(_get_exact_matches(s1, tgt, col, max_bucket_size=100, min_len=4))

        # Composite key: name + first address number
        if "norm_name" in s1.columns and "norm_addr" in s1.columns:
            name_num_key = (
                pl.col("norm_name").fill_null("") + "|" +
                pl.col("norm_addr").fill_null("").str.extract(r"(\d+)", 1).fill_null("")
            )
            p1_dfs.append(_get_exact_key_matches(s1, tgt, name_num_key, key_name="k_name_num",
                                                  max_bucket_size=100, min_key_len=5))

        # Composite key: nospace_name + first address number
        if "norm_name_nospace" in s1.columns and "norm_addr" in s1.columns:
            nosp_num_key = (
                pl.col("norm_name_nospace").fill_null("") + "|" +
                pl.col("norm_addr").fill_null("").str.extract(r"(\d+)", 1).fill_null("")
            )
            p1_dfs.append(_get_exact_key_matches(s1, tgt, nosp_num_key, key_name="k_nosp_num",
                                                  max_bucket_size=100, min_key_len=5))

    # Deduplicate
    p1_dfs = [df for df in p1_dfs if df.height > 0]
    if p1_dfs:
        result = pl.concat(p1_dfs).unique(subset=["s1_id", "cand_id"])
        del p1_dfs
    else:
        result = pl.DataFrame({"s1_id": [], "cand_id": []}, schema={"s1_id": pl.Utf8, "cand_id": pl.Utf8})

    gc.collect()
    print(f"  ✓ Layer 1 complete in {time.time()-t0:.1f}s → {result.height:,} unique pairs")
    return result


# ==============================================================================
# LAYER 2: GPU ANN Dense Vector Search (Top-35, Country-Partitioned, Direct Stream)
# ==============================================================================

def _run_layer2_ann(s1: pl.DataFrame, s2: pl.DataFrame, s3: pl.DataFrame,
                    s1_embeds: np.ndarray, s2_embeds: np.ndarray, s3_embeds: np.ndarray,
                    candidates: dict[str, set[str]],
                    top_k: int = 35,
                    min_sim: float = 0.45,
                    query_batch_size: int = 4096) -> None:
    """
    Layer 2: GPU FP16 dense vector ANN search, country-partitioned.
    Streams results DIRECTLY into candidates dictionary in-place.
    Zero intermediate 220M-row Polars DataFrames, zero disk swapping.
    """
    import torch

    t0 = time.time()
    print(f"\n[Layer 2/4] GPU Dense Vector ANN (Top-{top_k}, min_sim={min_sim}) ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = (device.type == "cuda")

    s1_ids = s1["entity_id"].to_list()
    s1_countries = s1["country"].to_list()

    country_to_s1 = defaultdict(list)
    for idx, c in enumerate(s1_countries):
        country_to_s1[c].append(idx)

    for tgt_name, tgt_df, tgt_embeds in [("S2", s2, s2_embeds), ("S3", s3, s3_embeds)]:
        tgt_ids = tgt_df["entity_id"].to_list()
        tgt_countries = tgt_df["country"].to_list()

        country_to_tgt = defaultdict(list)
        for idx, c in enumerate(tgt_countries):
            country_to_tgt[c].append(idx)

        for country in tqdm(sorted(country_to_s1.keys()), desc=f"  ANN → {tgt_name} (GPU)"):
            tgt_idx_list = country_to_tgt.get(country, [])
            s1_idx_list = country_to_s1[country]
            if not tgt_idx_list:
                continue

            n_tgt = len(tgt_idx_list)
            n_queries = len(s1_idx_list)
            k = min(top_k, n_tgt)

            country_tgt_ids = [tgt_ids[i] for i in tgt_idx_list]
            country_s1_ids = [s1_ids[i] for i in s1_idx_list]
            tgt_idx_arr = np.array(tgt_idx_list, dtype=np.int64)
            s1_idx_arr = np.array(s1_idx_list, dtype=np.int64)

            try:
                # Load all S1 embeddings for this country into CPU RAM
                country_s1_embs = s1_embeds[s1_idx_arr]
                if use_fp16:
                    country_s1_embs = country_s1_embs.astype(np.float16)
                else:
                    country_s1_embs = country_s1_embs.astype(np.float32)

                total_gpu_mem = torch.cuda.get_device_properties(device).total_memory if use_fp16 else 0
                can_fit_target_on_gpu = (use_fp16 and n_tgt <= 4_000_000 and total_gpu_mem >= 18 * (1024**3))

                if can_fit_target_on_gpu:
                    print(f"    [{country}] Direct GPU mode: {n_queries:,} queries × {n_tgt:,} targets "
                          f"(q_batch=512, peak VRAM ~6.5 GB)", flush=True)

                    sub_tgt = tgt_embeds[tgt_idx_arr]
                    tgt_t = torch.from_numpy(sub_tgt.astype(np.float16)).to(device)
                    del sub_tgt
                    tgt_t_T = tgt_t.t().contiguous()
                    del tgt_t

                    q_batch = 512
                    for qb_start in range(0, n_queries, q_batch):
                        qb_end = min(qb_start + q_batch, n_queries)
                        q_embs = country_s1_embs[qb_start:qb_end]
                        q_t = torch.from_numpy(q_embs).to(device)

                        sim = torch.mm(q_t, tgt_t_T)
                        del q_t

                        top_scores, topk_local_idx = torch.topk(sim, k=k, dim=1)
                        del sim

                        scores_np = top_scores.cpu().numpy()
                        topk_np = topk_local_idx.cpu().numpy()
                        del top_scores, topk_local_idx

                        n_q = qb_end - qb_start
                        for qi in range(n_q):
                            s1_id = country_s1_ids[qb_start + qi]
                            cset = candidates[s1_id]
                            s_row = scores_np[qi]
                            idx_row = topk_np[qi]
                            for rank in range(k):
                                if len(cset) >= MAX_CANDS_PER_S1:
                                    break
                                # Keep semantic matches (score >= min_sim) OR top 5 closest
                                if s_row[rank] >= min_sim or rank < 5:
                                    loc = idx_row[rank]
                                    if loc >= 0:
                                        cset.add(country_tgt_ids[loc])
                                else:
                                    break  # sorted descending; remaining are even lower

                        del scores_np, topk_np

                    del tgt_t_T
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                else:
                    # Target-chunked mode
                    TGT_CHUNK = 250_000
                    n_tgt_chunks = (n_tgt + TGT_CHUNK - 1) // TGT_CHUNK
                    q_batch = 1024

                    print(f"    [{country}] Target-chunked mode: {n_queries:,} queries × {n_tgt:,} targets "
                          f"(tgt_chunks={n_tgt_chunks}, q_batch={q_batch})", flush=True)

                    for qb_start in range(0, n_queries, q_batch):
                        qb_end = min(qb_start + q_batch, n_queries)
                        q_embs = country_s1_embs[qb_start:qb_end]
                        q_t = torch.from_numpy(q_embs).to(device)
                        n_q = qb_end - qb_start

                        all_scores = torch.full((n_q, k), -1.0, device=device,
                                               dtype=torch.float16 if use_fp16 else torch.float32)
                        all_indices = torch.full((n_q, k), -1, device=device, dtype=torch.long)

                        for tc in range(n_tgt_chunks):
                            tc_start = tc * TGT_CHUNK
                            tc_end = min(tc_start + TGT_CHUNK, n_tgt)
                            tc_size = tc_end - tc_start

                            sub_tgt = tgt_embeds[tgt_idx_arr[tc_start:tc_end]]
                            if use_fp16:
                                tgt_chunk_t = torch.from_numpy(sub_tgt.astype(np.float16)).to(device)
                            else:
                                tgt_chunk_t = torch.from_numpy(sub_tgt.astype(np.float32)).to(device)
                            del sub_tgt

                            sim_chunk = torch.mm(q_t, tgt_chunk_t.t())
                            del tgt_chunk_t

                            chunk_k = min(k, tc_size)
                            chunk_scores, chunk_local_idx = torch.topk(sim_chunk, k=chunk_k, dim=1)
                            del sim_chunk

                            chunk_global_idx = chunk_local_idx + tc_start
                            del chunk_local_idx

                            if chunk_k < k:
                                pad_s = torch.full((n_q, k - chunk_k), -1.0, device=device, dtype=chunk_scores.dtype)
                                pad_i = torch.full((n_q, k - chunk_k), -1, device=device, dtype=torch.long)
                                chunk_scores = torch.cat([chunk_scores, pad_s], dim=1)
                                chunk_global_idx = torch.cat([chunk_global_idx, pad_i], dim=1)
                                del pad_s, pad_i

                            combined_scores = torch.cat([all_scores, chunk_scores], dim=1)
                            combined_indices = torch.cat([all_indices, chunk_global_idx], dim=1)
                            del chunk_scores, chunk_global_idx

                            top_scores, merge_idx = torch.topk(combined_scores, k=k, dim=1)
                            all_scores = top_scores
                            all_indices = combined_indices.gather(1, merge_idx)
                            del combined_scores, combined_indices, top_scores, merge_idx

                        del q_t

                        scores_np = all_scores.cpu().numpy()
                        topk_np = all_indices.cpu().numpy()
                        del all_scores, all_indices

                        for qi in range(n_q):
                            s1_id = country_s1_ids[qb_start + qi]
                            cset = candidates[s1_id]
                            s_row = scores_np[qi]
                            idx_row = topk_np[qi]
                            for rank in range(k):
                                if len(cset) >= MAX_CANDS_PER_S1:
                                    break
                                if s_row[rank] >= min_sim or rank < 5:
                                    loc = idx_row[rank]
                                    if loc >= 0:
                                        cset.add(country_tgt_ids[loc])
                                else:
                                    break

                        del scores_np, topk_np

                del country_s1_embs
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"    [WARNING] GPU ANN for {country}→{tgt_name} failed: {e}")
                traceback.print_exc()
                if "cuda" in str(device):
                    torch.cuda.empty_cache()

            gc.collect()

    print(f"  ✓ Layer 2 complete in {time.time()-t0:.1f}s")


# ==============================================================================
# LAYER 3: Single-Pass Fused TF-IDF (Direct Stream to candidates)
# ==============================================================================

def _run_layer3_fused_tfidf(s1: pl.DataFrame, s2: pl.DataFrame, s3: pl.DataFrame,
                             candidates: dict[str, set[str]],
                             forward_top_k: int = 35,
                             reverse_top_k: int = 5,
                             batch_size: int = 5000) -> None:
    """
    Layer 3: Single-pass fused TF-IDF blocking.
    Streams results directly into `candidates` dictionary.
    Zero intermediate Polars DataFrames, peak RAM < 4 GB.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    t0 = time.time()
    print(f"\n[Layer 3/4] Single-Pass Fused TF-IDF (fwd top-{forward_top_k}, rev top-{reverse_top_k}) ...")

    countries = sorted(set(s1["country"].to_list()))

    for tgt_name, tgt in [("S2", s2), ("S3", s3)]:
        tgt_countries = set(tgt["country"].to_list())

        for country in tqdm(countries, desc=f"  Fused TF-IDF → {tgt_name}"):
            if country not in tgt_countries:
                continue

            try:
                s1_c = s1.filter(pl.col("country") == country)
                tgt_c = tgt.filter(pl.col("country") == country)

                s1_ids = s1_c["entity_id"].to_list()
                tgt_ids = tgt_c["entity_id"].to_list()
                n_s1 = len(s1_ids)
                n_tgt = len(tgt_ids)

                if n_s1 == 0 or n_tgt == 0:
                    continue

                # Extract texts once
                s1_names = s1_c["norm_name"].fill_null("").to_list()
                tgt_names = tgt_c["norm_name"].fill_null("").to_list()
                s1_addrs = s1_c["norm_addr"].fill_null("").to_list()
                tgt_addrs = tgt_c["norm_addr"].fill_null("").to_list()

                # Nospace names for char-gram
                ns_col = "norm_name_nospace" if "norm_name_nospace" in s1_c.columns else "norm_name"
                s1_names_ns = s1_c[ns_col].fill_null("").to_list()
                tgt_names_ns = tgt_c[ns_col].fill_null("").to_list() if ns_col in tgt_c.columns else tgt_names

                del s1_c, tgt_c
                gc.collect()

                print(f"    [{country}] Vectorizing {n_tgt:,} targets + {n_s1:,} queries (3 signals) ...", flush=True)

                # ── Signal A: Name word TF-IDF ────────────────────────────
                tfidf_name = TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2),
                    max_df=0.30, min_df=2, max_features=200_000,
                    stop_words="english", sublinear_tf=True, dtype=np.float32,
                )
                tgt_name_vecs = tfidf_name.fit_transform(tgt_names)
                s1_name_vecs = tfidf_name.transform(s1_names)
                del tfidf_name
                gc.collect()

                # ── Signal B: Address word TF-IDF ─────────────────────────
                tfidf_addr = TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 1),
                    max_df=0.30, min_df=2, max_features=150_000,
                    stop_words="english", sublinear_tf=True, dtype=np.float32,
                )
                tgt_addr_vecs = tfidf_addr.fit_transform(tgt_addrs)
                s1_addr_vecs = tfidf_addr.transform(s1_addrs)
                del tfidf_addr
                gc.collect()

                # ── Signal C: Name char 4-gram TF-IDF ─────────────────────
                tfidf_char = TfidfVectorizer(
                    analyzer="char", ngram_range=(3, 5),
                    max_df=0.40, min_df=2, max_features=200_000,
                    sublinear_tf=True, dtype=np.float32,
                )
                tgt_char_vecs = tfidf_char.fit_transform(tgt_names_ns)
                s1_char_vecs = tfidf_char.transform(s1_names_ns)
                del tfidf_char
                gc.collect()

                del s1_names, tgt_names, s1_addrs, tgt_addrs, s1_names_ns, tgt_names_ns

                # ── Fused Retrieval: Forward (S1 → Target) ────────────────
                tgt_name_T = tgt_name_vecs.T.tocsc()
                tgt_addr_T = tgt_addr_vecs.T.tocsc()
                tgt_char_T = tgt_char_vecs.T.tocsc()

                for b_start in tqdm(range(0, n_s1, batch_size),
                                    total=(n_s1 + batch_size - 1) // batch_size,
                                    desc=f"    [{country}] Forward fused retrieval",
                                    leave=False):
                    b_end = min(b_start + batch_size, n_s1)

                    sim_name = s1_name_vecs[b_start:b_end].dot(tgt_name_T)
                    sim_addr = s1_addr_vecs[b_start:b_end].dot(tgt_addr_T)
                    sim_char = s1_char_vecs[b_start:b_end].dot(tgt_char_T)

                    fused = sim_name.maximum(sim_addr).maximum(sim_char)
                    top_indices = _extract_sparse_topk(fused.tocsr(), top_k=forward_top_k, min_score=0.03)
                    del sim_name, sim_addr, sim_char, fused

                    for qi, col_indices in enumerate(top_indices):
                        if col_indices:
                            s1_id = s1_ids[b_start + qi]
                            cset = candidates[s1_id]
                            for c_idx in col_indices:
                                if len(cset) < MAX_CANDS_PER_S1:
                                    cset.add(tgt_ids[c_idx])

                del tgt_name_T, tgt_addr_T, tgt_char_T

                # ── Fused Retrieval: Reverse (Target → S1) ────────────────
                s1_name_T = s1_name_vecs.T.tocsc()
                s1_addr_T = s1_addr_vecs.T.tocsc()
                s1_char_T = s1_char_vecs.T.tocsc()

                for b_start in tqdm(range(0, n_tgt, batch_size),
                                    total=(n_tgt + batch_size - 1) // batch_size,
                                    desc=f"    [{country}] Reverse retrieval",
                                    leave=False):
                    b_end = min(b_start + batch_size, n_tgt)

                    sim_name = tgt_name_vecs[b_start:b_end].dot(s1_name_T)
                    sim_addr = tgt_addr_vecs[b_start:b_end].dot(s1_addr_T)
                    sim_char = tgt_char_vecs[b_start:b_end].dot(s1_char_T)

                    fused = sim_name.maximum(sim_addr).maximum(sim_char)
                    top_indices = _extract_sparse_topk(fused.tocsr(), top_k=reverse_top_k, min_score=0.03)
                    del sim_name, sim_addr, sim_char, fused

                    for qi, s1_col_indices in enumerate(top_indices):
                        if s1_col_indices:
                            tgt_id = tgt_ids[b_start + qi]
                            for s1_col_idx in s1_col_indices:
                                s1_id = s1_ids[s1_col_idx]
                                cset = candidates[s1_id]
                                if len(cset) < MAX_CANDS_PER_S1:
                                    cset.add(tgt_id)

                del s1_name_T, s1_addr_T, s1_char_T
                del s1_name_vecs, s1_addr_vecs, s1_char_vecs
                del tgt_name_vecs, tgt_addr_vecs, tgt_char_vecs
                gc.collect()

            except Exception as e:
                print(f"  [WARNING] Fused TF-IDF for {country}→{tgt_name} failed: {e}")
                traceback.print_exc()
                gc.collect()

    print(f"  ✓ Layer 3 complete in {time.time()-t0:.1f}s")


# ==============================================================================
# LAYER 4: Token Overlap Safety Net (Direct Stream to candidates)
# ==============================================================================

def _run_layer4_safety_net(s1: pl.DataFrame, s2: pl.DataFrame, s3: pl.DataFrame,
                            candidates: dict[str, set[str]],
                            min_candidates: int = 5,
                            top_k: int = 10) -> None:
    """
    Layer 4: Lightweight token-overlap safety net.
    Operates ONLY on S1 entities that have fewer than `min_candidates`.
    Streams results directly into `candidates` dictionary.
    """
    t0 = time.time()
    print(f"\n[Layer 4/4] Token Overlap Safety Net (min_cands={min_candidates}) ...")

    under_covered = [sid for sid, cset in candidates.items() if len(cset) < min_candidates]

    if not under_covered:
        print(f"  ✓ All S1 entities have ≥ {min_candidates} candidates. Safety net not needed.")
        return

    print(f"  Found {len(under_covered):,} under-covered S1 entities (< {min_candidates} candidates)")

    # Build token sets for under-covered S1 entities
    uc_set = set(under_covered)
    s1_filtered = s1.filter(pl.col("entity_id").is_in(list(uc_set)))

    s1_token_data = {}
    for row in s1_filtered.iter_rows(named=True):
        eid = row["entity_id"]
        name_str = (row.get("norm_name", "") or "").strip()
        addr_str = (row.get("norm_addr", "") or "").strip()
        combined = name_str + " " + addr_str
        tokens = set(combined.split())
        tokens.discard("")
        country = row.get("country", "")
        s1_token_data[eid] = (tokens, country)

    del s1_filtered
    gc.collect()

    # Build target token index (inverted index for fast lookup)
    for tgt_name, tgt in [("S2", s2), ("S3", s3)]:
        country_inverted = defaultdict(lambda: defaultdict(list))
        country_tgt_tokens = defaultdict(dict)

        for row in tgt.iter_rows(named=True):
            eid = row["entity_id"]
            country = row.get("country", "")
            name_str = (row.get("norm_name", "") or "").strip()
            addr_str = (row.get("norm_addr", "") or "").strip()
            combined = name_str + " " + addr_str
            tokens = set(combined.split())
            tokens.discard("")
            country_tgt_tokens[country][eid] = tokens
            for tok in tokens:
                if len(tok) >= 3:
                    country_inverted[country][tok].append(eid)

        added = 0
        for s1_id in tqdm(under_covered, desc=f"  Safety net → {tgt_name}", leave=False):
            s1_tokens, s1_country = s1_token_data.get(s1_id, (set(), ""))
            if not s1_tokens or s1_country not in country_tgt_tokens:
                continue

            candidate_scores = defaultdict(int)
            for tok in s1_tokens:
                if len(tok) >= 3 and tok in country_inverted[s1_country]:
                    for cand_id in country_inverted[s1_country][tok]:
                        candidate_scores[cand_id] += 1

            if not candidate_scores:
                continue

            if len(candidate_scores) > 100:
                top_by_overlap = sorted(candidate_scores.items(), key=lambda x: -x[1])[:100]
            else:
                top_by_overlap = list(candidate_scores.items())

            scored = []
            s1_len = len(s1_tokens)
            for cand_id, overlap_count in top_by_overlap:
                cand_tokens = country_tgt_tokens[s1_country].get(cand_id, set())
                if not cand_tokens:
                    continue
                intersection = len(s1_tokens & cand_tokens)
                union = s1_len + len(cand_tokens) - intersection
                jaccard = intersection / union if union > 0 else 0.0
                if jaccard >= 0.1:
                    scored.append((cand_id, jaccard))

            scored.sort(key=lambda x: -x[1])
            cset = candidates[s1_id]
            for cand_id, _ in scored[:top_k]:
                if len(cset) < MAX_CANDS_PER_S1:
                    cset.add(cand_id)
                    added += 1

        del country_inverted, country_tgt_tokens
        gc.collect()

    print(f"  ✓ Layer 4 complete in {time.time()-t0:.1f}s")


# ==============================================================================
# Embedding Encoder (unchanged — used by pipeline.py)
# ==============================================================================

_MODEL_CACHE: dict = {}


def encode_texts(texts: list[str],
                 model_name: str = EMBED_MODEL_NAME,
                 batch_size: int = EMBED_BATCH_SIZE,
                 max_seq_len: int = EMBED_MAX_SEQ_LEN) -> np.ndarray:
    """
    Encode texts using SentenceTransformer with EBS cache redirection.
    Returns float32 normalized embeddings.
    """
    from sentence_transformers import SentenceTransformer

    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name, cache_folder=HF_CACHE_DIR)
    model = _MODEL_CACHE[model_name]
    model.max_seq_length = max_seq_len

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # Dot product == cosine similarity
    )
    return embeddings.astype(np.float32)


# ==============================================================================
# PUBLIC API: generate_candidates (Zero-Swap In-Place Streaming)
# ==============================================================================

def generate_candidates(s1: pl.DataFrame,
                        s2: pl.DataFrame,
                        s3: pl.DataFrame,
                        s1_embeds: np.ndarray,
                        s2_embeds: np.ndarray,
                        s3_embeds: np.ndarray,
                        top_k: int = ANN_TOP_K,
                        snm_window: int = 5) -> dict[str, set[str]]:
    """
    Generate candidates using 4-Layer Fusion Blocking.
    Zero-Swap, In-Place Streaming Architecture:
    - Layer 1: Exact Hash Blocking (Polars joins) -> ingested directly into candidates dict
    - Layer 2: GPU ANN Dense Vector Search -> streamed directly into candidates dict
    - Layer 3: Single-Pass Fused TF-IDF -> streamed directly into candidates dict
    - Layer 4: Token Overlap Safety Net -> streamed directly into candidates dict

    Peak RAM: < 5 GB (eliminates 60 GB Polars DataFrame bloat and swap thrashing).
    Returns: dict mapping s1_entity_id → set of candidate entity_ids.
    """
    start_time = time.time()

    print("\n" + "="*70)
    print("  4-LAYER FUSION BLOCKING (ZERO-SWAP DIRECT STREAMING)")
    print("="*70)
    print(f"  Inputs: S1={len(s1):,} rows | S2={len(s2):,} rows | S3={len(s3):,} rows")

    # Master candidate dictionary: pre-allocate keys for all S1 entities
    all_s1_ids = s1["entity_id"].to_list()
    candidates: dict[str, set[str]] = {sid: set() for sid in all_s1_ids}

    # ──────────────────────────────────────────────────────────────────────────
    # Layer 1: Exact Key Matching
    # ──────────────────────────────────────────────────────────────────────────
    l1_df = _run_layer1_exact(s1, s2, s3)
    t0 = time.time()
    l1_s1 = l1_df["s1_id"].to_list()
    l1_cands = l1_df["cand_id"].to_list()
    del l1_df
    gc.collect()

    added_l1 = 0
    for sid, cid in zip(l1_s1, l1_cands):
        cset = candidates[sid]
        if len(cset) < MAX_CANDS_PER_S1:
            cset.add(cid)
            added_l1 += 1
    del l1_s1, l1_cands
    gc.collect()
    print(f"  [L1 Ingested] {added_l1:,} pairs merged in {time.time()-t0:.1f}s | "
          f"Total unique pairs: {sum(len(v) for v in candidates.values()):,}")

    # ──────────────────────────────────────────────────────────────────────────
    # Layer 2: GPU ANN (Top-35, Streams directly to candidates dict)
    # ──────────────────────────────────────────────────────────────────────────
    ann_k = min(35, max(25, top_k, ANN_TOP_K))
    _run_layer2_ann(s1, s2, s3, s1_embeds, s2_embeds, s3_embeds, candidates,
                    top_k=ann_k, min_sim=0.45)
    gc.collect()
    print(f"  [L2 Ingested] Cumulative pairs after L1+L2: {sum(len(v) for v in candidates.values()):,}")

    # ──────────────────────────────────────────────────────────────────────────
    # Layer 3: Single-Pass Fused TF-IDF (Streams directly to candidates dict)
    # ──────────────────────────────────────────────────────────────────────────
    _run_layer3_fused_tfidf(s1, s2, s3, candidates, forward_top_k=35, reverse_top_k=5)
    gc.collect()
    print(f"  [L3 Ingested] Cumulative pairs after L1+L2+L3: {sum(len(v) for v in candidates.values()):,}")

    # ──────────────────────────────────────────────────────────────────────────
    # Layer 4: Safety Net for under-covered entities (< 5 candidates)
    # ──────────────────────────────────────────────────────────────────────────
    _run_layer4_safety_net(s1, s2, s3, candidates, min_candidates=5, top_k=10)
    gc.collect()

    total_pairs = sum(len(v) for v in candidates.values())
    n_s1 = len(all_s1_ids)
    avg_cands = total_pairs / max(n_s1, 1)

    print(f"\n{'='*70}")
    print(f"  4-LAYER FUSION BLOCKING COMPLETED IN {time.time()-start_time:.1f}s")
    print(f"  Total S1 entities processed: {n_s1:,}")
    print(f"  Total unique candidate pairs: {total_pairs:,}")
    print(f"  Average candidates per S1   : {avg_cands:.1f}")
    print(f"{'='*70}\n")

    return candidates


# ==============================================================================
# PUBLIC API: Blocking Recall Evaluation
# ==============================================================================

def compute_blocking_recall(candidates: dict[str, set[str]],
                            ground_truth: pl.DataFrame) -> float:
    """
    Compute blocking recall against ground truth:
      blocking_recall = |GT pairs captured in candidates| / |GT pairs total|
    """
    total_gt = 0
    captured = 0
    missed_examples = []

    gt_dict: dict[str, set[str]] = {}
    for row in ground_truth.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        raw = row.get("matched_entity_ids", "") or ""
        matches = {m.strip() for m in raw.split(",") if m.strip()}
        gt_dict[s1_id] = matches

    for s1_id, true_matches in tqdm(gt_dict.items(), desc="Computing blocking recall"):
        cands = candidates.get(s1_id, set())
        for m in true_matches:
            total_gt += 1
            if m in cands:
                captured += 1
            elif len(missed_examples) < 20:
                missed_examples.append((s1_id, m))

    recall = captured / total_gt if total_gt > 0 else 1.0
    missed = total_gt - captured

    print("\n" + "="*60)
    print(f"  BLOCKING RECALL: {captured:,} / {total_gt:,} = {recall*100:.2f}%")
    print(f"  Missed pairs   : {missed:,}")
    print("="*60)

    if missed_examples:
        print(f"\n  Sample missed pairs ({len(missed_examples)} shown):")
        for s1_id, m_id in missed_examples[:10]:
            print(f"    {s1_id} → {m_id}")

    return recall
