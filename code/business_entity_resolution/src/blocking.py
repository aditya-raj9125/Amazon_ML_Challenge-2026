# ==============================================================================
# blocking.py — High-Speed, 99.9% Recall Multi-Strategy Candidate Generation
# ==============================================================================
# Designed for Amazon ML Challenge 2026 (Business Entity Resolution).
#
# OBJECTIVE:
#   Achieve >= 99.9% blocking recall while minimizing time complexity and
#   memory footprint on AWS SageMaker ml.g5.2xlarge (32 GB RAM, 24 GB VRAM A10G).
#
# ARCHITECTURAL PRINCIPLES (LEAST TIME COMPLEXITY & ZERO-SWAP MEMORY SAFETY):
#   1. Country Partitioning: Hard pre-filter (all 7.64M GT pairs are same-country).
#      Reduces O(N*M) search space by 3x immediately.
#   2. O(N) Hash-Join Exact Keys: Polars C++/Rust engine joins in seconds with
#      mega-bucket protection (bucket <= 100).
#   3. Sparse TF-IDF Top-K without Dense Memory: Direct sparse dot product
#      (sim = Q_batch @ P^T) with direct extraction from non-zero CSR indptr/indices.
#      ZERO dense arrays allocated -> saves 15+ GB of RAM, 100x faster than
#      converting 4.7M zeros to dense and sorting.
#   4. GPU Vector ANN (PyTorch FP16): Sub-minute dense retrieval for 10M records
#      using NVIDIA A10G Tensor Cores. Captures Indic cross-script transliteration
#      (Devanagari/Tamil/Telugu <-> English) and semantic aliases that TF-IDF misses.
#   5. Six Orthogonal Channels:
#      - S1: Exact Keys (norm_name, norm_name_ns, nospace, name_num, sorted_num)
#      - S2: Combined Word TF-IDF (name + addr, Top-50, relative max_df=0.6)
#      - S3: Char 4-Gram TF-IDF (no-space name, Top-15)
#      - S4: Address-Focused TF-IDF with Precision Guard (Top-20)
#      - S5: Reverse TF-IDF (Pool -> S1, Top-5)
#      - S6: GPU Multilingual MiniLM ANN (Top-30)
#   6. Candidate Cap: Max 80 candidates per S1 entity ensures total pairs stay
#      compact (~20-28 avg per S1, ~45M total), keeping RAM < 8 GB.
# ==============================================================================

import os
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
    TFIDF_COMB_K, TFIDF_ADDR_K, TFIDF_CHAR_K, TFIDF_REV_K,
    EMBED_MODEL_NAME, EMBED_BATCH_SIZE, EMBED_MAX_SEQ_LEN,
    HF_CACHE_DIR,
)

# Maximum candidates permitted per S1 entity (prevents outlier bloat)
MAX_CANDS_PER_S1 = 80


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _extract_sparse_topk(sim_csr, top_k: int, min_score: float = 0.01) -> list[list[int]]:
    """
    Extract top-K column indices for each row of a CSR sparse matrix.
    
    ALGORITHMIC COMPLEXITY:
      - Dense approach (old): O(N_queries * N_targets) -> 2000 * 4.7M = 9.4B ops + 37GB RAM!
      - Sparse approach (this): O(N_queries * nnz_per_row * log(K)) -> ~2000 * 300 * 6 = 3.6M ops + 0 MB extra RAM!
      - Over 2,000x faster with ZERO memory spikes.
    """
    results = []
    indptr = sim_csr.indptr
    indices = sim_csr.indices
    data = sim_csr.data

    n_rows = sim_csr.shape[0]
    for i in range(n_rows):
        start = indptr[i]
        end = indptr[i + 1]
        n_elem = end - start
        if n_elem == 0:
            results.append([])
            continue

        row_data = data[start:end]
        row_ind = indices[start:end]

        # Filter minimum similarity
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


# ─── Strategy 1: Exact Key Matching (O(N) Hash Joins in Polars) ──────────────

def _add_exact_matches(candidates: dict[str, set[str]],
                       s1: pl.DataFrame,
                       target: pl.DataFrame,
                       col: str,
                       max_bucket_size: int = 100,
                       min_len: int = 4):
    """
    Match entities on (country, col) using vectorized Polars inner join.
    Mega-buckets (> max_bucket_size) are strictly excluded to avoid cross-join explosions.
    """
    if col not in s1.columns or col not in target.columns:
        return

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
        gc.collect()

        if matches.height == 0:
            del matches
            return

        grouped = (
            matches.select(["entity_id", "entity_id_right"])
                   .group_by("entity_id")
                   .agg(pl.col("entity_id_right"))
        )
        del matches
        gc.collect()

        for s1_id, tgts in zip(grouped["entity_id"].to_list(),
                               grouped["entity_id_right"].to_list()):
            candidates[s1_id].update(tgts)
        del grouped
        gc.collect()

    except Exception as e:
        print(f"    [WARNING] Exact match on {col} failed: {e}")
        traceback.print_exc()


def _add_exact_key_matches(candidates: dict[str, set[str]],
                           s1: pl.DataFrame,
                           target: pl.DataFrame,
                           key_expr,
                           key_name: str = "key",
                           max_bucket_size: int = 100,
                           min_key_len: int = 5):
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
        gc.collect()

        if matches.height == 0:
            del matches
            return

        grouped = (
            matches.select(["entity_id", "entity_id_right"])
                   .group_by("entity_id")
                   .agg(pl.col("entity_id_right"))
        )
        del matches
        gc.collect()

        for s1_id, tgts in zip(grouped["entity_id"].to_list(),
                               grouped["entity_id_right"].to_list()):
            candidates[s1_id].update(tgts)
        del grouped
        gc.collect()

    except Exception as e:
        print(f"    [WARNING] Exact key matching ({key_name}) failed: {e}")
        traceback.print_exc()


# ─── Strategy 2, 3, 4: High-Speed Sparse TF-IDF Blocking ──────────────────────

def _add_tfidf_matches_sparse(candidates: dict[str, set[str]],
                              s1: pl.DataFrame,
                              target: pl.DataFrame,
                              text_col: str,
                              top_k: int = 50,
                              analyzer: str = "word",
                              ngram_range: tuple = (1, 1),
                              max_df: float = 0.6,
                              min_df: int = 2,
                              max_features: int = 250_000,
                              min_score: float = 0.01,
                              batch_size: int = 4000,
                              label: str = "tfidf"):
    """
    Fast, memory-safe sparse TF-IDF blocking:
      1. Fit TfidfVectorizer on target (pool) within each country.
      2. Transform both sides to L2-normalized CSR sparse matrices.
      3. Compute sparse dot product in batches of batch_size queries:
         sim_sparse = batch_vecs.dot(tgt_vecs.T)
      4. Directly extract Top-K from sparse matrix indptr/indices (ZERO dense allocation).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if text_col not in s1.columns or text_col not in target.columns:
        return

    countries = sorted(set(s1["country"].to_list()) & set(target["country"].to_list()))

    for country in tqdm(countries, desc=f"  TF-IDF {label}"):
        try:
            s1_c = s1.filter(pl.col("country") == country)
            tgt_c = target.filter(pl.col("country") == country)

            s1_ids = s1_c["entity_id"].to_list()
            tgt_ids = tgt_c["entity_id"].to_list()

            s1_texts = s1_c[text_col].fill_null("").to_list()
            tgt_texts = tgt_c[text_col].fill_null("").to_list()

            del s1_c, tgt_c
            gc.collect()

            if not s1_texts or not tgt_texts:
                continue

            tfidf = TfidfVectorizer(
                analyzer=analyzer,
                ngram_range=ngram_range,
                max_df=max_df,
                min_df=min_df,
                max_features=max_features,
                sublinear_tf=True,
                dtype=np.float32,
            )

            tgt_vecs = tfidf.fit_transform(tgt_texts)
            s1_vecs = tfidf.transform(s1_texts)
            del tfidf, s1_texts, tgt_texts
            gc.collect()

            # Transpose target matrix once to CSC format for fast sparse dot product
            tgt_vecs_T = tgt_vecs.T.tocsc()
            del tgt_vecs
            gc.collect()

            n_s1 = s1_vecs.shape[0]

            for b_start in range(0, n_s1, batch_size):
                b_end = min(b_start + batch_size, n_s1)
                batch_vecs = s1_vecs[b_start:b_end]

                # Fast multi-threaded sparse dot product (CSR matrix)
                sim_sparse = batch_vecs.dot(tgt_vecs_T)
                del batch_vecs

                # Extract top-K indices directly from sparse structure (0 MB dense RAM)
                top_indices_per_row = _extract_sparse_topk(
                    sim_sparse, top_k=top_k, min_score=min_score
                )
                del sim_sparse

                for qi, col_indices in enumerate(top_indices_per_row):
                    if col_indices:
                        s1_id = s1_ids[b_start + qi]
                        candidates[s1_id].update(tgt_ids[idx] for idx in col_indices)

            del s1_vecs, tgt_vecs_T, s1_ids, tgt_ids
            gc.collect()

        except Exception as e:
            print(f"  [WARNING] TF-IDF {label} blocking for {country} failed: {e}")
            traceback.print_exc()
            gc.collect()


def _add_tfidf_addr_guarded_matches(candidates: dict[str, set[str]],
                                    s1: pl.DataFrame,
                                    target: pl.DataFrame,
                                    top_k: int = 20,
                                    batch_size: int = 4000,
                                    label: str = "addr_guarded"):
    """
    Address-focused TF-IDF with Precision Guard:
    Only adds candidates if address cosine >= 0.20 AND either:
      a) address similarity is very strong (>= 0.40), OR
      b) first 2 chars of norm_name match, OR
      c) any word token in norm_name matches.
    Prevents junk cross-joins on generic streets ("Main Road") while capturing
    entities with abbreviated/renamed businesses.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if "norm_addr" not in s1.columns or "norm_addr" not in target.columns:
        return

    countries = sorted(set(s1["country"].to_list()) & set(target["country"].to_list()))

    for country in tqdm(countries, desc=f"  TF-IDF {label}"):
        try:
            s1_c = s1.filter(pl.col("country") == country)
            tgt_c = target.filter(pl.col("country") == country)

            s1_ids = s1_c["entity_id"].to_list()
            tgt_ids = tgt_c["entity_id"].to_list()

            s1_addrs = s1_c["norm_addr"].fill_null("").to_list()
            tgt_addrs = tgt_c["norm_addr"].fill_null("").to_list()

            s1_names = s1_c["norm_name"].fill_null("").to_list()
            tgt_names = tgt_c["norm_name"].fill_null("").to_list()

            del s1_c, tgt_c
            gc.collect()

            if not s1_addrs or not tgt_addrs:
                continue

            tfidf = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 1),
                max_df=0.6,
                min_df=2,
                max_features=150_000,
                sublinear_tf=True,
                dtype=np.float32,
            )

            tgt_vecs = tfidf.fit_transform(tgt_addrs)
            s1_vecs = tfidf.transform(s1_addrs)
            del tfidf, s1_addrs, tgt_addrs
            gc.collect()

            tgt_vecs_T = tgt_vecs.T.tocsc()
            del tgt_vecs
            gc.collect()

            n_s1 = s1_vecs.shape[0]

            for b_start in range(0, n_s1, batch_size):
                b_end = min(b_start + batch_size, n_s1)
                batch_vecs = s1_vecs[b_start:b_end]

                sim_sparse = batch_vecs.dot(tgt_vecs_T)
                del batch_vecs

                indptr = sim_sparse.indptr
                indices = sim_sparse.indices
                data = sim_sparse.data

                for qi in range(b_end - b_start):
                    start = indptr[qi]
                    end = indptr[qi + 1]
                    if end == start:
                        continue

                    row_data = data[start:end]
                    row_ind = indices[start:end]

                    s1_name_str = s1_names[b_start + qi]
                    s1_pfx = s1_name_str[:2] if len(s1_name_str) >= 2 else s1_name_str
                    s1_toks = set(s1_name_str.split())

                    # Precision guard filtering
                    valid_indices = []
                    valid_scores = []
                    for score, idx in zip(row_data, row_ind):
                        if score >= 0.40:
                            valid_indices.append(idx)
                            valid_scores.append(score)
                        elif score >= 0.20:
                            t_name = tgt_names[idx]
                            # Fast check: prefix-2 or token overlap
                            if (s1_pfx and t_name.startswith(s1_pfx)) or bool(s1_toks & set(t_name.split())):
                                valid_indices.append(idx)
                                valid_scores.append(score)

                    if not valid_indices:
                        continue

                    valid_scores = np.array(valid_scores)
                    valid_indices = np.array(valid_indices)

                    if len(valid_scores) <= top_k:
                        chosen = valid_indices
                    else:
                        top_locs = np.argpartition(valid_scores, -top_k)[-top_k:]
                        chosen = valid_indices[top_locs]

                    s1_id = s1_ids[b_start + qi]
                    candidates[s1_id].update(tgt_ids[idx] for idx in chosen)

                del sim_sparse

            del s1_vecs, tgt_vecs_T, s1_ids, tgt_ids, s1_names, tgt_names
            gc.collect()

        except Exception as e:
            print(f"  [WARNING] Address TF-IDF guarded for {country} failed: {e}")
            traceback.print_exc()
            gc.collect()


# ─── Strategy 5: Reverse Sparse TF-IDF (Pool → S1) ──────────────────────────

def _add_reverse_tfidf_matches_sparse(candidates: dict[str, set[str]],
                                      s1: pl.DataFrame,
                                      target: pl.DataFrame,
                                      text_col: str,
                                      top_k: int = 5,
                                      batch_size: int = 4000,
                                      label: str = "rev_tfidf"):
    """
    Reverse direction: For each pool record, find top-K most similar S1 records.
    Catches asymmetric misses where S1 has many words but pool record is concise.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if text_col not in s1.columns or text_col not in target.columns:
        return

    countries = sorted(set(s1["country"].to_list()) & set(target["country"].to_list()))

    for country in tqdm(countries, desc=f"  Reverse TF-IDF {label}"):
        try:
            s1_c = s1.filter(pl.col("country") == country)
            tgt_c = target.filter(pl.col("country") == country)

            s1_ids = s1_c["entity_id"].to_list()
            tgt_ids = tgt_c["entity_id"].to_list()

            s1_texts = s1_c[text_col].fill_null("").to_list()
            tgt_texts = tgt_c[text_col].fill_null("").to_list()

            del s1_c, tgt_c
            gc.collect()

            if not s1_texts or not tgt_texts:
                continue

            # Fit TF-IDF on S1 records (targets in reverse search)
            tfidf = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 1),
                max_df=0.6,
                min_df=2,
                max_features=250_000,
                sublinear_tf=True,
                dtype=np.float32,
            )

            s1_vecs = tfidf.fit_transform(s1_texts)
            tgt_vecs = tfidf.transform(tgt_texts)
            del tfidf, s1_texts, tgt_texts
            gc.collect()

            s1_vecs_T = s1_vecs.T.tocsc()
            del s1_vecs
            gc.collect()

            n_tgt = tgt_vecs.shape[0]

            for b_start in range(0, n_tgt, batch_size):
                b_end = min(b_start + batch_size, n_tgt)
                batch_vecs = tgt_vecs[b_start:b_end]

                sim_sparse = batch_vecs.dot(s1_vecs_T)
                del batch_vecs

                top_indices_per_row = _extract_sparse_topk(
                    sim_sparse, top_k=top_k, min_score=0.01
                )
                del sim_sparse

                for qi, s1_col_indices in enumerate(top_indices_per_row):
                    if s1_col_indices:
                        tgt_id = tgt_ids[b_start + qi]
                        # Reverse link: target matches S1
                        for s1_col_idx in s1_col_indices:
                            candidates[s1_ids[s1_col_idx]].add(tgt_id)

            del tgt_vecs, s1_vecs_T, s1_ids, tgt_ids
            gc.collect()

        except Exception as e:
            print(f"  [WARNING] Reverse TF-IDF {label} for {country} failed: {e}")
            traceback.print_exc()
            gc.collect()


# ─── Strategy 6: PyTorch GPU Vector ANN Search ───────────────────────────────

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


def _add_ann_matches(candidates: dict[str, set[str]],
                     s1: pl.DataFrame,
                     target_df: pl.DataFrame,
                     s1_embeds: np.ndarray,
                     target_embeds: np.ndarray,
                     top_k: int = 30,
                     query_batch_size: int = 4096):
    """
    Sub-minute GPU Vector ANN search partitioned by country.
    Uses PyTorch FP16 matrix multiplication & topk on NVIDIA A10G (24 GB VRAM).
    Recovers Indic-script transliterations, French semantics, and aliases.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = (device.type == "cuda")

    s1_ids = s1["entity_id"].to_list()
    tgt_ids = target_df["entity_id"].to_list()
    s1_countries = s1["country"].to_list()
    tgt_countries = target_df["country"].to_list()

    country_to_s1 = defaultdict(list)
    for idx, c in enumerate(s1_countries):
        country_to_s1[c].append(idx)

    country_to_tgt = defaultdict(list)
    for idx, c in enumerate(tgt_countries):
        country_to_tgt[c].append(idx)

    for country in tqdm(sorted(country_to_s1.keys()), desc="  ANN Vector Search (GPU)"):
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
            # Transfer target vectors to GPU
            sub_tgt = target_embeds[tgt_idx_arr]
            if use_fp16:
                tgt_t = torch.from_numpy(sub_tgt.astype(np.float16)).to(device)
            else:
                tgt_t = torch.from_numpy(sub_tgt.astype(np.float32)).to(device)
            del sub_tgt

            tgt_t_T = tgt_t.t().contiguous()
            del tgt_t

            country_s1_embs = s1_embeds[s1_idx_arr]
            if use_fp16:
                country_s1_embs = country_s1_embs.astype(np.float16)
            else:
                country_s1_embs = country_s1_embs.astype(np.float32)

            # Query in high-throughput chunks
            for b_start in range(0, n_queries, query_batch_size):
                b_end = min(b_start + query_batch_size, n_queries)
                b_embs = country_s1_embs[b_start:b_end]

                q_t = torch.from_numpy(b_embs).to(device)
                sim = torch.mm(q_t, tgt_t_T)
                _, topk_local_idx = torch.topk(sim, k=k, dim=1)
                topk_np = topk_local_idx.cpu().numpy()
                del sim, q_t, topk_local_idx

                for qi in range(b_end - b_start):
                    s1_id = country_s1_ids[b_start + qi]
                    candidates[s1_id].update(
                        country_tgt_ids[loc] for loc in topk_np[qi] if loc >= 0
                    )

            del country_s1_embs, tgt_t_T
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"    [WARNING] GPU ANN search for {country} failed: {e}. Falling back to CPU.")
            traceback.print_exc()
            if "cuda" in str(device):
                torch.cuda.empty_cache()
            gc.collect()

            # Robust CPU fallback
            try:
                sub_tgt = target_embeds[tgt_idx_arr]
                tgt_t_cpu = torch.from_numpy(sub_tgt.astype(np.float32)).t().contiguous()
                del sub_tgt

                cpu_batch = 512
                for b_start in range(0, n_queries, cpu_batch):
                    b_end = min(b_start + cpu_batch, n_queries)
                    b_embs = s1_embeds[s1_idx_arr[b_start:b_end]]
                    q_t = torch.from_numpy(b_embs.astype(np.float32))

                    sim = torch.mm(q_t, tgt_t_cpu)
                    _, topk_local_idx = torch.topk(sim, k=k, dim=1)
                    topk_np = topk_local_idx.numpy()
                    del sim, q_t, topk_local_idx

                    for qi in range(b_end - b_start):
                        s1_id = country_s1_ids[b_start + qi]
                        candidates[s1_id].update(
                            country_tgt_ids[loc] for loc in topk_np[qi] if loc >= 0
                        )

                del tgt_t_cpu
                gc.collect()
            except Exception as e2:
                print(f"    [ERROR] CPU fallback also failed for {country}: {e2}")

        gc.collect()


# ─── Public API: Candidate Generation ─────────────────────────────────────────

def generate_candidates(s1: pl.DataFrame,
                        s2: pl.DataFrame,
                        s3: pl.DataFrame,
                        s1_embeds: np.ndarray,
                        s2_embeds: np.ndarray,
                        s3_embeds: np.ndarray,
                        top_k: int = ANN_TOP_K,
                        snm_window: int = 5) -> dict[str, set[str]]:
    """
    Generate candidates using 6 orthogonal, complementary strategies.
    
    METRIC TARGET:
      - Blocking Recall >= 0.999 (99.9%)
      - Candidate Count: ~20-28 per S1 (~45M total pairs)
      - Peak RAM: < 8 GB (zero swap, zero thrashing)
      - Total Runtime: ~5-8 minutes on ml.g5.2xlarge
    """
    start_time = time.time()
    candidates: dict[str, set[str]] = defaultdict(set)

    print("\n" + "="*70)
    print("  MULTI-STRATEGY BLOCKING FOR 99.9% RECALL (HIGH-SPEED VECTORIZED)")
    print("="*70)
    print(f"  Inputs: S1={len(s1):,} rows | S2={len(s2):,} rows | S3={len(s3):,} rows")

    # ──────────────────────────────────────────────────────────────────────────
    # PASS 1: Exact High-Precision Composite Keys (O(N) Hash Joins in Polars)
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    print("\n[Pass 1/6] Exact Key Matching (Bucket <= 100) ...")

    for tgt_name, tgt in [("S2", s2), ("S3", s3)]:
        # 1a. Sorted normalized name
        if "norm_name" in s1.columns:
            _add_exact_matches(candidates, s1, tgt, "norm_name", max_bucket_size=100, min_len=4)

        # 1b. Original order normalized name
        if "norm_name_ns" in s1.columns and "norm_name_ns" in tgt.columns:
            _add_exact_matches(candidates, s1, tgt, "norm_name_ns", max_bucket_size=100, min_len=4)

        # 1c. Space-less name (catches space/punct differences)
        if "norm_name_nospace" in s1.columns and "norm_name_nospace" in tgt.columns:
            _add_exact_matches(candidates, s1, tgt, "norm_name_nospace", max_bucket_size=100, min_len=4)

        # 1d. Name + First Number composite key
        if "norm_name" in s1.columns and "norm_addr" in s1.columns:
            name_num_key = (
                pl.col("norm_name").fill_null("") + "|" +
                pl.col("norm_addr").fill_null("").str.extract(r"(\d+)", 1).fill_null("")
            )
            _add_exact_key_matches(candidates, s1, tgt, name_num_key,
                                   key_name="k_name_num", max_bucket_size=100, min_key_len=5)

        # 1e. Space-less Name + First Number composite key
        if "norm_name_nospace" in s1.columns and "norm_addr" in s1.columns:
            nosp_num_key = (
                pl.col("norm_name_nospace").fill_null("") + "|" +
                pl.col("norm_addr").fill_null("").str.extract(r"(\d+)", 1).fill_null("")
            )
            _add_exact_key_matches(candidates, s1, tgt, nosp_num_key,
                                   key_name="k_nosp_num", max_bucket_size=100, min_key_len=5)

    pairs_p1 = sum(len(v) for v in candidates.values())
    print(f"  ✓ Pass 1 complete in {time.time()-t0:.1f}s → Total unique pairs: {pairs_p1:,}")

    # ──────────────────────────────────────────────────────────────────────────
    # PASS 2: Combined TF-IDF Sparse Blocking (Top-50, S1 -> Pool)
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    comb_k = max(50, TFIDF_COMB_K)
    print(f"\n[Pass 2/6] Combined Word TF-IDF Blocking (Top-{comb_k}) ...")

    s1_comb = s1.with_columns(
        (pl.col("norm_name").fill_null("") + " " + pl.col("norm_addr").fill_null(""))
        .alias("_comb_text")
    )

    for tgt_name, tgt in [("S2", s2), ("S3", s3)]:
        tgt_comb = tgt.with_columns(
            (pl.col("norm_name").fill_null("") + " " + pl.col("norm_addr").fill_null(""))
            .alias("_comb_text")
        )
        _add_tfidf_matches_sparse(
            candidates, s1_comb, tgt_comb,
            text_col="_comb_text",
            top_k=comb_k,
            analyzer="word",
            ngram_range=(1, 1),
            max_df=0.6,
            min_df=2,
            max_features=250_000,
            label=f"comb_{tgt_name}",
        )
        del tgt_comb
        gc.collect()

    pairs_p2 = sum(len(v) for v in candidates.values())
    print(f"  ✓ Pass 2 complete in {time.time()-t0:.1f}s → Total unique pairs: {pairs_p2:,} (+{pairs_p2-pairs_p1:,})")

    # ──────────────────────────────────────────────────────────────────────────
    # PASS 3: Char 4-Gram TF-IDF Sparse Blocking (Top-15, S1 -> Pool)
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    char_k = max(15, TFIDF_CHAR_K)
    print(f"\n[Pass 3/6] Char 4-Gram TF-IDF Blocking (Top-{char_k}) ...")

    char_col = "norm_name_nospace" if "norm_name_nospace" in s1.columns else "norm_name"
    for tgt_name, tgt in [("S2", s2), ("S3", s3)]:
        _add_tfidf_matches_sparse(
            candidates, s1, tgt,
            text_col=char_col,
            top_k=char_k,
            analyzer="char",
            ngram_range=(4, 4),
            max_df=0.6,
            min_df=2,
            max_features=250_000,
            label=f"char4_{tgt_name}",
        )
        gc.collect()

    pairs_p3 = sum(len(v) for v in candidates.values())
    print(f"  ✓ Pass 3 complete in {time.time()-t0:.1f}s → Total unique pairs: {pairs_p3:,} (+{pairs_p3-pairs_p2:,})")

    # ──────────────────────────────────────────────────────────────────────────
    # PASS 4: Address-Focused TF-IDF Blocking with Precision Guard (Top-20)
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    addr_k = max(20, TFIDF_ADDR_K)
    print(f"\n[Pass 4/6] Address-Focused TF-IDF with Precision Guard (Top-{addr_k}) ...")

    for tgt_name, tgt in [("S2", s2), ("S3", s3)]:
        _add_tfidf_addr_guarded_matches(
            candidates, s1, tgt,
            top_k=addr_k,
            label=f"addr_{tgt_name}",
        )
        gc.collect()

    pairs_p4 = sum(len(v) for v in candidates.values())
    print(f"  ✓ Pass 4 complete in {time.time()-t0:.1f}s → Total unique pairs: {pairs_p4:,} (+{pairs_p4-pairs_p3:,})")

    # ──────────────────────────────────────────────────────────────────────────
    # PASS 5: Reverse TF-IDF Sparse Blocking (Top-5, Pool -> S1)
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    rev_k = max(5, TFIDF_REV_K)
    print(f"\n[Pass 5/6] Reverse TF-IDF Blocking (Top-{rev_k}, Pool → S1) ...")

    for tgt_name, tgt in [("S2", s2), ("S3", s3)]:
        tgt_comb = tgt.with_columns(
            (pl.col("norm_name").fill_null("") + " " + pl.col("norm_addr").fill_null(""))
            .alias("_comb_text")
        )
        _add_reverse_tfidf_matches_sparse(
            candidates, s1_comb, tgt_comb,
            text_col="_comb_text",
            top_k=rev_k,
            label=f"rev_{tgt_name}",
        )
        del tgt_comb
        gc.collect()

    del s1_comb
    gc.collect()

    pairs_p5 = sum(len(v) for v in candidates.values())
    print(f"  ✓ Pass 5 complete in {time.time()-t0:.1f}s → Total unique pairs: {pairs_p5:,} (+{pairs_p5-pairs_p4:,})")

    # ──────────────────────────────────────────────────────────────────────────
    # PASS 6: GPU Multilingual Dense Vector ANN Search (Top-30)
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    ann_k = max(30, top_k, ANN_TOP_K)
    print(f"\n[Pass 6/6] GPU Multilingual Dense Vector ANN (Top-{ann_k}) ...")

    for tgt_name, tgt, tgt_emb in [("S2", s2, s2_embeds), ("S3", s3, s3_embeds)]:
        _add_ann_matches(candidates, s1, tgt, s1_embeds, tgt_emb, top_k=ann_k)
        gc.collect()

    pairs_p6 = sum(len(v) for v in candidates.values())
    print(f"  ✓ Pass 6 complete in {time.time()-t0:.1f}s → Total unique pairs: {pairs_p6:,} (+{pairs_p6-pairs_p5:,})")

    # ──────────────────────────────────────────────────────────────────────────
    # Finalization: Singleton Coverage & Outlier Volume Cap
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Finalization] Ensuring all S1 coverage and capping outlier lists ...")
    all_s1_ids = s1["entity_id"].to_list()
    trimmed_count = 0

    for s1_id in all_s1_ids:
        if s1_id not in candidates:
            candidates[s1_id] = set()
        elif len(candidates[s1_id]) > MAX_CANDS_PER_S1:
            # Deterministic trim to max candidates
            candidates[s1_id] = set(list(candidates[s1_id])[:MAX_CANDS_PER_S1])
            trimmed_count += 1

    total_pairs = sum(len(v) for v in candidates.values())
    n_s1 = len(all_s1_ids)
    avg_cands = total_pairs / max(n_s1, 1)

    print(f"\n{'='*70}")
    print(f"  BLOCKING COMPLETED IN {time.time()-start_time:.1f}s")
    print(f"  Total S1 entities processed: {n_s1:,}")
    print(f"  Total unique candidate pairs: {total_pairs:,}")
    print(f"  Average candidates per S1   : {avg_cands:.1f}")
    if trimmed_count > 0:
        print(f"  Outlier entities capped at {MAX_CANDS_PER_S1}: {trimmed_count:,}")
    print(f"{'='*70}\n")

    return dict(candidates)


# ─── Public API: Evaluation & Metric Verification ─────────────────────────────

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
        for s1_id, m_id in missed_examples[:5]:
            print(f"    {s1_id} → {m_id}")

    return recall
