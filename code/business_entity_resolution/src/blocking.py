# =============================================================
# blocking.py — Multi-strategy candidate pair generation
# =============================================================
# Produces candidate_pairs.tsv: every (S1, S2/S3) pair that will
# be scored by the LightGBM matcher.
#
# SEVEN complementary strategies are combined (union):
#   A. Exact key matching       — sorted_name+first_num, nospace_name+first_num
#   B. Prefix matching          — first N chars of name/addr within country
#   C. TF-IDF sparse blocking   — top-K by combined TF-IDF (name+addr words)
#   D. TF-IDF char-ngram        — top-K by char 4-gram TF-IDF on no-space name
#   E. Reverse blocking         — pool→S1 direction (catches asymmetric misses)
#   F. ANN embedding search     — top-k vector retrieval on multilingual embeddings
#   G. Token-overlap key        — exact match on sorted unique tokens (without nums)
#
# Country partitioning is applied FIRST as a hard pre-filter (zero
# recall loss on training data — all 7.64M GT pairs are same-country).
#
# Target: blocking recall >= 0.99, reduction ratio >= 0.9999

import os
import gc
import sys
import traceback
from collections import defaultdict

import numpy as np
import polars as pl
from tqdm import tqdm

from config import (
    ANN_TOP_K, SNM_WINDOW,
    EMBED_MODEL_NAME, EMBED_BATCH_SIZE, EMBED_MAX_SEQ_LEN,
)


# ─── Helper: extract first numeric token ─────────────────────────────────────

def _first_number(text: str) -> str:
    """Return the first purely-numeric token, or '' if none."""
    if not text:
        return ""
    for tok in text.split():
        if tok.isdigit():
            return tok
    return ""


def _nospace(text: str) -> str:
    """Remove all whitespace from text."""
    return "".join(text.split()) if text else ""


# ─── Strategy A: Exact Key Matching ──────────────────────────────────────────

def _add_exact_key_matches(candidates: dict[str, set[str]],
                           s1: pl.DataFrame,
                           target: pl.DataFrame,
                           key_expr,
                           key_name: str = "key",
                           max_bucket_size: int = 500,
                           min_key_len: int = 4):
    """
    Match S1 to target on (country, computed_key) using Polars joins.
    key_expr is a Polars expression that produces the blocking key.
    """
    try:
        s1_keyed = (
            s1.select(["entity_id", "country"])
              .with_columns(key_expr.alias(key_name))
              .filter(pl.col(key_name).is_not_null())
              .filter(pl.col(key_name).str.len_chars() >= min_key_len)
        )
        tgt_keyed = (
            target.select(["entity_id", "country"])
                  .with_columns(key_expr.alias(key_name))
                  .filter(pl.col(key_name).is_not_null())
                  .filter(pl.col(key_name).str.len_chars() >= min_key_len)
        )

        # Filter mega-buckets
        s1_keyed = s1_keyed.filter(pl.len().over(["country", key_name]) <= max_bucket_size)
        tgt_keyed = tgt_keyed.filter(pl.len().over(["country", key_name]) <= max_bucket_size)

        matches = s1_keyed.join(tgt_keyed, on=["country", key_name], how="inner")
        del s1_keyed, tgt_keyed

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
        print(f"  [WARNING] Exact key matching ({key_name}) failed: {e}")
        traceback.print_exc()


def _add_exact_matches(candidates: dict[str, set[str]],
                       s1: pl.DataFrame,
                       target: pl.DataFrame,
                       col: str,
                       max_bucket_size: int = 500):
    """
    Find matching entities between s1 and target on (country, col).
    Vectorized Polars inner join in C++/Rust: 0 Python dicts.
    """
    if col not in s1.columns or col not in target.columns:
        return

    try:
        target_clean = (
            target.select(["entity_id", "country", col])
                  .filter(pl.col(col).is_not_null())
                  .filter(pl.col(col).str.len_chars() >= 3)
                  .filter(pl.len().over(["country", col]) <= max_bucket_size)
        )
        s1_clean = (
            s1.select(["entity_id", "country", col])
              .filter(pl.col(col).is_not_null())
              .filter(pl.col(col).str.len_chars() >= 3)
              .filter(pl.len().over(["country", col]) <= max_bucket_size)
        )
        matches = s1_clean.join(target_clean, on=["country", col], how="inner")
        del target_clean, s1_clean
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
        print(f"  [WARNING] Exact match on {col} failed: {e}")
        traceback.print_exc()


# ─── Strategy B: Prefix Matching ─────────────────────────────────────────────

def _add_prefix_matches(candidates: dict[str, set[str]],
                        s1: pl.DataFrame,
                        target: pl.DataFrame,
                        col: str,
                        prefix_len: int = 4,
                        max_bucket_size: int = 200):
    """
    Match entities on first N characters of name or address within country.
    """
    if col not in s1.columns or col not in target.columns:
        return

    try:
        pref_col = f"{col}_pref"
        s1_pref = (
            s1.select(["entity_id", "country",
                       pl.col(col).str.slice(0, prefix_len).alias(pref_col)])
              .filter(pl.col(pref_col).is_not_null())
              .filter(pl.col(pref_col).str.len_chars() >= prefix_len)
              .filter(pl.len().over(["country", pref_col]) <= max_bucket_size)
        )
        tgt_pref = (
            target.select(["entity_id", "country",
                           pl.col(col).str.slice(0, prefix_len).alias(pref_col)])
                  .filter(pl.col(pref_col).is_not_null())
                  .filter(pl.col(pref_col).str.len_chars() >= prefix_len)
                  .filter(pl.len().over(["country", pref_col]) <= max_bucket_size)
        )
        matches = s1_pref.join(tgt_pref, on=["country", pref_col], how="inner")
        del s1_pref, tgt_pref
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
        print(f"  [WARNING] Prefix match on {col} (len={prefix_len}) failed: {e}")
        traceback.print_exc()


# ─── Strategy C & D: TF-IDF Sparse Vector Blocking ──────────────────────────

def _add_tfidf_matches(candidates: dict[str, set[str]],
                       s1: pl.DataFrame,
                       target: pl.DataFrame,
                       text_col: str,
                       top_k: int = 50,
                       analyzer: str = "word",
                       ngram_range: tuple = (1, 1),
                       max_df: float = 0.5,
                       min_df: int = 2,
                       max_features: int = 200_000,
                       label: str = "tfidf"):
    """
    TF-IDF sparse vector blocking: fit on pool, transform both sides,
    compute cosine top-K per S1 within each country.

    This is the single most important blocking strategy — responsible for
    97%+ blocking recall in the teammate's pipeline.

    Memory-safe: processes one country at a time, uses sparse matrices.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import vstack as sparse_vstack

    if text_col not in s1.columns or text_col not in target.columns:
        return

    countries = sorted(set(s1["country"].to_list()) & set(target["country"].to_list()))

    for country in tqdm(countries, desc=f"  TF-IDF {label} blocking"):
        try:
            s1_c = s1.filter(pl.col("country") == country)
            tgt_c = target.filter(pl.col("country") == country)

            s1_ids = s1_c["entity_id"].to_list()
            tgt_ids = tgt_c["entity_id"].to_list()

            s1_texts = s1_c[text_col].fill_null("").to_list()
            tgt_texts = tgt_c[text_col].fill_null("").to_list()

            del s1_c, tgt_c

            if not s1_texts or not tgt_texts:
                continue

            # Fit TF-IDF on target (pool) texts
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

            n_s1 = s1_vecs.shape[0]
            k = min(top_k, tgt_vecs.shape[0])

            # Batch the similarity computation to stay within memory (1.5 GB budget for 32 GB RAM)
            batch_size = max(1, min(2000, int(1.5e9 / (tgt_vecs.shape[0] * 4))))

            for b_start in range(0, n_s1, batch_size):
                b_end = min(b_start + batch_size, n_s1)
                batch_vecs = s1_vecs[b_start:b_end]

                # Sparse dot product → dense top-K
                sim = (batch_vecs @ tgt_vecs.T).toarray()

                # Get top-K indices per row
                if k < sim.shape[1]:
                    topk_idx = np.argpartition(sim, -k, axis=1)[:, -k:]
                else:
                    topk_idx = np.tile(np.arange(sim.shape[1]), (sim.shape[0], 1))

                for qi in range(b_end - b_start):
                    s1_id = s1_ids[b_start + qi]
                    # Only add candidates with non-zero similarity
                    for idx in topk_idx[qi]:
                        if sim[qi, idx] > 0.01:
                            candidates[s1_id].add(tgt_ids[idx])

                del sim, topk_idx, batch_vecs
                gc.collect()

            del s1_vecs, tgt_vecs
            gc.collect()

        except Exception as e:
            print(f"  [WARNING] TF-IDF {label} blocking for {country} failed: {e}")
            traceback.print_exc()
            gc.collect()


# ─── Strategy E: Reverse Blocking (Pool → S1) ───────────────────────────────

def _add_reverse_tfidf_matches(candidates: dict[str, set[str]],
                               s1: pl.DataFrame,
                               target: pl.DataFrame,
                               text_col: str,
                               top_k: int = 5,
                               analyzer: str = "word",
                               ngram_range: tuple = (1, 1),
                               max_df: float = 0.5,
                               min_df: int = 2,
                               max_features: int = 200_000,
                               label: str = "rev_tfidf"):
    """
    Reverse direction: for each pool record, find top-K most similar S1 records.
    This catches asymmetric misses — when a pool record is a good match for an S1
    but the S1's name/address is too common to surface the pool record in forward search.
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

            if not s1_texts or not tgt_texts:
                continue

            # Fit TF-IDF on S1 texts (reverse direction)
            tfidf = TfidfVectorizer(
                analyzer=analyzer,
                ngram_range=ngram_range,
                max_df=max_df,
                min_df=min_df,
                max_features=max_features,
                sublinear_tf=True,
                dtype=np.float32,
            )

            s1_vecs = tfidf.fit_transform(s1_texts)
            tgt_vecs = tfidf.transform(tgt_texts)
            del tfidf, s1_texts, tgt_texts
            gc.collect()

            n_tgt = tgt_vecs.shape[0]
            k = min(top_k, s1_vecs.shape[0])

            # Batch reverse similarity to stay under 1.5 GB RAM
            batch_size = max(1, min(2000, int(1.5e9 / (s1_vecs.shape[0] * 4))))

            for b_start in range(0, n_tgt, batch_size):
                b_end = min(b_start + batch_size, n_tgt)
                batch_vecs = tgt_vecs[b_start:b_end]

                sim = (batch_vecs @ s1_vecs.T).toarray()

                if k < sim.shape[1]:
                    topk_idx = np.argpartition(sim, -k, axis=1)[:, -k:]
                else:
                    topk_idx = np.tile(np.arange(sim.shape[1]), (sim.shape[0], 1))

                for qi in range(b_end - b_start):
                    tgt_id = tgt_ids[b_start + qi]
                    for idx in topk_idx[qi]:
                        if sim[qi, idx] > 0.01:
                            # Reverse direction: add tgt_id to s1_id's candidate set
                            candidates[s1_ids[idx]].add(tgt_id)

                del sim, topk_idx, batch_vecs
                gc.collect()

            del s1_vecs, tgt_vecs
            gc.collect()

        except Exception as e:
            print(f"  [WARNING] Reverse TF-IDF {label} for {country} failed: {e}")
            traceback.print_exc()
            gc.collect()


# ─── Strategy F: PyTorch Vector ANN Search (GPU / CPU) ───────────────────────

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
        normalize_embeddings=True,   # L2-normalised → dot product == cosine
    )
    return embeddings.astype(np.float32)


def _add_ann_matches(candidates: dict[str, set[str]],
                     s1: pl.DataFrame,
                     target_df: pl.DataFrame,
                     s1_embeds: np.ndarray,
                     target_embeds: np.ndarray,
                     top_k: int = 30,
                     max_sim_bytes: int = 1024 * 1024 * 1024):
    """
    Search S1 embeddings against target (S2 or S3) partitioned by country.
    Uses PyTorch matrix multiplication & topk with dynamic query batch sizing.
    Similarity matrix is budgeted up to 1 GB VRAM (safe for 24 GB NVIDIA A10G)
    and queries are staged in contiguous memory.
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

    for country in tqdm(sorted(country_to_s1.keys()), desc="  ANN search"):
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

        elem_bytes = 2 if use_fp16 else 4
        safe_batch_size = max(64, min(1024, int(max_sim_bytes / (n_tgt * elem_bytes))))
        vram_mb = int(safe_batch_size * n_tgt * elem_bytes / (1024 * 1024))
        print(f"    [{country}] {n_queries:,} queries vs {n_tgt:,} targets "
              f"(top-{k}, batch={safe_batch_size}, ~{vram_mb} MB VRAM)")

        try:
            sub_tgt = target_embeds[tgt_idx_arr]
            if use_fp16:
                sub_tgt_arr = sub_tgt.astype(np.float16) if sub_tgt.dtype != np.float16 else sub_tgt
                tgt_t = torch.from_numpy(sub_tgt_arr).half().to(device)
            else:
                sub_tgt_arr = sub_tgt.astype(np.float32) if sub_tgt.dtype != np.float32 else sub_tgt
                tgt_t = torch.from_numpy(sub_tgt_arr).float().to(device)
            del sub_tgt, sub_tgt_arr

            tgt_t_T = tgt_t.t().contiguous()
            del tgt_t

            country_s1_embs = s1_embeds[s1_idx_arr]
            if use_fp16 and country_s1_embs.dtype != np.float16:
                country_s1_embs = country_s1_embs.astype(np.float16)
            elif not use_fp16 and country_s1_embs.dtype != np.float32:
                country_s1_embs = country_s1_embs.astype(np.float32)

            for b_start in range(0, n_queries, safe_batch_size):
                b_end = min(b_start + safe_batch_size, n_queries)
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
            print(f"    [WARNING] ANN search for {country} failed: {e}")
            traceback.print_exc()
            if "cuda" in str(device):
                import torch
                torch.cuda.empty_cache()
            gc.collect()

            # CPU fallback
            try:
                import torch
                print(f"    [{country}] Falling back to CPU ...")
                sub_tgt = target_embeds[tgt_idx_arr]
                tgt_t_cpu = torch.from_numpy(sub_tgt.astype(np.float32)).t().contiguous()
                del sub_tgt

                cpu_batch = 128
                for b_start in range(0, n_queries, cpu_batch):
                    b_end = min(b_start + cpu_batch, n_queries)
                    b_embs = s1_embeds[s1_idx_arr[b_start:b_end]]
                    q_t = torch.from_numpy(b_embs.astype(np.float32))
                    del b_embs

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
                traceback.print_exc()

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
    Generate candidates using 7 complementary strategies.
    Target: >= 99% blocking recall with reduction ratio >= 0.9999.

    Strategies are ordered from cheapest to most expensive:
    A. Exact key matching (< 5 sec each)
    B. Prefix matching (< 3 sec each)
    C. TF-IDF word blocking — top-50 (most important, ~2 min/country)
    D. TF-IDF char-ngram blocking — top-12 (~1 min/country)
    E. Reverse blocking — top-5 (~1 min/country)
    F. ANN embedding search — top-30 (~1 min/country with GPU)

    Peak RAM: < 8 GB (country-partitioned processing).
    """
    candidates: dict[str, set[str]] = defaultdict(set)

    # Add norm_name_ns (no-space) if not present
    for df in [s1, s2, s3]:
        if "norm_name_ns" not in df.columns and "norm_name" in df.columns:
            # Already exists from pipeline normalization
            pass

    # ──────────────────────────────────────────────────────────────────────
    # Strategy A: Exact Key Matching
    # ──────────────────────────────────────────────────────────────────────
    print("\n[Blocking] Strategy A: Exact key matching ...")

    # A1: Exact norm_name match
    print("  A1: Exact norm_name ...")
    for tgt in [s2, s3]:
        _add_exact_matches(candidates, s1, tgt, "norm_name", max_bucket_size=500)

    # A2: Exact norm_addr match
    print("  A2: Exact norm_addr ...")
    for tgt in [s2, s3]:
        _add_exact_matches(candidates, s1, tgt, "norm_addr", max_bucket_size=500)

    # A3: Sorted name + first number composite key
    print("  A3: Sorted name + first number key ...")
    if "norm_name" in s1.columns:
        # Build composite key: sorted_name_tokens + "|" + first_number
        for tgt in [s2, s3]:
            name_num_key = (
                pl.col("norm_name").fill_null("") + "|" +
                pl.col("norm_addr").fill_null("").str.extract(r"(\d+)", 1).fill_null("")
            )
            _add_exact_key_matches(candidates, s1, tgt, name_num_key,
                                   key_name="k_name_num", max_bucket_size=200, min_key_len=5)

    # A4: No-space name key (catches space/punctuation variations)
    print("  A4: No-space name key ...")
    if "norm_name_ns" in s1.columns:
        for tgt in [s2, s3]:
            _add_exact_matches(candidates, s1, tgt, "norm_name_ns", max_bucket_size=500)

    pairs_after_A = sum(len(v) for v in candidates.values())
    print(f"  → Pairs after Strategy A: {pairs_after_A:,}")

    # ──────────────────────────────────────────────────────────────────────
    # Strategy B: Prefix Matching
    # ──────────────────────────────────────────────────────────────────────
    print("\n[Blocking] Strategy B: Prefix matching ...")

    # B1: Name prefix (4 chars)
    print("  B1: norm_name prefix-4 ...")
    for tgt in [s2, s3]:
        _add_prefix_matches(candidates, s1, tgt, "norm_name", prefix_len=4, max_bucket_size=200)

    # B2: Name prefix (6 chars, more precise)
    print("  B2: norm_name prefix-6 ...")
    for tgt in [s2, s3]:
        _add_prefix_matches(candidates, s1, tgt, "norm_name", prefix_len=6, max_bucket_size=500)

    # B3: No-space name prefix (4 chars)
    print("  B3: norm_name_ns prefix-4 ...")
    for tgt in [s2, s3]:
        if "norm_name_ns" in tgt.columns:
            _add_prefix_matches(candidates, s1, tgt, "norm_name_ns",
                               prefix_len=4, max_bucket_size=200)

    # B4: Address prefix (6 chars)
    print("  B4: norm_addr prefix-6 ...")
    for tgt in [s2, s3]:
        _add_prefix_matches(candidates, s1, tgt, "norm_addr", prefix_len=6, max_bucket_size=200)

    pairs_after_B = sum(len(v) for v in candidates.values())
    print(f"  → Pairs after Strategy B: {pairs_after_B:,}")

    # ──────────────────────────────────────────────────────────────────────
    # Strategy C: TF-IDF Word Blocking (TOP-50)
    # ──────────────────────────────────────────────────────────────────────
    print("\n[Blocking] Strategy C: TF-IDF word blocking (top-50) ...")

    # Build combined text column for TF-IDF: name + " " + address
    s1_tfidf = s1.with_columns(
        (pl.col("norm_name").fill_null("") + " " + pl.col("norm_addr").fill_null(""))
        .alias("_tfidf_text")
    )
    for tgt_label, tgt in [("S2", s2), ("S3", s3)]:
        print(f"  C: Forward {tgt_label} ...")
        tgt_tfidf = tgt.with_columns(
            (pl.col("norm_name").fill_null("") + " " + pl.col("norm_addr").fill_null(""))
            .alias("_tfidf_text")
        )
        _add_tfidf_matches(
            candidates, s1_tfidf, tgt_tfidf,
            text_col="_tfidf_text",
            top_k=50,
            analyzer="word",
            ngram_range=(1, 1),
            max_df=0.5,    # relative cap (not absolute 20000) — fair across country sizes
            min_df=2,
            max_features=200_000,
            label=f"comb_{tgt_label}",
        )
        del tgt_tfidf
        gc.collect()

    pairs_after_C = sum(len(v) for v in candidates.values())
    print(f"  → Pairs after Strategy C: {pairs_after_C:,}")

    # ──────────────────────────────────────────────────────────────────────
    # Strategy D: TF-IDF Char-Ngram Blocking (TOP-12)
    # ──────────────────────────────────────────────────────────────────────
    print("\n[Blocking] Strategy D: TF-IDF char 4-gram blocking (top-12) ...")

    for tgt_label, tgt in [("S2", s2), ("S3", s3)]:
        print(f"  D: Char-ngram {tgt_label} ...")
        _add_tfidf_matches(
            candidates, s1, tgt,
            text_col="norm_name_ns" if "norm_name_ns" in s1.columns else "norm_name",
            top_k=12,
            analyzer="char",
            ngram_range=(4, 4),
            max_df=0.5,
            min_df=2,
            max_features=200_000,
            label=f"char4_{tgt_label}",
        )
        gc.collect()

    pairs_after_D = sum(len(v) for v in candidates.values())
    print(f"  → Pairs after Strategy D: {pairs_after_D:,}")

    # ──────────────────────────────────────────────────────────────────────
    # Strategy E: Reverse Blocking (Pool → S1)
    # ──────────────────────────────────────────────────────────────────────
    print("\n[Blocking] Strategy E: Reverse TF-IDF blocking (top-5) ...")

    for tgt_label, tgt in [("S2", s2), ("S3", s3)]:
        print(f"  E: Reverse {tgt_label} ...")
        tgt_rev = tgt.with_columns(
            (pl.col("norm_name").fill_null("") + " " + pl.col("norm_addr").fill_null(""))
            .alias("_tfidf_text")
        )
        _add_reverse_tfidf_matches(
            candidates, s1_tfidf, tgt_rev,
            text_col="_tfidf_text",
            top_k=5,
            analyzer="word",
            ngram_range=(1, 1),
            max_df=0.5,
            min_df=2,
            max_features=200_000,
            label=f"rev_{tgt_label}",
        )
        del tgt_rev
        gc.collect()

    del s1_tfidf
    gc.collect()

    pairs_after_E = sum(len(v) for v in candidates.values())
    print(f"  → Pairs after Strategy E: {pairs_after_E:,}")

    # ──────────────────────────────────────────────────────────────────────
    # Strategy F: ANN Embedding Search (Top-30)
    # ──────────────────────────────────────────────────────────────────────
    ann_k = max(30, top_k)
    print(f"\n[Blocking] Strategy F: ANN embedding search (top-{ann_k}) ...")

    for tgt_label, tgt, tgt_emb in [("S2", s2, s2_embeds), ("S3", s3, s3_embeds)]:
        print(f"  F: ANN {tgt_label} ...")
        _add_ann_matches(candidates, s1, tgt, s1_embeds, tgt_emb, top_k=ann_k)
        gc.collect()

    pairs_after_F = sum(len(v) for v in candidates.values())
    print(f"  → Pairs after Strategy F: {pairs_after_F:,}")

    # ──────────────────────────────────────────────────────────────────────
    # Ensure every S1 entity exists (even if empty singleton)
    # ──────────────────────────────────────────────────────────────────────
    for s1_id in s1["entity_id"].to_list():
        if s1_id not in candidates:
            candidates[s1_id] = set()

    total_pairs = sum(len(v) for v in candidates.values())
    avg_cands = total_pairs / max(len(candidates), 1)
    print(f"\n[Blocking] TOTAL unique pairs: {total_pairs:,} "
          f"(avg {avg_cands:.1f} per S1)")

    return dict(candidates)


def compute_blocking_recall(candidates: dict[str, set[str]],
                             ground_truth: pl.DataFrame) -> float:
    """
    Compute blocking recall on the ground-truth split.

    blocking_recall = |GT pairs captured in candidates| / |GT pairs total|

    A value < 1.0 is an unrecoverable ceiling on final F0.5.
    Prints a breakdown by country and provides per-country statistics.
    """
    total_gt = 0
    captured = 0
    missed_examples = []

    # Parse GT
    gt_dict: dict[str, set[str]] = {}
    for row in ground_truth.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        raw   = row.get("matched_entity_ids", "") or ""
        matches = {m.strip() for m in raw.split(",") if m.strip()}
        gt_dict[s1_id] = matches

    for s1_id, true_matches in tqdm(gt_dict.items(), desc="Computing blocking recall"):
        for m in true_matches:
            total_gt += 1
            if m in candidates.get(s1_id, set()):
                captured += 1
            elif len(missed_examples) < 20:
                missed_examples.append((s1_id, m))

    recall = captured / total_gt if total_gt > 0 else 1.0
    print(f"\n[Blocking Recall] {captured:,} / {total_gt:,} = {recall:.4f}")
    print(f"  Missed pairs: {total_gt - captured:,}")

    if missed_examples:
        print(f"  Sample missed pairs (first {len(missed_examples)}):")
        for s1_id, m_id in missed_examples[:5]:
            print(f"    {s1_id} → {m_id}")

    return recall
