# =============================================================
# features.py — Pairwise feature engineering for every candidate pair
# =============================================================
# Input  : two records (dict-like) with keys:
#              entity_id, business_name, business_address, country,
#              norm_name, norm_name_ns, norm_addr, norm_addr_ns,
#              embed_vec  (numpy array, optional — set to None if not yet computed)
# Output : a flat dict of ~25 numeric features ready for LightGBM
#
# Every feature is language-agnostic (derived from similarity scores,
# never from raw country identity). This is intentional — it makes
# the model generalise to unseen France records without retraining.

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from normalize import (
    extract_numeric_tokens,
    extract_landmark_tokens,
)


# ─── Jaccard helpers ──────────────────────────────────────────────────────────

def token_jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    def ngrams(s, n):
        return set(s[i:i+n] for i in range(len(s) - n + 1))
    sa = ngrams(a, n)
    sb = ngrams(b, n)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def safe_ratio(num: float, den: float) -> float:
    """Safe division, returns 0.0 on zero denominator."""
    return num / den if den > 0 else 0.0


# ─── Core feature builders ────────────────────────────────────────────────────

def compute_name_features(norm_a: str, norm_a_ns: str,
                           norm_b: str, norm_b_ns: str,
                           raw_a: str, raw_b: str) -> dict:
    """
    Compute all name-related pairwise features.

    norm_*    : token-sorted, legal-suffix-stripped normalised name
    norm_*_ns : same without sorting (preserves word order for edit dist)
    raw_*     : original business_name field

    Features computed
    -----------------
    name_exact              : 1 if raw strings match exactly
    name_norm_exact         : 1 if normalised (sorted) strings match
    name_token_jaccard      : Jaccard on token sets of norm names
    name_char3_jaccard      : Jaccard on char-3grams of norm names
    name_jaro_winkler       : Jaro-Winkler on raw names (typo-robust)
    name_edit_sim           : 1 - normalised Levenshtein on norm_ns names
    name_token_sort_ratio   : fuzz.token_sort_ratio (handles transpositions)
    name_partial_ratio      : fuzz.partial_ratio (substring matching)
    name_length_ratio       : min/max of lengths (shape similarity)
    """
    feats = {}

    # Exact / near-exact
    feats["name_exact"]       = float(raw_a.strip().lower() == raw_b.strip().lower())
    feats["name_norm_exact"]  = float(norm_a == norm_b)

    # Token-based
    feats["name_token_jaccard"]    = token_jaccard(norm_a, norm_b)
    feats["name_char3_jaccard"]    = char_ngram_jaccard(norm_a, norm_b, n=3)
    feats["name_char2_jaccard"]    = char_ngram_jaccard(norm_a, norm_b, n=2)

    # Edit-distance based
    feats["name_jaro_winkler"]     = fuzz.WRatio(raw_a, raw_b) / 100.0
    len_a = max(len(norm_a_ns), 1)
    len_b = max(len(norm_b_ns), 1)
    lev   = Levenshtein.distance(norm_a_ns, norm_b_ns)
    feats["name_edit_sim"]         = 1.0 - lev / max(len_a, len_b)

    # RapidFuzz ratios on non-sorted normalised names
    feats["name_token_sort_ratio"] = fuzz.token_sort_ratio(norm_a_ns, norm_b_ns) / 100.0
    feats["name_partial_ratio"]    = fuzz.partial_ratio(norm_a_ns, norm_b_ns) / 100.0

    # Shape
    len_n_a = len(norm_a.split())
    len_n_b = len(norm_b.split())
    feats["name_length_ratio"] = safe_ratio(min(len_n_a, len_n_b),
                                             max(len_n_a, len_n_b))
    return feats


def compute_address_features(norm_a: str, norm_a_ns: str,
                              norm_b: str, norm_b_ns: str,
                              raw_a: str, raw_b: str) -> dict:
    """
    Compute all address-related pairwise features.

    Features computed
    -----------------
    addr_exact              : exact match on raw addresses
    addr_norm_exact         : exact match on normalised addresses
    addr_token_jaccard      : Jaccard on token sets
    addr_char3_jaccard      : Jaccard on char-3grams
    addr_edit_sim           : 1 - normalised Levenshtein on norm_ns
    addr_token_sort_ratio   : fuzz.token_sort_ratio
    addr_partial_ratio      : fuzz.partial_ratio
    addr_length_ratio       : min/max token counts
    addr_numeric_exact      : 1 if any numeric tokens match exactly
    addr_numeric_jaccard    : Jaccard on numeric tokens (house # / PIN)
    addr_landmark_jaccard   : Jaccard on landmark tokens
    """
    feats = {}

    feats["addr_exact"]       = float(raw_a.strip().lower() == raw_b.strip().lower())
    feats["addr_norm_exact"]  = float(norm_a == norm_b)

    feats["addr_token_jaccard"]    = token_jaccard(norm_a, norm_b)
    feats["addr_char3_jaccard"]    = char_ngram_jaccard(norm_a, norm_b, n=3)

    len_a  = max(len(norm_a_ns), 1)
    len_b  = max(len(norm_b_ns), 1)
    lev    = Levenshtein.distance(norm_a_ns[:256], norm_b_ns[:256])  # cap for speed
    feats["addr_edit_sim"]         = 1.0 - lev / max(len_a, len_b)

    feats["addr_token_sort_ratio"] = fuzz.token_sort_ratio(norm_a_ns, norm_b_ns) / 100.0
    feats["addr_partial_ratio"]    = fuzz.partial_ratio(norm_a_ns, norm_b_ns) / 100.0

    tok_a = len(norm_a.split())
    tok_b = len(norm_b.split())
    feats["addr_length_ratio"] = safe_ratio(min(tok_a, tok_b), max(tok_a, tok_b))

    # Numeric token features — strong precision signal
    num_a = extract_numeric_tokens(raw_a)
    num_b = extract_numeric_tokens(raw_b)
    if num_a and num_b:
        inter = num_a & num_b
        union = num_a | num_b
        feats["addr_numeric_exact"]   = float(bool(inter))
        feats["addr_numeric_jaccard"] = len(inter) / len(union)
    else:
        feats["addr_numeric_exact"]   = 0.0
        feats["addr_numeric_jaccard"] = 0.0

    # Landmark token Jaccard
    lm_a = extract_landmark_tokens(raw_a)
    lm_b = extract_landmark_tokens(raw_b)
    if lm_a and lm_b:
        feats["addr_landmark_jaccard"] = len(lm_a & lm_b) / len(lm_a | lm_b)
    else:
        feats["addr_landmark_jaccard"] = 0.0

    return feats


def compute_cross_features(rec_a: dict, rec_b: dict) -> dict:
    """
    Cross-field / metadata features that don't fit neatly into
    name-only or address-only categories.

    Features computed
    -----------------
    country_equal       : 1 if countries match (weak — should not be a hard filter)
    embed_cosine        : cosine similarity of bi-encoder embeddings
    name_addr_product   : name_char3_jaccard * addr_char3_jaccard (interaction)
    """
    feats = {}

    feats["country_equal"] = float(
        str(rec_a.get("country", "")).strip().lower() ==
        str(rec_b.get("country", "")).strip().lower()
    )

    # Bi-encoder cosine — computed once during blocking, reused here
    ea = rec_a.get("embed_vec")
    eb = rec_b.get("embed_vec")
    if ea is not None and eb is not None:
        dot   = float(np.dot(ea, eb))
        norm  = float(np.linalg.norm(ea) * np.linalg.norm(eb))
        feats["embed_cosine"] = dot / norm if norm > 1e-9 else 0.0
    else:
        feats["embed_cosine"] = 0.0

    return feats


def build_feature_vector(rec_a: dict, rec_b: dict) -> dict:
    """
    Master feature builder. Calls all sub-builders and merges results.

    rec_a, rec_b must each have:
        business_name, business_address, country,
        norm_name, norm_name_ns, norm_addr, norm_addr_ns,
        embed_vec   (numpy array or None)

    Returns a flat dict with ~25 float features.
    """
    feats = {}

    feats.update(compute_name_features(
        norm_a    = rec_a["norm_name"],
        norm_a_ns = rec_a["norm_name_ns"],
        norm_b    = rec_b["norm_name"],
        norm_b_ns = rec_b["norm_name_ns"],
        raw_a     = rec_a.get("business_name", ""),
        raw_b     = rec_b.get("business_name", ""),
    ))

    feats.update(compute_address_features(
        norm_a    = rec_a["norm_addr"],
        norm_a_ns = rec_a["norm_addr_ns"],
        norm_b    = rec_b["norm_addr"],
        norm_b_ns = rec_b["norm_addr_ns"],
        raw_a     = rec_a.get("business_address", ""),
        raw_b     = rec_b.get("business_address", ""),
    ))

    feats.update(compute_cross_features(rec_a, rec_b))

    # Interaction term — GBDT can discover this, but helping it with an explicit
    # product feature speeds up convergence on small label budgets.
    feats["name_addr_product"] = (
        feats.get("name_char3_jaccard", 0.0) * feats.get("addr_char3_jaccard", 0.0)
    )

    return feats


# ─── Ordered feature names (for LightGBM DataFrame column order) ─────────────
FEATURE_NAMES = [
    # Name features
    "name_exact", "name_norm_exact",
    "name_token_jaccard", "name_char3_jaccard", "name_char2_jaccard",
    "name_jaro_winkler", "name_edit_sim",
    "name_token_sort_ratio", "name_partial_ratio",
    "name_length_ratio",
    # Address features
    "addr_exact", "addr_norm_exact",
    "addr_token_jaccard", "addr_char3_jaccard",
    "addr_edit_sim", "addr_token_sort_ratio", "addr_partial_ratio",
    "addr_length_ratio",
    "addr_numeric_exact", "addr_numeric_jaccard",
    "addr_landmark_jaccard",
    # Cross features
    "country_equal", "embed_cosine",
    # Interaction
    "name_addr_product",
]
