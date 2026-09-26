# =============================================================
# features.py — Pairwise feature engineering for every candidate pair
# =============================================================
# Input  : two records (dict-like) with keys:
#              entity_id, business_name, business_address, country,
#              norm_name, norm_name_ns, norm_addr, norm_addr_ns,
#              embed_vec  (numpy array, optional)
# Output : a flat dict of ~40 numeric features ready for LightGBM
#
# DESIGN PRINCIPLES (from teammate's audit):
# 1. Every feature is language-agnostic (never uses raw country identity)
# 2. House-number features are CRITICAL for precision (distractors differ
#    by 1-2 digits in house number but are otherwise near-identical)
# 3. Competition features (rank/gap within S1 group) add significant lift
# 4. Name frequency features help distinguish common from rare matches
# 5. Token-level containment catches partial/truncated name matches
# 6. Empty-string similarity → NaN (LightGBM handles natively)

import math
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from normalize import (
    extract_numeric_tokens,
    extract_landmark_tokens,
    extract_first_number,
    extract_legal_form,
    LEGAL_SUFFIXES,
)


# ─── Jaccard helpers ──────────────────────────────────────────────────────────

def token_jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return float("nan")  # LightGBM handles NaN natively
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def token_containment(a: str, b: str) -> float:
    """
    Fraction of tokens in the shorter string that appear in the longer.
    Better than Jaccard for partial/truncated name matches.
    E.g. 'acme' vs 'acme robotics inc' → 1.0
    """
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return float("nan")
    if not sa or not sb:
        return 0.0
    shorter, longer = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    return len(shorter & longer) / len(shorter) if shorter else 0.0


def char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    def ngrams(s, n):
        return set(s[i:i+n] for i in range(len(s) - n + 1))
    sa = ngrams(a, n)
    sb = ngrams(b, n)
    if not sa and not sb:
        return float("nan")
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def safe_ratio(num: float, den: float) -> float:
    """Safe division, returns 0.0 on zero denominator."""
    return num / den if den > 0 else 0.0


def _safe_fuzz(func, a: str, b: str) -> float:
    """Safely compute rapidfuzz ratio, returning NaN for empty-vs-empty."""
    if not a and not b:
        return float("nan")
    return func(a, b) / 100.0


# ─── Core feature builders ────────────────────────────────────────────────────

def compute_name_features(norm_a: str, norm_a_ns: str,
                           norm_b: str, norm_b_ns: str,
                           raw_a: str, raw_b: str) -> dict:
    """
    Compute all name-related pairwise features.
    """
    feats = {}

    # Exact / near-exact
    feats["name_exact"]       = float(raw_a.strip().lower() == raw_b.strip().lower()) if raw_a and raw_b else float("nan")
    feats["name_norm_exact"]  = float(norm_a == norm_b) if norm_a and norm_b else float("nan")

    # Token-based
    feats["name_token_jaccard"]    = token_jaccard(norm_a, norm_b)
    feats["name_token_contain"]    = token_containment(norm_a, norm_b)
    feats["name_char3_jaccard"]    = char_ngram_jaccard(norm_a, norm_b, n=3)
    feats["name_char2_jaccard"]    = char_ngram_jaccard(norm_a, norm_b, n=2)

    # Edit-distance based
    feats["name_jaro_winkler"]     = _safe_fuzz(fuzz.WRatio, raw_a, raw_b)
    len_a = max(len(norm_a_ns), 1)
    len_b = max(len(norm_b_ns), 1)
    if norm_a_ns or norm_b_ns:
        lev = Levenshtein.distance(norm_a_ns, norm_b_ns)
        feats["name_edit_sim"]     = 1.0 - lev / max(len_a, len_b)
    else:
        feats["name_edit_sim"]     = float("nan")

    # RapidFuzz ratios on non-sorted normalised names
    feats["name_token_sort_ratio"] = _safe_fuzz(fuzz.token_sort_ratio, norm_a_ns, norm_b_ns)
    feats["name_partial_ratio"]    = _safe_fuzz(fuzz.partial_ratio, norm_a_ns, norm_b_ns)

    # Shape features
    toks_a = norm_a.split() if norm_a else []
    toks_b = norm_b.split() if norm_b else []
    len_n_a = len(toks_a)
    len_n_b = len(toks_b)
    feats["name_length_ratio"] = safe_ratio(min(len_n_a, len_n_b), max(len_n_a, len_n_b))

    # Token difference count (how many extra/missing tokens)
    feats["name_token_diff"] = abs(len_n_a - len_n_b)

    # ── NEW: Last token equal (strong signal for Indian "Pvt" → "Pvt Ltd" patterns)
    if toks_a and toks_b:
        feats["name_last_eq"] = float(toks_a[-1] == toks_b[-1])
    else:
        feats["name_last_eq"] = float("nan")

    return feats


def compute_address_features(norm_a: str, norm_a_ns: str,
                              norm_b: str, norm_b_ns: str,
                              raw_a: str, raw_b: str) -> dict:
    """
    Compute all address-related pairwise features.
    Includes critical house-number features for distractor detection.
    """
    feats = {}

    feats["addr_exact"]       = float(raw_a.strip().lower() == raw_b.strip().lower()) if raw_a and raw_b else float("nan")
    feats["addr_norm_exact"]  = float(norm_a == norm_b) if norm_a and norm_b else float("nan")

    feats["addr_token_jaccard"]    = token_jaccard(norm_a, norm_b)
    feats["addr_token_contain"]    = token_containment(norm_a, norm_b)
    feats["addr_char3_jaccard"]    = char_ngram_jaccard(norm_a, norm_b, n=3)

    if (norm_a_ns or norm_b_ns):
        len_a  = max(len(norm_a_ns), 1)
        len_b  = max(len(norm_b_ns), 1)
        lev    = Levenshtein.distance(norm_a_ns[:256], norm_b_ns[:256])  # cap for speed
        feats["addr_edit_sim"]     = 1.0 - lev / max(len_a, len_b)
    else:
        feats["addr_edit_sim"]     = float("nan")

    feats["addr_token_sort_ratio"] = _safe_fuzz(fuzz.token_sort_ratio, norm_a_ns, norm_b_ns)
    feats["addr_partial_ratio"]    = _safe_fuzz(fuzz.partial_ratio, norm_a_ns, norm_b_ns)

    tok_a = len(norm_a.split()) if norm_a else 0
    tok_b = len(norm_b.split()) if norm_b else 0
    feats["addr_length_ratio"] = safe_ratio(min(tok_a, tok_b), max(tok_a, tok_b))

    # ── Empty address flags (important for ~4.5% of candidates with empty addresses)
    feats["addr_empty_a"] = float(not raw_a or not raw_a.strip())
    feats["addr_empty_b"] = float(not raw_b or not raw_b.strip())

    # ── Numeric token features — CRITICAL precision signal ──────────────
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

    # ── NEW: First number (house number) features ────────────────────────
    # These are the #1 precision signal for rejecting near-copy distractors
    # (records that share name/address but differ in house number)
    first_a = extract_first_number(raw_a)
    first_b = extract_first_number(raw_b)

    if first_a and first_b:
        feats["first_num_match"] = float(first_a == first_b)
        try:
            diff = abs(int(first_a) - int(first_b))
            feats["first_num_diff"] = min(diff, 9999)  # cap extreme values
        except (ValueError, OverflowError):
            feats["first_num_diff"] = 9999
    elif not first_a and not first_b:
        feats["first_num_match"] = float("nan")
        feats["first_num_diff"] = float("nan")
    else:
        feats["first_num_match"] = 0.0
        feats["first_num_diff"] = 9999

    # How many numeric tokens each side has
    feats["num_count_diff"] = abs(len(num_a) - len(num_b))

    # ── Landmark token Jaccard
    lm_a = extract_landmark_tokens(raw_a)
    lm_b = extract_landmark_tokens(raw_b)
    if lm_a and lm_b:
        feats["addr_landmark_jaccard"] = len(lm_a & lm_b) / len(lm_a | lm_b)
    else:
        feats["addr_landmark_jaccard"] = 0.0

    return feats


def compute_cross_features(rec_a: dict, rec_b: dict) -> dict:
    """
    Cross-field / metadata features.
    """
    feats = {}

    feats["country_equal"] = float(
        str(rec_a.get("country", "")).strip().lower() ==
        str(rec_b.get("country", "")).strip().lower()
    )

    # ── Bi-encoder cosine similarity
    ea = rec_a.get("embed_vec")
    eb = rec_b.get("embed_vec")
    if ea is not None and eb is not None:
        ea_f = np.asarray(ea, dtype=np.float32)
        eb_f = np.asarray(eb, dtype=np.float32)
        dot   = float(np.dot(ea_f, eb_f))
        norm  = float(np.linalg.norm(ea_f) * np.linalg.norm(eb_f))
        feats["embed_cosine"] = dot / norm if norm > 1e-9 else 0.0
    else:
        feats["embed_cosine"] = 0.0

    # ── Legal form features
    legal_a = extract_legal_form(rec_a.get("business_name", ""))
    legal_b = extract_legal_form(rec_b.get("business_name", ""))
    feats["legal_eq"] = float(legal_a == legal_b) if (legal_a and legal_b) else float("nan")
    feats["legal_either"] = float(bool(legal_a) or bool(legal_b))

    # ── Name-address cross features
    # Does the name appear in the other side's address? (common in India)
    name_a = rec_a.get("norm_name", "")
    name_b = rec_b.get("norm_name", "")
    addr_a = rec_a.get("norm_addr", "")
    addr_b = rec_b.get("norm_addr", "")

    # Cross-field token overlap
    name_toks_a = set(name_a.split()) if name_a else set()
    addr_toks_a = set(addr_a.split()) if addr_a else set()
    name_toks_b = set(name_b.split()) if name_b else set()
    addr_toks_b = set(addr_b.split()) if addr_b else set()

    # Name-in-address overlap
    if name_toks_a and addr_toks_b:
        feats["name_in_addr_b"] = len(name_toks_a & addr_toks_b) / len(name_toks_a)
    else:
        feats["name_in_addr_b"] = 0.0

    return feats


def build_feature_vector(rec_a: dict, rec_b: dict) -> dict:
    """
    Master feature builder. Calls all sub-builders and merges results.

    rec_a, rec_b must each have:
        business_name, business_address, country,
        norm_name, norm_name_ns, norm_addr, norm_addr_ns,
        embed_vec   (numpy array or None)

    Returns a flat dict with ~40 float features.
    """
    feats = {}

    # Safely extract all fields with defaults
    norm_name_a = rec_a.get("norm_name", "") or ""
    norm_name_ns_a = rec_a.get("norm_name_ns", "") or ""
    norm_addr_a = rec_a.get("norm_addr", "") or ""
    norm_addr_ns_a = rec_a.get("norm_addr_ns", "") or ""
    raw_name_a = rec_a.get("business_name", "") or ""
    raw_addr_a = rec_a.get("business_address", "") or ""

    norm_name_b = rec_b.get("norm_name", "") or ""
    norm_name_ns_b = rec_b.get("norm_name_ns", "") or ""
    norm_addr_b = rec_b.get("norm_addr", "") or ""
    norm_addr_ns_b = rec_b.get("norm_addr_ns", "") or ""
    raw_name_b = rec_b.get("business_name", "") or ""
    raw_addr_b = rec_b.get("business_address", "") or ""

    feats.update(compute_name_features(
        norm_a=norm_name_a, norm_a_ns=norm_name_ns_a,
        norm_b=norm_name_b, norm_b_ns=norm_name_ns_b,
        raw_a=raw_name_a, raw_b=raw_name_b,
    ))

    feats.update(compute_address_features(
        norm_a=norm_addr_a, norm_a_ns=norm_addr_ns_a,
        norm_b=norm_addr_b, norm_b_ns=norm_addr_ns_b,
        raw_a=raw_addr_a, raw_b=raw_addr_b,
    ))

    feats.update(compute_cross_features(rec_a, rec_b))

    # ── Interaction terms — help GBDT converge faster
    name_j = feats.get("name_char3_jaccard", 0.0)
    addr_j = feats.get("addr_char3_jaccard", 0.0)
    if math.isnan(name_j) if isinstance(name_j, float) else False:
        name_j = 0.0
    if math.isnan(addr_j) if isinstance(addr_j, float) else False:
        addr_j = 0.0

    feats["name_addr_product"] = name_j * addr_j

    # Name strong + address weak mismatch (distractor signal)
    name_tj = feats.get("name_token_jaccard", 0.0)
    if isinstance(name_tj, float) and math.isnan(name_tj):
        name_tj = 0.0
    addr_tj = feats.get("addr_token_jaccard", 0.0)
    if isinstance(addr_tj, float) and math.isnan(addr_tj):
        addr_tj = 0.0
    feats["name_strong_addr_weak"] = float(name_tj > 0.8 and addr_tj < 0.3)

    return feats


# ─── Ordered feature names (for LightGBM DataFrame column order) ─────────────
FEATURE_NAMES = [
    # Name features
    "name_exact", "name_norm_exact",
    "name_token_jaccard", "name_token_contain",
    "name_char3_jaccard", "name_char2_jaccard",
    "name_jaro_winkler", "name_edit_sim",
    "name_token_sort_ratio", "name_partial_ratio",
    "name_length_ratio", "name_token_diff", "name_last_eq",
    # Address features
    "addr_exact", "addr_norm_exact",
    "addr_token_jaccard", "addr_token_contain", "addr_char3_jaccard",
    "addr_edit_sim", "addr_token_sort_ratio", "addr_partial_ratio",
    "addr_length_ratio",
    "addr_empty_a", "addr_empty_b",
    "addr_numeric_exact", "addr_numeric_jaccard",
    "first_num_match", "first_num_diff", "num_count_diff",
    "addr_landmark_jaccard",
    # Cross features
    "country_equal", "embed_cosine",
    "legal_eq", "legal_either", "name_in_addr_b",
    # Interaction features
    "name_addr_product", "name_strong_addr_weak",
]
