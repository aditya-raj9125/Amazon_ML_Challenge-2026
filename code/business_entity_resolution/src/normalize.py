# =============================================================
# normalize.py — Text normalization utilities
# =============================================================
# All normalization is built ONLY from the training vocabulary —
# no external databases, geocoding APIs, or internet lookups.
#
# Design goals
# ─────────────
# 1. Language-agnostic: works for English, Hindi-transliterated, and
#    unseen French text without country-specific branches.
# 2. Cheap: pure string operations — no regex that scales with corpus size.
# 3. Deterministic: same input always produces same output (no randomness).

import re
import unicodedata

# ─── Legal suffix vocabulary ──────────────────────────────────────────────────
# Built from the training corpus (most common legal tokens across US + India).
# Expand freely — this list is never looked up externally.
LEGAL_SUFFIXES = {
    # English / US
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "llp", "lp", "plc", "co", "company", "companies", "group", "grp",
    "associates", "assoc", "enterprises", "ent", "solutions", "svcs",
    "services", "holdings", "intl", "international", "usa", "us",
    # India
    "pvt", "private", "pvtltd", "ltdpvt",
    # French (for test-time robustness)
    "sarl", "sas", "sa", "sasu", "eurl", "sci", "snc",
    # Generic
    "the",
}

# ─── Address abbreviation map ─────────────────────────────────────────────────
# Normalise the most frequent road/direction tokens.
ADDR_ABBREV = {
    "st":   "street",
    "str":  "street",
    "rd":   "road",
    "ave":  "avenue",
    "av":   "avenue",
    "blvd": "boulevard",
    "dr":   "drive",
    "ln":   "lane",
    "ct":   "court",
    "pl":   "place",
    "sq":   "square",
    "hwy":  "highway",
    "fwy":  "freeway",
    "pkwy": "parkway",
    "nr":   "near",
    "n":    "north",
    "s":    "south",
    "e":    "east",
    "w":    "west",
}

# ─── Landmark prefixes to strip ───────────────────────────────────────────────
# These tokens precede a landmark reference ("Near SBI ATM").
# Stripped before computing street-level similarity.
LANDMARK_PREFIXES = {"near", "nr", "opp", "opposite", "behind", "adj", "adjacent",
                     "next", "beside", "above", "below", "front"}

_PUNCT_RE = re.compile(r"[^\w\s]")          # keeps letters, digits, spaces
_MULTI_SPACE_RE = re.compile(r"\s+")
_AMP_RE = re.compile(r"\s*&\s*")


# ─── Core normalisation helpers ───────────────────────────────────────────────

def unicode_nfkc(text: str) -> str:
    """
    NFKC normalisation: converts compatibility characters to canonical form.
    Critical for Hindi-transliterated text and French diacritics —
    e.g. 'é' and 'e\u0301' both become 'é' (consistently).
    """
    return unicodedata.normalize("NFKC", text)


def lowercase_strip(text: str) -> str:
    """Lower-case and strip leading/trailing whitespace."""
    return text.strip().lower()


def replace_ampersand(text: str) -> str:
    """'A & B' → 'a and b'. Done before tokenisation so 'and' is a real token."""
    return _AMP_RE.sub(" and ", text)


def remove_punct(text: str) -> str:
    """Remove all punctuation, keeping letters, digits, spaces."""
    return _PUNCT_RE.sub(" ", text)


def collapse_spaces(text: str) -> str:
    """Collapse multiple whitespace into single space."""
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def remove_legal_suffixes(tokens: list[str]) -> list[str]:
    """
    Strip legal-suffix tokens from a token list.
    E.g. ['acme', 'robotics', 'inc', '.'] → ['acme', 'robotics']
    """
    return [t for t in tokens if t not in LEGAL_SUFFIXES]


def expand_addr_abbrev(tokens: list[str]) -> list[str]:
    """
    Replace abbreviated road tokens with their canonical form.
    E.g. ['500', 'market', 'st'] → ['500', 'market', 'street']
    """
    return [ADDR_ABBREV.get(t, t) for t in tokens]


# ─── Compound normalisers ─────────────────────────────────────────────────────

def normalize_name(raw: str) -> str:
    """
    Full business-name normalisation pipeline.

    Steps:
      1. NFKC Unicode normalisation
      2. Lower-case
      3. Replace '&' with 'and'
      4. Remove punctuation
      5. Tokenise by whitespace
      6. Strip legal suffixes
      7. Sort tokens (makes 'Acme Robotics' == 'Robotics Acme')
      8. Rejoin

    Returns the normalised string.
    """
    if not raw or not isinstance(raw, str):
        return ""
    text = unicode_nfkc(raw)
    text = lowercase_strip(text)
    text = replace_ampersand(text)
    text = remove_punct(text)
    text = collapse_spaces(text)
    tokens = text.split()
    tokens = remove_legal_suffixes(tokens)
    tokens = sorted(tokens)          # token-sort for word-order robustness
    return " ".join(tokens)


def normalize_name_no_sort(raw: str) -> str:
    """
    Same as normalize_name but WITHOUT token sorting.
    Used to preserve original word order for features that
    are sensitive to it (e.g. token-sort similarity delta).
    """
    if not raw or not isinstance(raw, str):
        return ""
    text = unicode_nfkc(raw)
    text = lowercase_strip(text)
    text = replace_ampersand(text)
    text = remove_punct(text)
    text = collapse_spaces(text)
    tokens = text.split()
    tokens = remove_legal_suffixes(tokens)
    return " ".join(tokens)


def normalize_address(raw: str) -> str:
    """
    Full address normalisation pipeline.

    Steps:
      1. NFKC Unicode
      2. Lower-case
      3. Remove punctuation
      4. Tokenise
      5. Expand road abbreviations
      6. Sort tokens (component reordering is common in Indian addresses)
      7. Rejoin

    Landmark prefixes ('Near', 'Opp.') are NOT stripped here — they are
    handled separately in feature_engineering.py so that landmark-token
    Jaccard can be computed as its own feature.
    """
    if not raw or not isinstance(raw, str):
        return ""
    text = unicode_nfkc(raw)
    text = lowercase_strip(text)
    text = remove_punct(text)
    text = collapse_spaces(text)
    tokens = text.split()
    tokens = expand_addr_abbrev(tokens)
    tokens = sorted(tokens)
    return " ".join(tokens)


def normalize_address_no_sort(raw: str) -> str:
    """Address normalisation preserving original token order."""
    if not raw or not isinstance(raw, str):
        return ""
    text = unicode_nfkc(raw)
    text = lowercase_strip(text)
    text = remove_punct(text)
    text = collapse_spaces(text)
    tokens = text.split()
    tokens = expand_addr_abbrev(tokens)
    return " ".join(tokens)


# ─── Utility extractors ───────────────────────────────────────────────────────

def extract_numeric_tokens(text: str) -> set[str]:
    """
    Return the set of purely-numeric tokens in the text.
    E.g. '500 Market Street 94105' → {'500', '94105'}

    Numeric tokens (house numbers, PINs, postal codes) are strong
    precision signals — two addresses sharing a house number are very
    likely the same location even if the rest of the text diverges.
    """
    if not text:
        return set()
    return {t for t in text.split() if t.isdigit()}


def extract_landmark_tokens(raw_address: str) -> set[str]:
    """
    Extract tokens that appear AFTER a landmark prefix.
    E.g. 'Near SBI ATM, Connaught Place' → {'sbi', 'atm'}

    Used as a separate Jaccard feature rather than being mixed into
    the main address similarity score.
    """
    if not raw_address or not isinstance(raw_address, str):
        return set()
    text = unicode_nfkc(raw_address)
    text = lowercase_strip(text)
    text = remove_punct(text)
    tokens = text.split()
    landmark_toks = set()
    capture = False
    for tok in tokens:
        if tok in LANDMARK_PREFIXES:
            capture = True
            continue
        if capture:
            # Capture until we hit another landmark prefix or end
            if tok in LANDMARK_PREFIXES:
                capture = True
            else:
                landmark_toks.add(tok)
    return landmark_toks


def build_text_for_embedding(name: str, address: str) -> str:
    """
    Concatenate name and address into the string fed to the sentence encoder.
    We use the RAW (non-normalised) text so the multilingual model can use
    its full vocabulary — normalisation could destroy script-specific
    information the encoder knows how to handle.
    """
    name    = (name    or "").strip()
    address = (address or "").strip()
    return f"{name} {address}".strip()
