# =============================================================
# normalize.py — Text normalization utilities
# =============================================================
# All normalization is built ONLY from the training vocabulary —
# no external databases, geocoding APIs, or internet lookups.
#
# Design goals
# ─────────────
# 1. Language-agnostic: works for English, Hindi-transliterated, and
#    French text with country-specific branches ONLY where needed.
# 2. Cheap: pure string operations — no regex that scales with corpus size.
# 3. Deterministic: same input always produces same output (no randomness).
# 4. French-ready: handles French legal forms, address abbreviations,
#    and articles without degrading US/India performance.

import re
import unicodedata

# ─── Legal suffix vocabulary ──────────────────────────────────────────────────
# Built from the training corpus (most common legal tokens across US + India).
# French forms added for test-time robustness on unseen France data.
LEGAL_SUFFIXES = {
    # English / US
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "llp", "lp", "plc", "co", "company", "companies", "group", "grp",
    "associates", "assoc", "enterprises", "ent", "solutions", "svcs",
    "services", "holdings", "intl", "international", "usa", "us",
    # India
    "pvt", "private", "pvtltd", "ltdpvt",
    # French legal forms (for test-time robustness)
    "sarl", "sas", "sa", "sasu", "eurl", "sci", "snc", "ei",
    "scea", "gaec", "earl", "gie", "sem",
    # Generic
    "the",
}

# ─── Honorific / title prefixes to strip ─────────────────────────────────────
HONORIFIC_PREFIXES = {
    # India
    "shri", "smt", "m/s", "ms", "dr", "mr", "mrs",
    # French
    "m", "mme", "mlle",
}

# ─── French articles (dropped only from core name, not from address) ─────────
FRENCH_ARTICLES = {"du", "de", "la", "le", "les", "des", "d", "l", "au", "aux"}

# ─── Address abbreviation map ─────────────────────────────────────────────────
# Normalise the most frequent road/direction tokens across US, India, France.
ADDR_ABBREV = {
    # US/India
    "st":    "street",
    "str":   "street",
    "rd":    "road",
    "ave":   "avenue",
    "av":    "avenue",
    "blvd":  "boulevard",
    "dr":    "drive",
    "ln":    "lane",
    "ct":    "court",
    "pl":    "place",
    "sq":    "square",
    "hwy":   "highway",
    "fwy":   "freeway",
    "pkwy":  "parkway",
    "nr":    "near",
    "n":     "north",
    "s":     "south",
    "e":     "east",
    "w":     "west",
    # French address abbreviations
    "r":     "rue",
    "bd":    "boulevard",
    "all":   "allee",
    "ch":    "chemin",
    "imp":   "impasse",
    "qu":    "quai",
    "rte":   "route",
    "fbg":   "faubourg",
    "ste":   "sainte",
    # French saint abbreviation (only in address context)
    # "st" is already mapped to "street" — context-dependent handling below
}

# ─── French region/department → state-level normalization ─────────────────────
FRENCH_STATE_LOOKUP = {
    "hauts de france": "hauts de france",
    "nord": "hauts de france",
    "pas de calais": "hauts de france",
    "ile de france": "ile de france",
    "nouvelle aquitaine": "nouvelle aquitaine",
    "gironde": "nouvelle aquitaine",
    "pays de la loire": "pays de la loire",
    "loire atlantique": "pays de la loire",
    "occitanie": "occitanie",
    "auvergne rhone alpes": "auvergne rhone alpes",
}

# ─── US/India state abbreviation lookup ───────────────────────────────────────
STATE_LOOKUP = {
    # US states (common abbreviations)
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
    # India states (common abbreviations)
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam",
    "br": "bihar", "cg": "chhattisgarh", "ga": "goa", "gj": "gujarat",
    "hr": "haryana", "hp": "himachal pradesh", "jh": "jharkhand",
    "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh",
    "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya", "mz": "mizoram",
    "nl": "nagaland", "od": "odisha", "pb": "punjab", "rj": "rajasthan",
    "sk": "sikkim", "tn": "tamil nadu", "tg": "telangana", "tr": "tripura",
    "up": "uttar pradesh", "uk": "uttarakhand", "wb": "west bengal",
    "dl": "delhi", "jk": "jammu and kashmir",
}

# ─── Landmark prefixes to strip ───────────────────────────────────────────────
LANDMARK_PREFIXES = {"near", "nr", "opp", "opposite", "behind", "adj", "adjacent",
                     "next", "beside", "above", "below", "front"}

_PUNCT_RE = re.compile(r"[^\w\s]")          # keeps letters, digits, spaces
_MULTI_SPACE_RE = re.compile(r"\s+")
_AMP_RE = re.compile(r"\s*&\s*")
# Leetspeak pattern: common digit→letter substitutions in business names
_LEET_MAP = str.maketrans("01345", "olsas")
# French N° and # → numero
_NUMERO_RE = re.compile(r"n[°o]?\s*", re.IGNORECASE)


# ─── Core normalisation helpers ───────────────────────────────────────────────

def unicode_nfkc(text: str) -> str:
    """
    NFKC normalisation: converts compatibility characters to canonical form.
    Critical for Hindi-transliterated text and French diacritics.
    """
    return unicodedata.normalize("NFKC", text)


def strip_accents(text: str) -> str:
    """
    Remove diacritical marks (accents) while preserving base characters.
    'é' → 'e', 'ü' → 'u', etc.
    Important for matching French text with transliterated versions.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


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


def fix_leetspeak(text: str) -> str:
    """
    Fix common leetspeak substitutions in business names.
    E.g. 'C0ok' → 'Cook', '5ervices' → 'Services'
    Only applied when surrounded by alpha characters.
    """
    # Simple case: translate isolated digits that look like letters
    result = []
    tokens = text.split()
    for tok in tokens:
        if any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
            result.append(tok.translate(_LEET_MAP))
        else:
            result.append(tok)
    return " ".join(result)


def remove_legal_suffixes(tokens: list[str]) -> list[str]:
    """
    Strip legal-suffix tokens from a token list.
    E.g. ['acme', 'robotics', 'inc', '.'] → ['acme', 'robotics']
    """
    return [t for t in tokens if t not in LEGAL_SUFFIXES]


def remove_honorifics(tokens: list[str]) -> list[str]:
    """
    Strip honorific/title prefix tokens.
    E.g. ['shri', 'rajesh', 'kumar'] → ['rajesh', 'kumar']
    """
    # Only strip from the beginning
    while tokens and tokens[0] in HONORIFIC_PREFIXES:
        tokens = tokens[1:]
    return tokens


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
      2. Strip accents (é→e, ü→u)
      3. Lower-case
      4. Replace '&' with 'and'
      5. Fix leetspeak (0→o, 5→s in mixed alpha-digit tokens)
      6. Remove punctuation
      7. Tokenise by whitespace
      8. Strip honorific prefixes (Shri, Smt, M/s, Dr)
      9. Strip legal suffixes (Inc, Corp, LLC, SARL, etc.)
     10. Sort tokens (makes 'Acme Robotics' == 'Robotics Acme')
     11. Rejoin

    Returns the normalised string.
    """
    if not raw or not isinstance(raw, str):
        return ""
    text = unicode_nfkc(raw)
    text = strip_accents(text)
    text = lowercase_strip(text)
    text = replace_ampersand(text)
    text = fix_leetspeak(text)
    text = remove_punct(text)
    text = collapse_spaces(text)
    tokens = text.split()
    tokens = remove_honorifics(tokens)
    tokens = remove_legal_suffixes(tokens)
    tokens = [t for t in tokens if len(t) > 0]
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
    text = strip_accents(text)
    text = lowercase_strip(text)
    text = replace_ampersand(text)
    text = fix_leetspeak(text)
    text = remove_punct(text)
    text = collapse_spaces(text)
    tokens = text.split()
    tokens = remove_honorifics(tokens)
    tokens = remove_legal_suffixes(tokens)
    tokens = [t for t in tokens if len(t) > 0]
    return " ".join(tokens)


def normalize_address(raw: str) -> str:
    """
    Full address normalisation pipeline.

    Steps:
      1. NFKC Unicode
      2. Strip accents
      3. Lower-case
      4. Remove punctuation
      5. Tokenise
      6. Expand road abbreviations (including French: R.→rue, Bd→boulevard)
      7. Sort tokens (component reordering is common in Indian addresses)
      8. Rejoin

    Landmark prefixes ('Near', 'Opp.') are NOT stripped here — they are
    handled separately in feature_engineering.py so that landmark-token
    Jaccard can be computed as its own feature.
    """
    if not raw or not isinstance(raw, str):
        return ""
    text = unicode_nfkc(raw)
    text = strip_accents(text)
    text = lowercase_strip(text)
    # Handle French N° → numero
    text = _NUMERO_RE.sub("numero ", text)
    text = remove_punct(text)
    text = collapse_spaces(text)
    tokens = text.split()
    tokens = expand_addr_abbrev(tokens)
    tokens = [t for t in tokens if len(t) > 0]
    tokens = sorted(tokens)
    return " ".join(tokens)


def normalize_address_no_sort(raw: str) -> str:
    """Address normalisation preserving original token order."""
    if not raw or not isinstance(raw, str):
        return ""
    text = unicode_nfkc(raw)
    text = strip_accents(text)
    text = lowercase_strip(text)
    text = _NUMERO_RE.sub("numero ", text)
    text = remove_punct(text)
    text = collapse_spaces(text)
    tokens = text.split()
    tokens = expand_addr_abbrev(tokens)
    tokens = [t for t in tokens if len(t) > 0]
    return " ".join(tokens)


# ─── Utility extractors ───────────────────────────────────────────────────────

def extract_first_number(text: str) -> str:
    """
    Return the first purely-numeric token in the text, or '' if none.
    E.g. '500 Market Street 94105' → '500'
    This is typically the house number — the strongest precision signal.
    """
    if not text:
        return ""
    for t in text.split():
        if t.isdigit():
            return t
    return ""


def extract_numeric_tokens(text: str) -> set[str]:
    """
    Return the set of purely-numeric tokens in the text.
    E.g. '500 Market Street 94105' → {'500', '94105'}
    """
    if not text:
        return set()
    return {t for t in text.split() if t.isdigit()}


def extract_landmark_tokens(raw_address: str) -> set[str]:
    """
    Extract tokens that appear AFTER a landmark prefix.
    E.g. 'Near SBI ATM, Connaught Place' → {'sbi', 'atm'}
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
            if tok in LANDMARK_PREFIXES:
                capture = True
            else:
                landmark_toks.add(tok)
    return landmark_toks


def extract_legal_form(raw_name: str) -> str:
    """
    Extract the legal form suffix from a business name.
    E.g. 'Acme Corp Inc' → 'inc'
    E.g. 'Pharmacie SARL' → 'sarl'
    Returns empty string if no legal form found.
    """
    if not raw_name or not isinstance(raw_name, str):
        return ""
    tokens = raw_name.strip().lower().split()
    for tok in reversed(tokens):
        cleaned = remove_punct(tok).strip()
        if cleaned in LEGAL_SUFFIXES:
            return cleaned
    return ""


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
