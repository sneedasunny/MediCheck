"""
MediCheck matching logic.

Identifies candidate medicine records from noisy text (OCR output or a
typed query) using:
  1. ingredient-set matching (with synonyms and fuzzy matching for OCR errors)
  2. brand / product-name matching
  3. strength numbers (e.g. 625)
  4. dosage form (tablet vs suspension, etc.)
  5. a small contribution from TF-IDF similarity

It returns candidates with a confidence label and the reasons for the match.
It never claims a medicine is confirmed: it only ranks possible records.
"""

import re
from difflib import SequenceMatcher

# --------------------------------------------------------------------------
# Synonyms. Applied to BOTH the scanned text and the dataset ingredients,
# so each side ends up using one canonical term. Longer phrases first.
# Only add entries for terms that mean the same active ingredient.
# --------------------------------------------------------------------------
SYNONYMS = [
    ("clavulanate potassium", "clavulanic acid"),
    ("potassium clavulanate", "clavulanic acid"),
    ("clavulanate", "clavulanic acid"),
    ("amoxycillin", "amoxicillin"),
]

# Words that describe the product form/pack rather than the brand.
GENERIC_WORDS = {
    "tablet", "tablets", "tab", "tabs", "capsule", "capsules", "cap", "caps",
    "syrup", "suspension", "oral", "dry", "drops", "drop", "injection",
    "cream", "gel", "ointment", "inhaler", "powder", "solution", "forte",
    "duo", "chewable", "mg", "gm", "ml", "mcg", "iu", "ip", "bp", "usp",
    "gastro", "resistant", "and", "with", "for", "the", "each", "film",
    "coated", "sr", "xr", "dds", "bid", "ds", "es", "plus",
}

# Dosage form keywords found in scanned text -> dataset dosage_form words.
FORM_KEYWORDS = {
    "tablet": ["tablet", "tablets", "tab", "tabs"],
    "capsule": ["capsule", "capsules"],
    "suspension": ["suspension"],
    "syrup": ["syrup"],
    "injection": ["injection"],
    "drops": ["drops"],
}

FUZZY_CUTOFF = 0.85
MIN_FUZZY_LEN = 6   # shorter words must match exactly


def _clean(value):
    if value is None:
        return ""
    try:
        if value != value:  # NaN
            return ""
    except Exception:
        pass
    return str(value).strip()


def normalize(text):
    """Lowercase, remove punctuation, apply synonyms, collapse spaces."""
    text = _clean(text).casefold()
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    text = re.sub(r"(?<![0-9])\.|\.(?![0-9])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for variant, canonical in SYNONYMS:
        text = re.sub(rf"\b{re.escape(variant)}\b", canonical, text)
    return text


def parse_ingredients(active_ingredients):
    """
    Extract ingredient names (without strengths) from the dataset field.
    Handles both "Amoxycillin (500mg)   Clavulanic Acid (125mg)" and
    "Paracetamol 500 mg" / "A 10 mg/5 mL + B 4 mg/5 mL" formats.
    """
    text = _clean(active_ingredients)
    if not text:
        return []

    text = re.sub(r"\([^)]*\)", " | ", text)             # drop "(500mg)"
    text = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|gm|ml|iu|%)(/\s*\d*\s*\w+)?",
                  " | ", text, flags=re.I)                # drop "500 mg/5 mL"
    parts = re.split(r"\s{2,}|\+|,|\||;|/", text)

    names = []
    for part in parts:
        name = normalize(part)
        name = re.sub(r"\b\d+(\.\d+)?\b", "", name).strip()
        name = re.sub(r"\s+", " ", name)
        if len(name) >= 4 and name not in {"na", "not available"}:
            if name not in names:
                names.append(name)
    return names


# Letters OCR commonly produces in place of digits (E25 -> 625, 1O0 -> 100)
DIGIT_CONFUSIONS = {"e": "6", "g": "6", "b": "6", "o": "0",
                    "l": "1", "i": "1", "s": "5", "z": "2"}


def _corrected_numbers(tokens):
    """Numbers recovered from tokens like 'e25' (at most one confused letter)."""
    found = set()
    for t in tokens:
        if 3 <= len(t) <= 4 and sum(c.isdigit() for c in t) >= len(t) - 1:
            fixed = "".join(DIGIT_CONFUSIONS.get(c, c) for c in t)
            if fixed.isdigit() and fixed != t:
                found.add(fixed)
    return found


def _numbers(text):
    return set(re.findall(r"\b\d{2,4}(?:\.\d+)?\b", text))


class Matcher:
    def __init__(self, medicines, tfidf_scores_fn=None):
        """
        medicines: pandas DataFrame (the medicine index).
        tfidf_scores_fn: optional function(text) -> array of TF-IDF scores,
                         one per row of `medicines`.
        """
        self.medicines = medicines
        self.tfidf_scores_fn = tfidf_scores_fn
        self._fuzzy_cache = {}
        self.records = []

        vocab = set()
        for _, row in medicines.iterrows():
            ingredients = parse_ingredients(row.get("active_ingredients", ""))
            vocab.update(ingredients)
            self.records.append({"ingredients": set(ingredients)})
        self.ingredient_vocab = sorted(vocab)

        self.vocab_words = {w for ing in vocab for w in ing.split()}

        for rec, (_, row) in zip(self.records, medicines.iterrows()):
            name = normalize(row.get("name", ""))
            brand = normalize(row.get("brand_name", ""))
            rec["name"] = name
            rec["brand"] = brand
            rec["form"] = normalize(row.get("dosage_form", ""))

            ingredient_words = {
                w for ing in rec["ingredients"] for w in ing.split()
            }
            # Brand key words: not generic words, not numbers, not ingredient
            # words (so "Amoxycillin 500mg Capsule" has no brand key).
            rec["brand_keys"] = [
                w for w in (brand or name).split()
                if w not in GENERIC_WORDS
                and not re.search(r"\d", w)
                and w not in ingredient_words
                and len(w) >= 3
            ]
            rec["numbers"] = _numbers(name)
            rec["ingredient_numbers"] = set(
                re.findall(r"\d+(?:\.\d+)?",
                           re.sub(r"[^0-9.]+", " ",
                                  _clean(row.get("active_ingredients", ""))))
            )

    # ------------------------------------------------------------------
    def _word_in(self, word, tokens, strict=False):
        """
        True if `word` is in `tokens` exactly, or fuzzily for long words.
        A token that is itself a known ingredient word (e.g. "amoxicillin")
        is never fuzzy-matched to a different word (e.g. "ampicillin").
        """
        if word in tokens:
            return True
        if len(word) < MIN_FUZZY_LEN:
            return False
        key = (word, id(tokens))
        if key in self._fuzzy_cache:
            return self._fuzzy_cache[key]
        found = any(
            abs(len(word) - len(t)) <= 2
            and t not in self.vocab_words
            and SequenceMatcher(None, word, t).ratio() >= FUZZY_CUTOFF
            for t in tokens
        )
        self._fuzzy_cache[key] = found
        return found

    def detect_ingredients(self, tokens):
        """Which known ingredients appear in the text (fuzzy for OCR errors)?"""
        found = set()
        for ing in self.ingredient_vocab:
            words = ing.split()
            long_words = [w for w in words if len(w) >= 5]
            check = long_words if long_words else words
            if check and all(self._word_in(w, tokens) for w in check):
                found.add(ing)
        return found

    # ------------------------------------------------------------------
    def rank(self, text, limit=8, ocr_mode=False):
        """Return a list of (row_index, score, confidence, reasons)."""
        norm = normalize(text)
        tokens = set(norm.split())
        self._fuzzy_cache = {}
        numbers = _numbers(norm)
        corrected = _corrected_numbers(tokens) - numbers

        detected = self.detect_ingredients(tokens)
        wants_forms = {
            form for form, words in FORM_KEYWORDS.items()
            if any(w in tokens for w in words)
        }

        tfidf = None
        if self.tfidf_scores_fn is not None:
            tfidf = self.tfidf_scores_fn(text)

        results = []
        for i, rec in enumerate(self.records):
            score = 0.0
            reasons = []
            ing_match = False
            brand_match = False
            form_conflict = False

            # 1. Ingredient-set match (Jaccard: both ingredients beat one)
            if detected and rec["ingredients"]:
                overlap = detected & rec["ingredients"]
                if overlap:
                    union = detected | rec["ingredients"]
                    jaccard = len(overlap) / len(union)
                    score += 0.50 * jaccard
                    if overlap == rec["ingredients"] == detected:
                        ing_match = True
                        reasons.append("all ingredients match")
                    else:
                        reasons.append("partial ingredient match")

            # 2. Brand / product name
            if rec["name"] and rec["name"] in norm:
                score += 0.60
                brand_match = True
                reasons.append("product name found")
            elif rec["brand_keys"] and all(
                self._word_in(w, tokens) for w in rec["brand_keys"][:2]
            ) and not (
                # Short brand words (e.g. "Pan") are easily produced by OCR
                # noise, so in OCR mode require the strength number as well.
                ocr_mode
                and min(len(w) for w in rec["brand_keys"][:2]) < 5
                and not (rec["numbers"] & numbers)
            ):
                score += 0.45
                brand_match = True
                reasons.append("brand name found")

            # 3. Strength numbers from the product name (e.g. 625)
            shared = rec["numbers"] & numbers
            if shared:
                score += 0.15
                reasons.append("strength " + "/".join(sorted(shared)) + " found")

            # 3a. Strength recovered from an OCR digit confusion (E25 -> 625).
            # Only trusted when the brand was also found.
            fixed = (rec["numbers"] & corrected) - shared
            if fixed and brand_match:
                shared = shared | fixed
                score += 0.12
                reasons.append(
                    "strength " + "/".join(sorted(fixed)) + " found (OCR-corrected)"
                )

            # 3b. Per-ingredient amounts (e.g. 500 and 125 on a 625 strip).
            # Helps when the total strength itself was not read.
            amounts = (rec["ingredient_numbers"] & numbers) - shared
            if amounts and not shared:
                score += min(0.05 * len(amounts), 0.10)
                reasons.append("ingredient amount " + "/".join(sorted(amounts)) + " found")

            # 4. Dosage form
            if wants_forms:
                if any(f in rec["form"] for f in wants_forms):
                    score += 0.10
                    reasons.append("dosage form matches")
                elif rec["form"] and rec["form"] != "not available":
                    score -= 0.15
                    form_conflict = True
                    reasons.append("dosage form differs")

            # 5. Small TF-IDF contribution (handles plain name searches)
            if tfidf is not None:
                score += 0.10 * float(tfidf[i])

            strength_ok = (not rec["numbers"]) or bool(shared)
            if (brand_match and strength_ok and not form_conflict
                    and (ing_match or not detected)):
                confidence = "likely"
            elif brand_match or ing_match:
                confidence = "possible"
            else:
                confidence = "weak"

            if score > 0:
                results.append((i, round(score, 3), confidence, reasons))

        results.sort(key=lambda r: r[1], reverse=True)
        return results[:limit], sorted(detected)