"""Per-language settings: how words are cut into alignment units, plus abbreviations.

A *unit* is what MFA treats as a "phone". Usually one letter; multi-letter sounds
(digraphs) are one unit whose time is later split back over its letters.
Letters marked silent (Polish palatal 'i' before a vowel) get zero duration.
"""
import re
import unicodedata

# (unit, condition) -- condition: None = always, 'V' = only if followed by a vowel.
# Longest units first. Each unit: (letters, silent_letter_positions, condition)
MULTI = {
    "hr": [("dž", (), None), ("lj", (), None), ("nj", (), None)],
    "rs": [("dž", (), None), ("lj", (), None), ("nj", (), None)],
    "sr": [("dž", (), None), ("lj", (), None), ("nj", (), None)],
    "si": [],
    "sl": [],
    "cz": [("ch", (), None)],
    "cs": [("ch", (), None)],
    "pl": [
        ("dzi", (2,), "V"),
        ("ci", (1,), "V"), ("si", (1,), "V"), ("zi", (1,), "V"), ("ni", (1,), "V"),
        ("dź", (), None), ("dż", (), None), ("dz", (), None),
        ("cz", (), None), ("sz", (), None), ("rz", (), None), ("ch", (), None),
    ],
}

VOWELS = {"pl": set("aąeęioóuy")}

# Abbreviations that are read out expanded (token must end with '.').
# Extend per language; they get kind 'abbrev' and need variants like numerals.
ABBREV = {
    "hr": {"dr", "mr", "sc", "tzv", "itd", "npr", "tj", "god", "g", "st", "br", "čl", "sl", "prof", "mil", "mlrd", "kn", "min"},
    "rs": {"dr", "mr", "tzv", "itd", "npr", "tj", "god", "g", "st", "br", "čl", "sl", "prof", "mil", "mlrd", "din", "min"},
    "si": {"dr", "mag", "npr", "itd", "oz", "t", "i", "tj", "št", "čl", "odst", "mio", "mrd", "prof", "min", "ga", "go"},
    "cz": {"dr", "ing", "mgr", "např", "atd", "tzv", "tj", "č", "odst", "mil", "mld", "prof", "kč", "min"},
    "pl": {"dr", "prof", "np", "itd", "tzw", "tj", "nr", "art", "ust", "pkt", "mln", "mld", "zł", "ok", "proc", "min", "r"},
}

# Serbian Cyrillic -> Latin (only used if Cyrillic shows up)
CYR = dict(zip("абвгдђежзијклљмнњопрстћуфхцчџш",
               ["a", "b", "v", "g", "d", "đ", "e", "ž", "z", "i", "j", "k", "l", "lj", "m", "n", "nj",
                "o", "p", "r", "s", "t", "ć", "u", "f", "h", "c", "č", "dž", "š"]))


def lang_key(lang):
    return lang.lower()


def normalize_word(core, core_offset, lang):
    """core: token text without edge punctuation; core_offset: its char offset in `text`.
    Returns (norm_chars, orig_offsets): lowercase letters only, each with the
    original character offset it came from (hyphens, apostrophes etc. dropped)."""
    chars, offs = [], []
    for i, ch in enumerate(unicodedata.normalize("NFC", core)):
        if not ch.isalpha():
            continue
        low = ch.lower()
        if lang in ("rs", "sr") and low in CYR:
            low = CYR[low]
        for c in low:
            chars.append(c)
            offs.append(core_offset + i)
    return chars, offs


def split_units(chars, lang):
    """chars: list of lowercase letters. Returns list of
    (unit_label, [norm_char_indices], [silent flags])."""
    rules = MULTI.get(lang, [])
    vowels = VOWELS.get(lang, set())
    s = "".join(chars)
    out, i = [], 0
    while i < len(s):
        hit = None
        for letters, silent, cond in rules:
            n = len(letters)
            if s.startswith(letters, i):
                if cond == "V" and not (i + n < len(s) and s[i + n] in vowels):
                    continue
                hit = (letters, silent, n)
                break
        if hit:
            letters, silent, n = hit
            out.append((letters, list(range(i, i + n)), [k in silent for k in range(n)]))
            i += n
        else:
            out.append((s[i], [i], [False]))
            i += 1
    return out


SAFE = re.compile(r"[^0-9a-z]+")


def safe_key(prefix, s):
    return prefix + SAFE.sub("_", unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()).strip("_")
