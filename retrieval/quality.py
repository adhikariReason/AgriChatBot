# -*- coding: utf-8 -*-
"""Passage-level OCR quality scoring.

About two thirds of the corpus was OCRed with tesseract.  Clean prose survives,
but tables, ingredient lists and trade-name registers come out with Latin
fragments and digits injected into Devanagari ("डा. after अर्याल",
"फोन/४19:54%0", "अनुरोध Bl").  Indexing those passages does not just waste
space -- they pull queries toward rows of noise -- so every passage gets a set
of noise statistics and a verdict: keep, or drop with a named reason.

Signals (all computed on non-space characters / whitespace tokens):

  latin_ratio   Latin letters / (Latin + Devanagari letters).  Real Nepali
                government prose quotes English names, so a little is normal.
  stray_ratio   share of whitespace tokens that are OCR debris -- 1-3 letter
                Latin fragments ("Bl", "aT"), mixed-script tokens
                ("फोन/४19:54%0"), punctuation soup.  Quoted English words,
                acronyms, URLs and plain numbers are NOT debris.
  digit_ratio   digits (either script) / non-space chars.  High = table.
  symbol_ratio  characters that are neither letters, digits nor ordinary
                punctuation.
  malformed     share of Devanagari tokens that are not valid orthography
                (start with a vowel sign/halant, doubled vowel signs ...).
  short_ratio   share of tokens of length <= 2 (table cells, OCR shards).
  deva_letters  absolute amount of Devanagari text (too little = not prose).
"""
from __future__ import annotations

import re

_DEVA_LETTER = re.compile(r"[ऄ-हऽक़-ॡ]")   # vowels+consonants
_DEVA_ANY = re.compile(r"[ऀ-ॿ]")
_DEVA_DIGIT = re.compile(r"[०-९]")
_LATIN = re.compile(r"[A-Za-z]")
_ASCII_DIGIT = re.compile(r"[0-9]")
_DEVA_TOKEN = re.compile(r"[ऀ-ॿ]+")
_SIGN = "ा-्ऀ-ःॢॣ़"      # matras, halant, nukta, marks
_MALFORMED = re.compile(
    r"^[ा-्ऀ-ः]"                       # starts with a dependent sign
    r"|[ा-ौॢॣ][ा-ौॢॣ]"  # two vowel signs in a row
    r"|्[ा-्]"                               # halant + vowel sign / halant
    r"|़़"
)
_URLISH = re.compile(r"(https?://|www\.|@|\.(gov|com|org|np|pdf)\b)", re.I)
_EDGE = ".,;:!?()[]{}'\"-\u2013\u2014/|*\u2022"
_ALLOWED_PUNCT = set(".,;:!?()[]{}'\"-\u2013\u2014/%\u0964\u0965&+=*#@_<>\u00b0")
_UNITS = {"kg", "g", "mg", "ml", "gm", "cm", "mm", "ha", "ppm", "ph", "pH", "ds", "dS", "m", "l", "pc", "no", "sc", "wp", "ec", "sl", "wg", "gr", "wdg", "of", "to", "in", "and", "or", "the", "for"}
_ACRONYM = re.compile(r"^[A-Z][A-Za-z.\-]{1,9}$")            # FAO, PMAMP, DAP, NPK, Dr.
_ENGLISH_WORD = re.compile(r"^[A-Z]?[a-z]{4,}$")                # a properly formed English word
_NUMBER = re.compile(r"^[0-9]{1,4}([./:,-][0-9]{1,4})*%?$")


def is_debris(tok: str) -> bool:
    """Token that looks like OCR debris rather than deliberately quoted English.

    Real Nepali agriculture prose quotes English names ("(Powdery mildew)"),
    acronyms (FAO, DAP), URLs and plain numbers -- those are fine.  Debris is:
    a Latin/digit fragment of 1-3 letters ("Bl", "aT", "Fl"), a token that
    mixes scripts ("फोन/४19:54%0"), or punctuation soup ("s]Gb|,").
    """
    core = tok.strip(_EDGE)
    if not core:
        return False
    has_deva = bool(_DEVA_ANY.search(core))
    has_latin = bool(_LATIN.search(core))
    has_adigit = bool(_ASCII_DIGIT.search(core))
    if has_deva:
        return has_latin or has_adigit                      # mixed-script token
    if not (has_latin or has_adigit):
        # punctuation soup such as "@)*@" (a page number set in a legacy font)
        return len(core) >= 2 and bool(re.search(r"[@#$%^&*=~`\\<>_{}\[\]|+]", core))
    if _URLISH.search(core) or _NUMBER.match(core):
        return False
    if _ENGLISH_WORD.match(core) or (_ACRONYM.match(core) and len(core) >= 3) or core in _UNITS:
        return False
    return True


def clean(text: str):
    """Remove debris tokens (running headers set in a legacy font, OCR shards).

    Returns (cleaned_text, n_removed).  Only applied to passages that already
    passed the filter, so this repairs mild noise rather than rescuing garbage;
    the count is stored on the passage so nothing is changed silently.
    """
    toks = text.split(" ")
    keep = [t for t in toks if not is_debris(t)]
    return " ".join(keep), len(toks) - len(keep)


def stats(text: str) -> dict:
    nonspace = re.sub(r"\s+", "", text)
    n = max(1, len(nonspace))
    deva = len(_DEVA_LETTER.findall(text))
    latin = len(_LATIN.findall(text))
    digits = len(_ASCII_DIGIT.findall(text)) + len(_DEVA_DIGIT.findall(text))
    symbols = sum(
        1 for c in nonspace
        if not (_DEVA_ANY.match(c) or c.isalnum() or c in _ALLOWED_PUNCT)
    )
    toks = text.split()
    nt = max(1, len(toks))
    stray = sum(1 for t in toks if is_debris(t))
    short = sum(1 for t in toks if len(t.strip(".,;:!?()[]।|")) <= 2)
    dtoks = _DEVA_TOKEN.findall(text)
    bad = sum(1 for t in dtoks if _MALFORMED.search(t))
    return {
        "deva_letters": deva,
        "latin_ratio": latin / max(1, latin + deva),
        "stray_ratio": stray / nt,
        "digit_ratio": digits / n,
        "symbol_ratio": symbols / n,
        "malformed": bad / max(1, len(dtoks)),
        "short_ratio": short / nt,
        "deva_share": deva / n,
    }


# Thresholds were set by reading passages either side of each cut (README).
T_LATIN = 0.50        # more than half the letters Latin: an English passage
T_STRAY = 0.08        # > 8% of tokens are OCR debris (see is_debris)
T_DIGIT = 0.25        # numeric table
T_SYMBOL = 0.06
T_MALFORMED = 0.10
T_SHORT = 0.45
MIN_DEVA = 150        # fewer Devanagari letters than this is not Nepali prose


def verdict(text: str):
    """Return (keep: bool, reason: str|None, stats: dict)."""
    s = stats(text)
    if s["deva_letters"] < MIN_DEVA:
        return False, "not_nepali_prose", s
    if s["latin_ratio"] > T_LATIN:
        return False, "mostly_english", s
    if s["stray_ratio"] > T_STRAY:
        return False, "latin_digit_debris", s
    if s["digit_ratio"] > T_DIGIT:
        return False, "numeric_table", s
    if s["symbol_ratio"] > T_SYMBOL:
        return False, "symbol_noise", s
    if s["malformed"] > T_MALFORMED:
        return False, "malformed_devanagari", s
    if s["short_ratio"] > T_SHORT:
        return False, "fragmented", s
    return True, None, s
