# -*- coding: utf-8 -*-
"""Nepali text handling for AgriChatBot.

Nepali reaches this bot in two very different shapes:

    Devanagari    धानमा मरुवा रोग लाग्यो, के गर्ने?
    Roman Nepali  dhan ma maruwa rog lagyo, k garne?

Roman Nepali has no standard orthography -- "छ" is written cha / chha / chh /
xa depending on the typist -- so matching raw strings is hopeless.  Everything
here exists to push both shapes into one comparable space:

    Devanagari --.
                  >-- detailed roman --> collapse_tight  (precision key)
    Roman ------'                    \-> collapse_loose  (recall key)

Both keys are fed to the retrieval engine, which lets a tight match outrank a
loose one while still catching wildly misspelled input.
"""

from __future__ import annotations

import re
import unicodedata

DEVA_RANGE = re.compile(r"[ऀ-ॿ]")
LATIN_RANGE = re.compile(r"[A-Za-z]")

# --------------------------------------------------------------------------
# Devanagari tables
# --------------------------------------------------------------------------

# Retroflex and dental rows deliberately share a romanization (ट->t like त):
# Nepali speakers rarely distinguish them when typing Roman, and keeping them
# apart only costs recall.
_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "w",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    # nukta forms, in case NFC left them composed
    "क़": "k", "ख़": "kh", "ग़": "g", "ज़": "j", "ड़": "d", "ढ़": "dh", "फ़": "ph",
}

_INDEPENDENT_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "ऍ": "e", "ऑ": "o",
}

_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॅ": "e", "ॉ": "o",
}

_HALANT = "्"
_ANUSVARA = "ं"       # ं  -- a real nasal, keep it
_CHANDRABINDU = "ँ"   # ँ  -- written inconsistently, drop it
_VISARGA = "ः"
_NUKTA = "़"
_AVAGRAHA = "ऽ"

_DEVA_DIGITS = {d: str(i) for i, d in enumerate("०१२३४५६७८९")}

_SKIP = {_CHANDRABINDU, _NUKTA, _AVAGRAHA, "‌", "‍"}


def normalize_unicode(text: str) -> str:
    """NFC-normalize and strip zero-width joiners that break matra sequences."""
    text = unicodedata.normalize("NFC", text or "")
    for ch in ("​", "‌", "‍", "﻿"):
        text = text.replace(ch, "")
    return text


def has_devanagari(text: str) -> bool:
    return bool(DEVA_RANGE.search(text or ""))


def script_of(text: str) -> str:
    """'deva', 'latin', 'mixed' or 'other'."""
    deva = len(DEVA_RANGE.findall(text or ""))
    latin = len(LATIN_RANGE.findall(text or ""))
    if deva and latin:
        return "mixed"
    if deva:
        return "deva"
    if latin:
        return "latin"
    return "other"


def deva_to_roman(text: str) -> str:
    """Devanagari -> detailed roman, keeping aspiration and vowel length.

    The inherent 'a' of a bare consonant is emitted unless the consonant is
    word-final or carries a matra/halant -- so कस्तो becomes ``kasto`` rather
    than ``kasatoa``.
    """
    text = normalize_unicode(text)
    chars = list(text)
    out = []
    n = len(chars)
    for idx, ch in enumerate(chars):
        if ch in _SKIP:
            continue
        if ch in _CONSONANTS:
            out.append(_CONSONANTS[ch])
            nxt = chars[idx + 1] if idx + 1 < n else ""
            # inherent 'a' only when nothing else supplies a vowel and the
            # syllable is not word-final
            if nxt in _MATRAS or nxt == _HALANT:
                continue
            if nxt == "" or not DEVA_RANGE.match(nxt):
                continue  # word-final schwa is not pronounced in Nepali
            if nxt in (_ANUSVARA, _VISARGA, _CHANDRABINDU):
                continue
            out.append("a")
        elif ch in _MATRAS:
            out.append(_MATRAS[ch])
        elif ch in _INDEPENDENT_VOWELS:
            out.append(_INDEPENDENT_VOWELS[ch])
        elif ch == _ANUSVARA:
            out.append("n")
        elif ch == _VISARGA:
            out.append("h")
        elif ch == _HALANT:
            continue
        elif ch in _DEVA_DIGITS:
            out.append(_DEVA_DIGITS[ch])
        else:
            out.append(ch)
    return "".join(out)


# --------------------------------------------------------------------------
# Roman Nepali normalization
# --------------------------------------------------------------------------

# Applied in order; these fold the common typist shortcuts onto the same
# detailed roman spelling that deva_to_roman produces.
_ROMAN_FIXES = [
    (re.compile(r"chhh+"), "chh"),
    (re.compile(r"x"), "chh"),        # kasto xa -> kasto chha
    (re.compile(r"z"), "j"),
    (re.compile(r"f"), "ph"),
    (re.compile(r"v"), "w"),
    (re.compile(r"q"), "k"),
    (re.compile(r"c(?!h)"), "ch"),    # bare c is always an affricate here
    (re.compile(r"ck"), "k"),
    (re.compile(r"sch"), "sh"),
    (re.compile(r"\by\b"), "yo"),
    (re.compile(r"\bk\b"), "ke"),     # "k garne" -> "ke garne"
    (re.compile(r"\bm\b"), "ma"),
    (re.compile(r"\bn\b"), "na"),
    (re.compile(r"\bh\b"), "ho"),
    (re.compile(r"\bcha\b"), "chha"),
    (re.compile(r"\bchan\b"), "chhan"),
    (re.compile(r"\bth\b"), "the"),
]


def normalize_roman(text: str) -> str:
    text = normalize_unicode(text).lower()
    text = re.sub(r"['`’]", "", text)
    for pattern, repl in _ROMAN_FIXES:
        text = pattern.sub(repl, text)
    return text


# --------------------------------------------------------------------------
# Collapse levels
# --------------------------------------------------------------------------

_TIGHT_RULES = [
    (re.compile(r"aa+"), "a"),
    (re.compile(r"ee|ii|y(?=[aeiou])|i+"), "i"),
    (re.compile(r"oo|uu|u+"), "u"),
    (re.compile(r"ay\b"), "e"),
    (re.compile(r"ai|ei"), "e"),
    (re.compile(r"au|ou"), "o"),
    (re.compile(r"sh|ss"), "s"),
    (re.compile(r"ng|ny|nn"), "n"),
    (re.compile(r"w"), "b"),          # व / ब are interchangeable in practice
    (re.compile(r"([kgcjtdpbmnrlsh])\1+"), r"\1"),
]

# Aspiration is the single biggest source of Roman Nepali disagreement
# (khet/ket, chha/cha), so the recall key throws it away entirely.
_LOOSE_RULES = [
    (re.compile(r"chh|ch"), "c"),
    (re.compile(r"kh"), "k"),
    (re.compile(r"gh"), "g"),
    (re.compile(r"jh"), "j"),
    (re.compile(r"th"), "t"),
    (re.compile(r"dh"), "d"),
    (re.compile(r"ph"), "p"),
    (re.compile(r"bh"), "b"),
]


def _to_roman(text: str) -> str:
    """Whatever the script, produce detailed roman."""
    text = normalize_unicode(text)
    if not has_devanagari(text):
        return normalize_roman(text)
    parts = []
    for token in re.findall(r"[ऀ-ॿ]+|[^ऀ-ॿ]+", text):
        if has_devanagari(token):
            parts.append(deva_to_roman(token))
        else:
            parts.append(normalize_roman(token))
    return normalize_roman("".join(parts))


def _clean(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def collapse_tight(text: str) -> str:
    """Precision key: vowel length and sibilants folded, aspiration kept."""
    roman = _to_roman(text)
    for pattern, repl in _TIGHT_RULES:
        roman = pattern.sub(repl, roman)
    return _clean(roman)


def collapse_loose(text: str) -> str:
    """Recall key: tight key with aspiration thrown away as well."""
    roman = _to_roman(text)
    for pattern, repl in _LOOSE_RULES:
        roman = pattern.sub(repl, roman)
    for pattern, repl in _TIGHT_RULES:
        roman = pattern.sub(repl, roman)
    roman = re.sub(r"([kgcjtdpbmnrlsh])\1+", r"\1", roman)
    return _clean(roman)


# --------------------------------------------------------------------------
# Morphology
# --------------------------------------------------------------------------

# Nepali glues its postpositions onto the noun (धानमा = धान + मा) while Roman
# typists usually split them (dhan ma).  Stripping the suffixes makes the two
# spellings meet at the same stem, which matters far more here than it would
# in English.  Longest match first.
_SUFFIXES = [
    "harulai", "haruko", "haruka", "haruki", "harule", "haruma", "harusanga",
    "haru", "haru",
    "bhanda", "bhandapani", "dekhi", "samma", "sanga", "sangai", "bata",
    "madhye", "pachhi", "aghi", "tira", "jasto", "jastai",
    "lai", "ko", "ka", "ki", "le", "ma", "mai", "kai", "sit", "siti",
]

# Verb inflections that carry no topical signal.
_VERB_SUFFIXES = [
    "chhau", "chhan", "chhin", "chhu", "chha", "chh",
    "cau", "can", "cin", "cu", "ca", "c",
    "yeko", "eko", "eka", "eki", "ieko",
    "ne", "nu", "na", "yo", "e",
]

_MIN_STEM = 3

# Short endings like "na"/"e"/"yo" collide with real stems (dhan, biu), so they
# only strip off a comfortably long word.
_RISKY_VERB_SUFFIXES = {"na", "ne", "nu", "yo", "e"}
_MIN_STEM_RISKY = 4

# Question words decide which agronomy intent is meant ("kahile ropne" = when
# to plant vs "kasari ropne" = how to plant), and several of them end in what
# looks like a postposition -- kahi+le.  Never touch them.
_PROTECTED = {
    "kahile", "kasari", "kati", "kaha", "kahan", "kun", "kina", "kasto",
    "kata", "kaho", "kehi", "kahi",
}

# Bare postpositions carry no topic; Roman typists split them off as separate
# tokens where Devanagari glues them on, so they must not survive as features.
_BARE_POSTPOSITIONS = {
    "ko", "ka", "ki", "le", "lai", "bata", "sanga", "dekhi", "samma",
    "haru", "madhye", "bhanda", "tira", "sit",
}

# Deliberately short: Nepali question words (kasari/kahile/kati/kun) are highly
# discriminative for agronomy intents, so they are NOT stopwords.
_STOPWORDS = {
    "ra", "ani", "tara", "pani", "ho", "hoina", "hos", "huns", "hun", "hunch",
    "yo", "tyo", "yi", "ti", "yas", "tyas", "yasto", "tyasto",
    "mero", "hamro", "tapai", "timro", "malai", "hamilai", "tapaile",
    "ma", "mai", "cha", "ca", "chha", "bhaneko", "bhanne", "arko", "aru",
    "sakinch", "sakch", "parch", "garch", "mailey", "la", "ni", "nai",
    "ch", "chh", "c", "chan", "can", "chhan",
}


def strip_suffixes(word: str) -> str:
    """Peel postpositions and verb endings off a collapsed-latin token."""
    if word in _PROTECTED:
        return word
    stripped_postposition = False
    changed = True
    while changed:
        changed = False
        for suf in _SUFFIXES:
            if word.endswith(suf) and len(word) - len(suf) >= _MIN_STEM:
                word = word[: -len(suf)]
                changed = True
                stripped_postposition = True
                break
    if not stripped_postposition:
        for suf in _VERB_SUFFIXES:
            floor = _MIN_STEM_RISKY if suf in _RISKY_VERB_SUFFIXES else _MIN_STEM
            if word.endswith(suf) and len(word) - len(suf) >= floor:
                word = word[: -len(suf)]
                break
    # धान -> dhana -> dhan: the trailing inherent vowel left behind by
    # suffix stripping is noise.
    if len(word) > _MIN_STEM and word.endswith("a"):
        word = word[:-1]
    return word


_WEAK_NASAL = re.compile(r"n(?=[kgcjtdpbmrlsh])")


def drop_weak_nasal(text: str) -> str:
    """Remove nasals sitting before a consonant.

    ं / ँ are written inconsistently in Devanagari and hardly ever typed in
    Roman Nepali, so they are noise for matching.
    """
    return _WEAK_NASAL.sub("", text)


def tokenize(text: str) -> list:
    return [t for t in drop_weak_nasal(collapse_tight(text)).split() if t]


def stems(text: str) -> list:
    """Content stems, stopwords removed."""
    out = []
    for token in tokenize(text):
        if token in _STOPWORDS or token in _BARE_POSTPOSITIONS:
            continue
        stem = strip_suffixes(token)
        if len(stem) < 2 or stem in _STOPWORDS or stem in _BARE_POSTPOSITIONS:
            continue
        out.append(stem)
    return out


def keys(text: str) -> dict:
    """The three views of an utterance the retrieval engine indexes."""
    tight = collapse_tight(text)
    loose = drop_weak_nasal(collapse_loose(text))
    return {
        "tight": tight,
        "loose": loose,
        "stems": " ".join(stems(text)),
    }
