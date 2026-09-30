# -*- coding: utf-8 -*-
"""Roman Nepali -> Devanagari, for the query side of dense retrieval.

BGE-M3 and friends have seen Devanagari Nepali but essentially no Roman
Nepali, so "dhan ma kira lagyo" must become "धानमा कीरा लाग्यो" before it is
embedded.  (agrichat.nepali_text goes the other way, Devanagari -> roman, and
deliberately throws information away; this module is the reverse and is
deliberately *not* an inverse of it.)

Two tiers, tried in order for every word:

1. Corpus lexicon.  Every Devanagari word in the indexed corpus is folded with
   the project's own collapse keys (agrichat.nepali_text.collapse_tight /
   collapse_loose), so a typist's "dhaan", "dhan" or "dhann" all meet at the
   corpus's most frequent surface form धान.  This absorbs Roman Nepali's
   spelling chaos (aspiration, vowel length, cha/chha/xa) using exactly the
   equivalences the existing chatbot already relies on, and it resolves the
   schwa ambiguity (dhan = धन or धान?) by what the corpus actually says.
2. Phonetic rules, for words the corpus has never seen (names, typos).

Postpositions typed as separate words ("dhan ma", "gahu ko") are glued back on
(धानमा, गहुँको) because that is how Devanagari is written and therefore how
the embedding model has seen it.
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agrichat.nepali_text import (                     # noqa: E402  (imported, never modified)
    collapse_loose, collapse_tight, drop_weak_nasal, has_devanagari, normalize_roman,
)

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
LEXICON_CACHE = os.path.join(ART, "translit_lexicon.json")

_DEVA_WORD = re.compile(r"^[ऄ-हऽक़-ॡा-्ऀ-ः़]+$")
_MALFORMED_START = re.compile(r"^[ा-्ऀ-ः]")

# postposition (as typed) -> Devanagari suffix
POSTPOSITIONS = {
    "ma": "मा", "mai": "मै", "ko": "को", "ka": "का", "ki": "की", "lai": "लाई",
    "le": "ले", "bata": "बाट", "dekhi": "देखि", "samma": "सम्म", "sanga": "सँग",
    "sita": "सित", "haru": "हरू", "harulai": "हरूलाई", "haruko": "हरूको",
    "haruma": "हरूमा", "harule": "हरूले", "tira": "तिर", "jasto": "जस्तो",
}

# ---------------------------------------------------------------------------
# tier 2: phonetic rules
# ---------------------------------------------------------------------------
_CONS = [
    ("chh", "छ"), ("ksh", "क्ष"), ("kh", "ख"), ("gh", "घ"), ("ch", "च"), ("jh", "झ"),
    ("th", "थ"), ("dh", "ध"), ("ph", "फ"), ("bh", "भ"), ("sh", "श"), ("gy", "ज्ञ"),
    ("k", "क"), ("g", "ग"), ("j", "ज"), ("t", "त"), ("d", "द"), ("n", "न"),
    ("p", "प"), ("b", "ब"), ("m", "म"), ("y", "य"), ("r", "र"), ("l", "ल"),
    ("w", "व"), ("v", "व"), ("s", "स"), ("h", "ह"), ("f", "फ"), ("z", "ज"),
    ("q", "क"), ("c", "च"), ("x", "छ"),
]
_VOWELS = [  # (roman, independent, matra)
    ("aa", "आ", "ा"), ("ai", "ऐ", "ै"), ("au", "औ", "ौ"), ("ee", "ई", "ी"),
    ("ii", "ई", "ी"), ("oo", "ऊ", "ू"), ("uu", "ऊ", "ू"),
    ("a", "अ", ""), ("i", "इ", "ि"), ("u", "उ", "ु"), ("e", "ए", "े"), ("o", "ओ", "ो"),
]
_CONS_SORTED = sorted(_CONS, key=lambda kv: -len(kv[0]))
_VOW_SORTED = sorted(_VOWELS, key=lambda v: -len(v[0]))
_HAL = "्"


def _match(table, word, i, key=lambda e: e[0]):
    for entry in table:
        r = key(entry)
        if word.startswith(r, i):
            return entry
    return None


def rule_translit(word: str) -> str:
    """Phonetic fallback for one lowercase roman word."""
    word = re.sub(r"[^a-z]", "", word.lower())
    out, i, n = [], 0, len(word)
    prev_cons = False           # last emitted item is a bare consonant awaiting a vowel
    while i < n:
        v = _match(_VOW_SORTED, word, i)
        if v:
            roman, indep, matra = v
            if prev_cons:
                out.append(matra)          # "a" gives "" (inherent vowel)
            else:
                out.append(indep)
            prev_cons = False
            i += len(roman)
            continue
        c = _match(_CONS_SORTED, word, i)
        if c:
            roman, deva = c
            if prev_cons:
                # consonant directly after a consonant: join them
                out.append(_HAL)
            # nasal before a stop after a vowel is almost always an anusvara
            elif roman == "n" and out and i + 1 < n and word[i + 1] in "kgjtdpbc" and not _is_vowel(word, i + 1):
                out.append("ं")
                i += 1
                prev_cons = False
                continue
            out.append(deva)
            prev_cons = True
            i += len(roman)
            continue
        i += 1
    return "".join(out)


def _is_vowel(word, i):
    return word[i] in "aeiou"


# ---------------------------------------------------------------------------
# tier 1: corpus lexicon
# ---------------------------------------------------------------------------
# single-letter "words" are almost always list markers (क), ख) ...) in this
# corpus; only these are real one-letter Nepali words.
_ONE_LETTER_OK = {"र", "म", "न", "छ", "त", "थ"}


# words whose most frequent corpus reading is wrong for a farmer query
# (पनि "also" outnumbers पानी "water" in running text; रा is a list-marker)
_OVERRIDES = {"ra": "र", "pani": "पानी"}


def _fold(key: str) -> str:
    """e/i and o/u are interchangeable in Roman Nepali typing (bemari/bimari)."""
    return key.replace("e", "i").replace("o", "u")


def _skeleton(key: str) -> str:
    """Key with every 'a' removed: typists drop the schwa that deva_to_roman keeps
    (golbheda vs golabheda, upchar vs upachar)."""
    sk = key.replace("a", "")
    return sk if len(sk) >= 2 else key


class Lexicon:
    def __init__(self, tight, loose, skeleton, fold=None):
        self.tight = tight       # key -> deva word (most frequent)
        self.loose = loose
        self.skeleton = skeleton
        self.fold = fold or {}

    @classmethod
    def from_counts(cls, counts, min_freq=2, min_freq_skeleton=3):
        by_tight = collections.defaultdict(collections.Counter)
        by_loose = collections.defaultdict(collections.Counter)
        by_skel = collections.defaultdict(collections.Counter)
        by_fold = collections.defaultdict(collections.Counter)
        for w, c in counts.items():
            if c < min_freq or not _DEVA_WORD.match(w) or _MALFORMED_START.match(w):
                continue
            if len(w) == 1 and w not in _ONE_LETTER_OK:
                continue
            kt = drop_weak_nasal(collapse_tight(w)).replace(" ", "")
            kl = drop_weak_nasal(collapse_loose(w)).replace(" ", "")
            if kt:
                by_tight[kt][w] += c
            if kl:
                by_loose[kl][w] += c
                by_fold[_fold(kl)][w] += c
                if c >= min_freq_skeleton:
                    by_skel[_skeleton(_fold(kl))][w] += c
        pick = lambda d: {k: v.most_common(1)[0][0] for k, v in d.items()}
        return cls(pick(by_tight), pick(by_loose), pick(by_skel), pick(by_fold))

    def lookup(self, roman_word: str):
        if roman_word in _OVERRIDES:
            return _OVERRIDES[roman_word]
        rw = normalize_roman(roman_word)
        if not rw:
            return None
        rw = re.sub(r"(?<=[aeiou])y$", "", rw)       # garney / diney -> garne / dine
        kt = drop_weak_nasal(collapse_tight(rw)).replace(" ", "")
        if kt in self.tight:
            return self.tight[kt]
        kl = drop_weak_nasal(collapse_loose(rw)).replace(" ", "")
        if kl in self.loose:
            return self.loose[kl]
        # typed a final "a" that Devanagari-side keys drop (ra -> र)
        if len(kl) > 1 and kl.endswith("a") and kl[:-1] in self.loose:
            return self.loose[kl[:-1]]
        if _fold(kl) in self.fold:
            return self.fold[_fold(kl)]
        # typed without the internal schwa (upchar -> उपचार)
        return self.skeleton.get(_skeleton(_fold(kl)))


def build_lexicon(passages=None, save=True):
    if passages is None:
        with open(os.path.join(ART, "passages.jsonl"), encoding="utf-8") as fh:
            passages = [json.loads(l)["text"] for l in fh]
    counts = collections.Counter()
    for t in passages:
        for tok in re.findall(r"[ऀ-ॿ]+", t):
            counts[tok] += 1
    lex = Lexicon.from_counts(counts)
    if save:
        os.makedirs(ART, exist_ok=True)
        with open(LEXICON_CACHE, "w", encoding="utf-8") as fh:
            json.dump({"tight": lex.tight, "loose": lex.loose, "skeleton": lex.skeleton, "fold": lex.fold}, fh, ensure_ascii=False)
    return lex


_lex = None


def get_lexicon():
    global _lex
    if _lex is None:
        if os.path.exists(LEXICON_CACHE):
            with open(LEXICON_CACHE, encoding="utf-8") as fh:
                d = json.load(fh)
            _lex = Lexicon(d["tight"], d["loose"], d["skeleton"], d.get("fold"))
        else:
            _lex = build_lexicon()
    return _lex


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def word_to_deva(word: str, lexicon=None, use_lexicon=True) -> str:
    lex = lexicon or (get_lexicon() if use_lexicon else None)
    if lex is not None:
        hit = lex.lookup(word)
        if hit:
            return hit
    return rule_translit(word)


def to_devanagari(text: str, glue=True, use_lexicon=True) -> str:
    """Transliterate the roman-script parts of ``text``; Devanagari passes through.

    Numbers and punctuation are kept.  Returns the Devanagari string that should
    be embedded.
    """
    if not text:
        return text
    tokens = re.findall(r"[A-Za-z']+|[ऀ-ॿ]+|[0-9]+|[^\sA-Za-z0-9ऀ-ॿ]", text)
    out = []
    prev_content = False       # previous output token can take a glued postposition
    for tok in tokens:
        if re.match(r"[A-Za-z']", tok):
            low = tok.lower().replace("'", "")
            if glue and low in POSTPOSITIONS and prev_content:
                out[-1] += POSTPOSITIONS[low]
                prev_content = True
                continue
            out.append(word_to_deva(low, use_lexicon=use_lexicon))
            prev_content = low not in POSTPOSITIONS
        elif re.match(r"[ऀ-ॿ0-9]", tok):
            out.append(tok)
            prev_content = True
        else:
            out.append(tok)
            prev_content = False
    s = " ".join(out)
    return re.sub(r"\s+([?,.!;:])", r"\1", s)


def is_roman(text: str) -> bool:
    """True when the query is (mostly) Latin script, i.e. needs transliterating."""
    letters = re.findall(r"[A-Za-zऀ-ॿ]", text or "")
    if not letters:
        return False
    deva = sum(1 for c in letters if has_devanagari(c))
    return deva / len(letters) < 0.5


if __name__ == "__main__":
    for q in sys.argv[1:] or ["dhan ma kira lagyo", "gahu ko rog ko upchar k ho", "bakhra lai kun khop lagaune"]:
        print(q, "->", to_devanagari(q))
