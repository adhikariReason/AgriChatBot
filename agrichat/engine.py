# -*- coding: utf-8 -*-
"""Retrieval engine.

Accuracy, not fluency, is the goal here, so the design leans on three things:

1.  Three parallel TF-IDF views of the same utterance (precision key, recall
    key, content stems).  A Devanagari question and its Roman Nepali twin land
    close together in all three -- see nepali_text.
2.  Crop-entity matching, because "धानमा कीरा" and "आलुमा कीरा" are lexically
    almost identical and only the crop word tells them apart.
3.  An explicit refusal to answer.  A wrong confident answer costs a farmer a
    season; asking "did you mean one of these?" costs them one line.  Both the
    absolute score and the margin over the runner-up have to clear a bar.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import kb
from .nepali_text import keys, stems, tokenize

# Crop names, written the way the KB writes them, mapped to every spelling a
# farmer might type.  Matched on the collapsed key so script does not matter.
CROP_ALIASES = {
    "धान": ["धान", "चामल", "dhan", "dhaan", "paddy", "rice", "bhat"],
    "मकै": ["मकै", "makai", "maize", "corn"],
    "गहुँ": ["गहुँ", "गहु", "gahu", "gahun", "wheat"],
    "कोदो": ["कोदो", "kodo", "millet"],
    "फापर": ["फापर", "fapar", "phapar", "buckwheat"],
    "आलु": ["आलु", "alu", "aalu", "potato"],
    "गोलभेँडा": ["गोलभेँडा", "गोलभेडा", "टमाटर", "golbheda", "golbhenda", "tamatar", "tomato"],
    "काउली": ["काउली", "kauli", "cauliflower"],
    "बन्दा": ["बन्दा", "बन्दागोभी", "banda", "cabbage"],
    "खुर्सानी": ["खुर्सानी", "अकबरे", "khursani", "akabare", "dalle", "chilli", "chili"],
    "प्याज": ["प्याज", "pyaj", "pyaz", "onion"],
    "लसुन": ["लसुन", "lasun", "garlic"],
    "तरकारी": ["तरकारी", "साग", "tarkari", "vegetable", "sabji"],
    "अदुवा": ["अदुवा", "aduwa", "aduva", "ginger"],
    "बेसार": ["बेसार", "besar", "turmeric"],
    "अलैंची": ["अलैंची", "अलैची", "alaichi", "alainchi", "cardamom"],
    "केरा": ["केरा", "kera", "banana"],
    "बाख्रा": ["बाख्रा", "बोका", "पाठा", "bakhra", "boka", "patha", "goat"],
    "कुखुरा": ["कुखुरा", "चल्ला", "ब्रोइलर", "kukhura", "challa", "broiler", "layer", "chicken"],
    "गाई": ["गाई", "गोरु", "बाछी", "gai", "gaai", "goru", "cow"],
    "भैंसी": ["भैंसी", "भैसी", "राँगा", "bhaisi", "bhainsi", "buffalo"],
    "माछा": ["माछा", "machha", "macha", "fish"],
    "मौरी": ["मौरी", "मह", "mauri", "maha", "honey", "bee"],
    "घाँस": ["घाँस", "घास", "पराल", "ghans", "ghas", "paral", "napier", "berseem"],
}


def _flat(text: str) -> str:
    """Char n-grams run on the space-free key.

    Devanagari glues postpositions on (धानमा) where Roman Nepali splits them
    (dhan ma); dropping spaces stops that from looking like a difference.
    """
    return text.replace(" ", "")


class Result:
    __slots__ = ("entry", "score", "margin", "confident", "suggestions", "crops")

    def __init__(self, entry, score, margin, confident, suggestions, crops):
        self.entry = entry
        self.score = score
        self.margin = margin
        self.confident = confident
        self.suggestions = suggestions
        self.crops = crops


class AgriEngine:
    # Tuned by `python eval/evaluate.py --tune`.  The stems view carries the
    # most weight because Nepali content words survive stemming intact while
    # the phonetic keys also encode inflection noise; the loose key outranks
    # the tight one because Roman Nepali aspiration is close to random.
    WEIGHTS = {"tight": 0.15, "loose": 0.35, "stems": 0.50}
    MIN_SCORE = 0.32       # below this we never answer
    MIN_MARGIN = 0.045     # top-1 must also beat top-2 by this much
    CROP_BONUS = 0.30      # naming a crop is close to decisive in agronomy,
    CROP_PENALTY = 0.30    # so mismatches are pushed down hard

    # Short queries inflate cosine similarity -- "kun tarkari" (which
    # vegetable) scored 0.758 and was answered with a pest-identification
    # guide. A question carrying almost no content has not been asked yet,
    # whatever it scores, so it needs a real margin before we answer.
    MIN_CONTENT_STEMS = 2
    SHORT_QUERY_MARGIN = 0.35

    def __init__(self, entries=None):
        self.entries = entries if entries is not None else kb.load()
        self.by_id = {e.id: e for e in self.entries}

        # Flatten every pattern, remembering which intent it came from.
        self._owner = []
        views = {"tight": [], "loose": [], "stems": []}
        for idx, entry in enumerate(self.entries):
            for pattern in entry.patterns:
                self._owner.append(idx)
                k = keys(pattern)
                views["tight"].append(_flat(k["tight"]))
                views["loose"].append(_flat(k["loose"]))
                views["stems"].append(k["stems"])
        self._owner = np.asarray(self._owner)

        self._vec = {}
        self._mat = {}
        for name in ("tight", "loose"):
            vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True)
            self._mat[name] = vec.fit_transform(views[name])
            self._vec[name] = vec
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)
        self._mat["stems"] = vec.fit_transform(views["stems"])
        self._vec["stems"] = vec

        # crop name -> collapsed alias keys
        self._crop_keys = {
            crop: {_flat(keys(a)["loose"]) for a in aliases if keys(a)["loose"]}
            for crop, aliases in CROP_ALIASES.items()
        }

    # -- crop detection ----------------------------------------------------

    def detect_crops(self, text: str) -> set:
        """Which crops the user named.

        Matching is per token, not a substring search over the whole
        utterance: भात (rice) collapses to "bat", which sits inside
        "battery", and a raw substring search read "phone ko battery kasto
        hunxa" as a question about धान and answered it with a rice-planting
        guide.

        Prefix matching is allowed only for tokens that were written in
        Devanagari, where morphemes are glued together (धानबाली, धानमा) and a
        prefix is genuinely the head noun. Roman input is space-separated, so
        it never needs prefix matching -- and that is exactly where an English
        word can start with a short alias by accident.
        """
        exact = {_flat(keys(t)["loose"]) for t in tokenize(text)}
        exact |= {_flat(keys(t)["loose"]) for t in stems(text)}
        exact.discard("")

        deva_tokens = set()
        for run in re.findall(r"[\u0900-\u097F]+", text or ""):
            deva_tokens.add(_flat(keys(run)["loose"]))
            deva_tokens |= {_flat(keys(t)["loose"]) for t in stems(run)}
        deva_tokens.discard("")

        found = set()
        for crop, aliases in self._crop_keys.items():
            for alias in aliases:
                if len(alias) < 3:
                    continue
                if alias in exact or any(t.startswith(alias) for t in deva_tokens):
                    found.add(crop)
                    break
        return found

    # -- scoring -----------------------------------------------------------

    def _pattern_scores(self, text: str) -> np.ndarray:
        k = keys(text)
        query = {
            "tight": _flat(k["tight"]),
            "loose": _flat(k["loose"]),
            "stems": k["stems"],
        }
        total = np.zeros(self._mat["tight"].shape[0])
        for name, weight in self.WEIGHTS.items():
            if not query[name].strip():
                continue
            qv = self._vec[name].transform([query[name]])
            # rows are L2-normalized by TfidfVectorizer, so the dot product
            # is already the cosine similarity
            total += weight * (self._mat[name] @ qv.T).toarray().ravel()
        return total

    def rank(self, text: str) -> list:
        """(entry, score) for every intent, best first."""
        pattern_scores = self._pattern_scores(text)

        per_intent = np.zeros(len(self.entries))
        np.maximum.at(per_intent, self._owner, pattern_scores)

        crops = self.detect_crops(text)
        if crops:
            for idx, entry in enumerate(self.entries):
                if not entry.crops:
                    continue
                if crops & set(entry.crops):
                    per_intent[idx] += self.CROP_BONUS
                else:
                    # the user named a crop and this intent is about a
                    # different one -- "धानमा कीरा" must not return the potato
                    # answer just because the rest of the sentence matches
                    per_intent[idx] -= self.CROP_PENALTY

        order = np.argsort(-per_intent)
        return [(self.entries[i], float(per_intent[i])) for i in order]

    def content_stems(self, text: str) -> list:
        """Stems that carry topic, ignoring question words."""
        from .nepali_text import _PROTECTED
        return [s for s in stems(text) if s not in _PROTECTED]

    def answer(self, text: str) -> Result:
        ranked = self.rank(text)
        top, score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = score - runner_up

        min_margin = self.MIN_MARGIN
        if len(self.content_stems(text)) < self.MIN_CONTENT_STEMS:
            min_margin = max(min_margin, self.SHORT_QUERY_MARGIN)

        confident = score >= self.MIN_SCORE and margin >= min_margin
        suggestions = [e for e, s in ranked[:3] if s > 0.12]
        return Result(top, score, margin, confident, suggestions, self.detect_crops(text))

    # -- presentation ------------------------------------------------------

    def reply(self, text: str) -> str:
        """The string a farmer actually sees."""
        if not text or not text.strip():
            return "कृपया आफ्नो प्रश्न लेख्नुहोस्।"

        result = self.answer(text)
        if result.confident:
            out = result.entry.answer
            followups = [self.by_id[f] for f in result.entry.followups[:3] if f in self.by_id]
            if followups:
                out += "\n\nसम्बन्धित विषय:\n" + "\n".join(
                    f"  • {e.patterns[0]}" for e in followups
                )
            return out

        # Not sure -- offer the near misses rather than guess.  A wrong
        # confident answer is worse than no answer for a farmer acting on it.
        if result.suggestions:
            lines = "\n".join(f"  {i}. {e.patterns[0]}" for i, e in enumerate(result.suggestions, 1))
            return (
                "माफ गर्नुहोस्, मैले तपाईंको प्रश्न राम्ररी बुझिनँ।\n\n"
                "तपाईंले यीमध्ये कुनै सोध्न खोज्नुभएको हो?\n" + lines +
                "\n\nहोइन भने बाली वा पशुको नाम र समस्या खुलाएर फेरि सोध्नुहोस् — "
                "जस्तै \"धानको पात खैरो भयो\" वा \"bakhra lai kun khop lagaune\"।"
            )
        return (
            "माफ गर्नुहोस्, यो विषयमा मसँग जानकारी छैन।\n\n"
            "म बाली, तरकारी, नगदे बाली, रोग–कीरा, माटो–मल, सिँचाइ, पशुपन्छी, "
            "अनुदान र बजारबारे सोधिएका प्रश्नको जवाफ दिन सक्छु।\n"
            "नजिकको कृषि ज्ञान केन्द्र वा पालिकाको कृषि शाखामा सम्पर्क गर्नुहोस्।"
        )
