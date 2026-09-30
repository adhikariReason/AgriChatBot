# -*- coding: utf-8 -*-
"""Symptom triage.

A farmer does not type a well-formed question.  They type "मेरो बिरुवा मर्दै छ"
or "bot marna lagyo" -- my plant is dying.  No amount of retrieval answers that
well, because the question is underspecified: it is missing the crop, the
affected part and the symptom, and those three decide everything.

So this asks back, the way a plant clinic does.  The one rule that keeps it
from being tedious: **never ask a question whose answer cannot change the
outcome.**  If every rule still in play points at the same intent, answer now.

All the knowledge lives in data/diagnostic/rules.json; the logic here is
deliberately small so the browser port stays cheap.
"""

from __future__ import annotations

import json
import os

from .nepali_text import collapse_loose, keys

RULES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "diagnostic", "rules.json")

WILDCARD = "*"
SLOT_ORDER = ("crop", "part", "symptom")


def _flat(text: str) -> str:
    return text.replace(" ", "")


class Question:
    """A question to put back to the farmer."""
    __slots__ = ("slot", "text", "options")

    def __init__(self, slot, text, options):
        self.slot = slot
        self.text = text
        self.options = options      # list of display labels

    def __repr__(self):
        return f"<Question {self.slot}: {self.text}>"


class Diagnostic:
    def __init__(self, engine, rules_path: str = RULES_PATH):
        with open(rules_path, encoding="utf-8") as fh:
            self.spec = json.load(fh)
        self.engine = engine
        self.rules = self.spec["rules"]

        # alias -> canonical value, keyed on the collapsed form so that script
        # and spelling stop mattering (फूल / ful / phool / full all land here)
        self._alias = {}
        for slot, cfg in self.spec["slots"].items():
            table = {}
            for option in cfg["options"]:
                for alias in option["aliases"]:
                    k = _flat(collapse_loose(alias))
                    if k:
                        table[k] = option["value"]
            self._alias[slot] = table

        self._triggers = {_flat(collapse_loose(t)) for t in self.spec["triggers"]}
        self._triggers.discard("")

    # -- reading the farmer's words ---------------------------------------

    def looks_like_symptom(self, text: str) -> bool:
        """Is this a problem report rather than a how-to question?"""
        blob = _flat(collapse_loose(text))
        return any(t in blob for t in self._triggers)

    def extract(self, text: str, into: dict = None) -> dict:
        """Pull whatever slots the text already answers.

        Longer aliases win, so "पूरै बोट" is not read as "बोट".
        """
        slots = dict(into or {})
        blob = _flat(collapse_loose(text))

        crops = self.engine.detect_crops(text)
        if crops and "crop" not in slots:
            slots["crop"] = sorted(crops)[0]

        for slot in ("part", "symptom"):
            if slot in slots:
                continue
            best = None
            for alias_key, value in self._alias[slot].items():
                if alias_key and alias_key in blob:
                    if best is None or len(alias_key) > len(best[0]):
                        best = (alias_key, value)
            if best:
                slots[slot] = best[1]
        return slots

    # -- narrowing ---------------------------------------------------------

    def matching(self, slots: dict) -> list:
        """Rules still consistent with what we know."""
        out = []
        for rule in self.rules:
            ok = True
            for slot in SLOT_ORDER:
                known = slots.get(slot)
                want = rule.get(slot, WILDCARD)
                if known is not None and want != WILDCARD and want != known:
                    ok = False
                    break
            if ok:
                out.append(rule)

        # Crop-specific knowledge beats the generic fallbacks.  Without this,
        # knowing the crop is अलैंची still leaves the crop="*" rules in play,
        # their differing intents make the outcome look undecided, and triage
        # asks a question that cannot change the answer.
        if slots.get("crop"):
            specific = [r for r in out if r.get("crop", WILDCARD) != WILDCARD]
            if specific:
                return specific
        return out

    def resolve(self, slots: dict) -> str:
        """The intent the surviving rules agree on, most specific first."""
        candidates = self.matching(slots)
        if not candidates:
            return None

        def specificity(rule):
            return sum(1 for s in SLOT_ORDER if rule.get(s, WILDCARD) != WILDCARD)

        candidates.sort(key=specificity, reverse=True)
        return candidates[0]["intent"]

    def next_question(self, slots: dict):
        """The next slot worth asking about, or None if asking cannot help."""
        candidates = self.matching(slots)
        if not candidates:
            return None
        outcomes = {r["intent"] for r in candidates}
        if len(outcomes) <= 1:
            return None          # the answer is already decided -- do not ask

        for slot in SLOT_ORDER:
            if slot in slots:
                continue
            # would knowing this actually split the surviving rules?
            values = {r.get(slot, WILDCARD) for r in candidates}
            if values == {WILDCARD}:
                continue
            if slot == "crop":
                return Question("crop", self.spec["crop_question"],
                                self.spec["crop_examples"])
            cfg = self.spec["slots"][slot]
            # offer only options that are actually still possible
            live = {r.get(slot) for r in candidates} | {WILDCARD}
            options = [o["label"] for o in cfg["options"]
                       if o["value"] in live or WILDCARD in live]
            return Question(slot, cfg["question"], options)
        return None
