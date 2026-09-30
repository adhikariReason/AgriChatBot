# -*- coding: utf-8 -*-
"""Multi-turn state.

The retrieval engine is stateless: one question in, one answer out.  That is
the right shape for "धानमा मल कति हाल्ने" and the wrong shape for "मेरो बिरुवा
मर्दै छ", which needs two or three questions back before any answer is worth
giving.

This holds the state for that, and decides which mode a turn is in:

    confident retrieval match      -> answer
    symptom report, no match       -> diagnostic questions
    anything else without a match  -> the existing "did you mean ...?"
"""

from __future__ import annotations

from .diagnostic import Diagnostic

# Words that make an utterance a question in its own right rather than an
# answer to ours.  "धान" is a crop slot; "धानमा मल कति हाल्ने" is a new subject.
QUESTION_WORDS = ("कति", "कसरी", "कहिले", "कहाँ", "किन", "कुन", "के गर्ने",
                  "kati", "kasari", "kahile", "kaha", "kina", "kun")

ANSWER = "answer"
QUESTION = "question"
CLARIFY = "clarify"
UNKNOWN = "unknown"


class Reply:
    __slots__ = ("kind", "text", "intent", "options", "slots", "score", "confident")

    def __init__(self, kind, text, intent=None, options=None, slots=None,
                 score=0.0, confident=False):
        self.kind = kind
        self.text = text
        self.intent = intent
        self.options = options or []
        self.slots = slots or {}
        self.score = score
        self.confident = confident

    def __repr__(self):
        return f"<Reply {self.kind} intent={self.intent}>"


class Conversation:
    """One farmer's session."""

    def __init__(self, engine, diagnostic: Diagnostic = None):
        self.engine = engine
        self.diagnostic = diagnostic or Diagnostic(engine)
        self.reset()

    def reset(self) -> None:
        self.slots = {}
        self.pending = None      # the Question we are waiting on
        self.asked = []

    # ---------------------------------------------------------------- turn

    def send(self, text: str) -> Reply:
        text = (text or "").strip()
        if not text:
            return Reply(UNKNOWN, "कृपया आफ्नो प्रश्न लेख्नुहोस्।")

        if self.pending is not None:
            reply = self._continue_diagnostic(text)
            if reply is not None:
                return reply
            # the farmer changed the subject -- drop the thread and start over
            self.reset()

        result = self.engine.answer(text)
        symptom_report = self.diagnostic.looks_like_symptom(text)
        slots = self.diagnostic.extract(text) if symptom_report else {}

        # A problem report that never says what was actually seen is the case
        # triage exists for -- "आलुको पातमा समस्या छ" names a crop and a part
        # but no symptom, and retrieval will happily answer the wrong thing.
        # Any report that does name a symptom goes to retrieval as before.
        underspecified = symptom_report and "symptom" not in slots

        if result.confident and not underspecified:
            self.reset()
            return Reply(ANSWER, self._answer_text(result.entry), result.entry.id,
                         score=result.score, confident=True)

        if symptom_report:
            self.slots = slots
            return self._advance(result)

        return Reply(CLARIFY, self.engine.reply(text),
                     options=[e.label if hasattr(e, "label") else e.patterns[0]
                              for e in result.suggestions],
                     score=result.score)

    # ------------------------------------------------------------ internals

    def _continue_diagnostic(self, text: str):
        # A reply that is itself a question is a change of subject, not an
        # answer to ours.
        if any(w in text for w in QUESTION_WORDS) and self.engine.answer(text).confident:
            return None

        before = dict(self.slots)
        self.slots = self.diagnostic.extract(text, into=self.slots)

        if self.slots == before:
            # nothing understood.  If it reads like a fresh question, let the
            # caller start over; otherwise re-ask, once.
            if self.engine.answer(text).confident or not self.diagnostic.looks_like_symptom(text):
                return None
            q = self.pending
            return Reply(QUESTION,
                         "माफ गर्नुहोस्, बुझिनँ। " + q.text,
                         options=q.options, slots=dict(self.slots))
        return self._advance(None)

    def _advance(self, result) -> Reply:
        question = self.diagnostic.next_question(self.slots)

        if question is not None and question.slot not in self.asked:
            self.pending = question
            self.asked.append(question.slot)
            return Reply(QUESTION, question.text, options=question.options,
                         slots=dict(self.slots))

        intent_id = self.diagnostic.resolve(self.slots)
        entry = self.engine.by_id.get(intent_id) if intent_id else None
        if entry is None:
            self.reset()
            if result is not None:
                return Reply(CLARIFY, self.engine.reply(""), score=result.score)
            return Reply(UNKNOWN,
                         "माफ गर्नुहोस्, यो समस्या पहिचान गर्न सकिनँ। "
                         "नजिकको कृषि ज्ञान केन्द्रमा बोट वा पात देखाउनुहोस्।")

        slots = dict(self.slots)
        self.reset()
        return Reply(ANSWER, self._answer_text(entry, slots), entry.id,
                     slots=slots, confident=True)

    def _answer_text(self, entry, slots: dict = None) -> str:
        out = ""
        if slots:
            said = " · ".join(v for v in (slots.get("crop"), slots.get("part"),
                                          slots.get("symptom")) if v)
            if said:
                out += f"[{said}]\n\n"
        out += entry.answer
        follow = [self.engine.by_id[f] for f in entry.followups[:3]
                  if f in self.engine.by_id]
        if follow:
            out += "\n\nसम्बन्धित विषय:\n" + "\n".join(
                f"  • {e.patterns[0]}" for e in follow)
        return out
