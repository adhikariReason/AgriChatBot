# -*- coding: utf-8 -*-
"""Conversation-level tests for symptom triage.

Two things are measured, because either one alone is misleading:
  * does the conversation reach the right intent, and
  * how many questions did it take to get there.

A triage that always asks three questions is no better than a form, and one
that asks none has not triaged anything.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agrichat import AgriEngine                        # noqa: E402
from agrichat.conversation import ANSWER, Conversation  # noqa: E402

# (label, turns, expected intent, max questions allowed)
CASES = [
    ("wide open, devanagari",
     ["मेरो बिरुवा मर्दै छ", "धान", "पात", "खैरो दाग"], "dhan_maruwa_rog", 3),
    ("wide open, roman",
     ["mero bot marna lagyo", "makai", "munta"], "makai_fauji_kira", 2),
    ("rice panicle drying",
     ["बिरुवा बिग्रियो", "धान", "बाला", "सुक्यो"], "dhan_maruwa_rog", 3),
    ("rice deadheart is a pest, not a disease",
     ["समस्या भयो", "धान", "मुन्टा", "सुक्यो"], "dhan_kira", 3),
    ("potato: crop and part given, symptom missing",
     ["आलुको पातमा समस्या छ", "कालो दाग"], "alu_dadhuwa_rog", 1),
    ("cauliflower curd insects, roman, one line",
     ["kauli ko full ma kira lagyo"], "kauli_banda_kira", 0),
    ("ginger rhizome rot, one line",
     ["अदुवाको गानो कुहियो"], "aduwa_gano_kuhine", 0),
    ("cardamom: every rule agrees, so never ask",
     ["alaichi ma samasya cha"], "alaichi_rog", 0),
    ("banana yellowing",
     ["केरा बिग्रियो"], "kera_rog", 0),
    ("tomato",
     ["golbheda ma samasya"], "golbheda_tuta_rog", 0),
    ("goat, livestock branch",
     ["मेरो बाख्रा मर्यो"], "bakhra_khop", 0),
    ("chilli leaf curl",
     ["khursani ko paat khumchiyo"], "khursani_kheti", 0),
    ("onion",
     ["प्याजमा समस्या छ"], "pyaj_lasun", 0),
]

# These must NOT enter triage -- they are how-to questions, not problem reports.
NOT_TRIAGE = [
    "कम्पोस्ट कसरी बनाउने",
    "धानमा मल कति हाल्ने",
    "बाख्रालाई कुन खोप लगाउने",
    "krishi anudan kasari paincha",
    "मौरी पालन कसरी गर्ने",
]


def transcript(engine, turns):
    """Replay a conversation; return (final intent, questions asked, kinds)."""
    convo = Conversation(engine)
    questions, kinds, reply = 0, [], None
    for turn in turns:
        reply = convo.send(turn)
        kinds.append(reply.kind)
        if reply.kind == ANSWER:
            break
        questions += 1
    return (reply.intent if reply else None), questions, kinds


def dump(path: str) -> int:
    """Write the reference transcripts for eval/verify_web_port.js."""
    import json
    engine = AgriEngine()
    rows = []
    for label, turns, expected, _max_q in CASES:
        intent, questions, kinds = transcript(engine, turns)
        rows.append({"label": label, "turns": turns, "expected": expected,
                     "intent": intent, "questions": questions, "kinds": kinds})
    for text in NOT_TRIAGE:
        intent, questions, kinds = transcript(engine, [text])
        rows.append({"label": "not-triage: " + text, "turns": [text],
                     "expected": None, "intent": intent,
                     "questions": questions, "kinds": kinds})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False)
    print(f"wrote {path} ({len(rows)} transcripts)")
    return 0


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--dump":
        return dump(sys.argv[2])
    engine = AgriEngine()
    failures = []
    total_questions = 0

    for label, turns, expected, max_q in CASES:
        convo = Conversation(engine)
        questions = 0
        reply = None
        unused = 0
        for i, turn in enumerate(turns):
            reply = convo.send(turn)
            if reply.kind == ANSWER:
                unused = len(turns) - i - 1
                break
            questions += 1
        total_questions += questions

        if reply is None or reply.kind != ANSWER:
            failures.append(f"{label}: never reached an answer "
                            f"(ended in {reply.kind if reply else 'nothing'})")
        elif unused:
            failures.append(f"{label}: answered with {unused} turn(s) of the "
                            f"scripted conversation unused -- fix the case")
        elif reply.intent != expected:
            failures.append(f"{label}: got {reply.intent}, wanted {expected}")
        elif questions > max_q:
            failures.append(f"{label}: asked {questions} questions, "
                            f"at most {max_q} should be needed")

    for text in NOT_TRIAGE:
        convo = Conversation(engine)
        reply = convo.send(text)
        if reply.kind == "question":
            failures.append(f"how-to question wrongly triaged: {text!r}")

    print(f"triage conversations : {len(CASES)}")
    print(f"questions asked      : {total_questions} "
          f"({total_questions / len(CASES):.1f} per conversation)")
    print(f"how-to questions kept out of triage: {len(NOT_TRIAGE)}")
    if failures:
        print(f"\nFAILED ({len(failures)})")
        for f in failures:
            print("  -", f)
        return 1
    print("\nall diagnostic tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
