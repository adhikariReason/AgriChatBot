# -*- coding: utf-8 -*-
"""AgriChatBot -- a Nepali-language assistant for farmers in Nepal.

Ask in Devanagari or Roman Nepali; both reach the same answer:

    You: धानमा मरुवा रोग लाग्यो, के गर्ने?
    You: dhan ma maruwa rog lagyo k garne

Usage:
    python chatbot.py                   interactive
    python chatbot.py "dhan ma kira"    single question
    python chatbot.py --debug           show match scores
"""

from __future__ import annotations

import argparse
import sys

from agrichat import AgriEngine
from agrichat.conversation import ANSWER, QUESTION, Conversation

BANNER = """
╭──────────────────────────────────────────────────────────╮
│   कृषि सहयोगी च्याटबोट  ·  AgriChatBot                   │
│   नेपाली किसानका लागि                                    │
╰──────────────────────────────────────────────────────────╯

नेपाली (देवनागरी वा रोमन) मा सोध्नुहोस् — जस्तै:
    धानमा मरुवा रोग लाग्यो, के गर्ने?
    bakhra lai kun khop lagaune
    compost mal kasari banaune

समस्या के हो थाहा नभए पनि हुन्छ — यसो भन्नुहोस्:
    मेरो बिरुवा मर्दै छ
    mero bot marna lagyo
अनि म केही प्रश्न सोधेर पत्ता लगाउँछु।

बाहिर निस्कन: बिदा / exit / quit  (वा Ctrl-C)
"""

EXIT_WORDS = {"exit", "quit", "q", "बिदा", "बन्द", "band"}


def show(reply) -> None:
    print(reply.text)
    if reply.kind == QUESTION and reply.options:
        print()
        for option in reply.options:
            print(f"    • {option}")


def run_once(engine: AgriEngine, question: str, debug: bool = False) -> None:
    if debug:
        ranked = engine.rank(question)
        result = engine.answer(question)
        print(f"[crops detected: {result.crops or '-'}]")
        print("[top 5 intents]")
        for entry, score in ranked[:5]:
            print(f"    {score:6.3f}  {entry.id}")
        print(f"[score {result.score:.3f}  margin {result.margin:.3f}  "
              f"confident {result.confident}]\n")
    show(Conversation(engine).send(question))


def main() -> int:
    ap = argparse.ArgumentParser(description="Nepali agriculture chatbot")
    ap.add_argument("question", nargs="*", help="ask one question and exit")
    ap.add_argument("--debug", action="store_true", help="show matching scores")
    args = ap.parse_args()

    engine = AgriEngine()

    if args.question:
        run_once(engine, " ".join(args.question), args.debug)
        return 0

    print(BANNER)
    print(f"({len(engine.entries)} विषयमा जानकारी उपलब्ध छ)\n")
    convo = Conversation(engine)
    while True:
        try:
            message = input("तपाईं: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nनमस्ते! 🌾")
            return 0
        if not message:
            continue
        if message.lower() in EXIT_WORDS:
            print("नमस्ते! खेतीपातीमा सफलता मिलोस्। 🌾")
            return 0
        print()
        if args.debug:
            result = engine.answer(message)
            print(f"[crops {result.crops or '-'}  score {result.score:.3f}  "
                  f"margin {result.margin:.3f}  confident {result.confident}]")
            print(f"[slots {convo.slots}  pending "
                  f"{convo.pending.slot if convo.pending else None}]\n")
        show(convo.send(message))
        print()


if __name__ == "__main__":
    sys.exit(main())
