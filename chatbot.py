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

BANNER = """
╭──────────────────────────────────────────────────────────╮
│   कृषि सहयोगी च्याटबोट  ·  AgriChatBot                   │
│   नेपाली किसानका लागि                                    │
╰──────────────────────────────────────────────────────────╯

नेपाली (देवनागरी वा रोमन) मा सोध्नुहोस् — जस्तै:
    धानमा मरुवा रोग लाग्यो, के गर्ने?
    bakhra lai kun khop lagaune
    compost mal kasari banaune

बाहिर निस्कन: बिदा / exit / quit  (वा Ctrl-C)
"""

EXIT_WORDS = {"exit", "quit", "q", "बिदा", "बन्द", "band"}


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
    print(engine.reply(question))


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
        run_once(engine, message, args.debug)
        print()


if __name__ == "__main__":
    sys.exit(main())
