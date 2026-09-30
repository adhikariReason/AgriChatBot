# -*- coding: utf-8 -*-
"""Property tests for the text layer.

The engine's whole accuracy story rests on one claim: a Devanagari question
and its Roman Nepali twin produce nearly the same keys.  These tests pin that
down so a future tweak to the transliteration tables cannot quietly break it.

Run:  python eval/test_nepali_text.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agrichat.nepali_text import (  # noqa: E402
    collapse_loose, collapse_tight, has_devanagari, keys, script_of, stems,
)

# (Devanagari, Roman Nepali) pairs that mean the same thing.
PAIRS = [
    ("तपाईंलाई कस्तो छ", "tapailai kasto xa"),
    ("तपाईंलाई कस्तो छ", "tapaailaai kasto cha"),
    ("धानमा मरुवा रोग लाग्यो", "dhan ma maruwa rog lagyo"),
    ("मकैको खेती कसरी गर्ने", "makai ko kheti kasari garne"),
    ("गहुँमा मल कति हाल्ने", "gahu ma mal kati halne"),
    ("आलुमा कीरा लागेको छ", "aalu ma kira lageko cha"),
    ("बाख्रालाई कुन खोप लगाउने", "bakhra lai kun khop lagaune"),
    ("कम्पोस्ट मल कसरी बनाउने", "compost mal kasari banaune"),
    ("धान कहिले रोप्ने", "dhaan kahile ropne"),
    ("प्याज कहिले टिप्ने", "pyaj kahile tipne"),
]

# Spellings of the same Roman word that must collapse together.
SPELLING_VARIANTS = [
    ["cha", "chha", "chhaa"],
    ["kheti", "khetee", "khetii"],
    ["aalu", "alu", "aaloo"],
    ["dhan", "dhaan"],
    ["kasari", "kasaree"],
    ["garne", "garnay" ],
]

failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        return
    failures.append(f"{name}: {detail}")


def overlap(a: str, b: str) -> float:
    """Jaccard overlap of character trigrams."""
    def grams(s):
        s = s.replace(" ", "")
        return {s[i:i + 3] for i in range(max(len(s) - 2, 1))}
    ga, gb = grams(a), grams(b)
    return len(ga & gb) / len(ga | gb) if (ga | gb) else 1.0


def main() -> int:
    check("script_of deva", script_of("धान") == "deva")
    check("script_of latin", script_of("dhan") == "latin")
    check("script_of mixed", script_of("धान ma kira") == "mixed")
    check("has_devanagari", has_devanagari("धान") and not has_devanagari("dhan"))

    # x / chh / ch must all land on the same consonant
    check("xa == chha", collapse_tight("xa") == collapse_tight("chha"),
          f"{collapse_tight('xa')!r} vs {collapse_tight('chha')!r}")

    for variants in SPELLING_VARIANTS:
        loose = {collapse_loose(v) for v in variants}
        check("spelling variants collapse", len(loose) == 1, f"{variants} -> {loose}")

    for deva, roman in PAIRS:
        kd, kr = keys(deva), keys(roman)

        ov = overlap(kd["tight"], kr["tight"])
        check("tight key overlap", ov >= 0.5,
              f"{deva!r} vs {roman!r}: {ov:.2f} ({kd['tight']!r} / {kr['tight']!r})")

        ov = overlap(kd["loose"], kr["loose"])
        check("loose key overlap", ov >= 0.5,
              f"{deva!r} vs {roman!r}: {ov:.2f} ({kd['loose']!r} / {kr['loose']!r})")

        # The point of the stemmer: धानमा and "dhan ma" must share content stems.
        sd, sr = set(stems(deva)), set(stems(roman))
        shared = len(sd & sr) / max(len(sd | sr), 1)
        check("stem overlap", shared >= 0.5,
              f"{deva!r} vs {roman!r}: {shared:.2f} ({sorted(sd)} / {sorted(sr)})")

    # Question words must survive stemming -- "when to plant" and "how to
    # plant" are different intents and only this word separates them.
    check("kahile kept", "kahile" in stems("धान कहिले रोप्ने"), stems("धान कहिले रोप्ने"))
    check("kasari kept", "kasari" in stems("धान कसरी रोप्ने"), stems("धान कसरी रोप्ने"))
    check("kahile != kasari", set(stems("धान कहिले रोप्ने")) != set(stems("धान कसरी रोप्ने")))

    # Postpositions must not survive as standalone features.
    check("postposition dropped", "ko" not in stems("makai ko kheti"), stems("makai ko kheti"))

    # Empty / junk input must not explode.
    for junk in ["", "   ", "???", "123", "‍"]:
        keys(junk)

    if failures:
        print(f"FAILED ({len(failures)})")
        for f in failures:
            print("  -", f)
        return 1
    print("all text-layer tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
