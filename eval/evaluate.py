# -*- coding: utf-8 -*-
"""Measure retrieval accuracy against eval/testset.json.

Run:  python eval/evaluate.py            report current settings
      python eval/evaluate.py --tune     grid-search weights and thresholds
      python eval/evaluate.py --errors   list every mistake

The headline number is CONFIDENT-WRONG: how often the bot hands a farmer an
answer to a question they did not ask.  Raw top-1 accuracy matters less,
because a low-confidence miss turns into "did you mean ...?" which costs the
user a line, while a confident miss costs them a season.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agrichat.engine import AgriEngine  # noqa: E402

TESTSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testset.json")


def load_cases(path: str = TESTSET) -> list:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["cases"]


def check_leakage(engine: AgriEngine, cases: list) -> list:
    """Test queries that are verbatim knowledge-base patterns.

    Such a case is a guaranteed hit and silently inflates every number here.
    They creep in whenever patterns are added to close a coverage gap, so this
    runs on every evaluation rather than on request.
    """
    patterns = {}
    for entry in engine.entries:
        for pattern in entry.patterns:
            patterns.setdefault(pattern.strip(), entry.id)
    return [(c["q"], patterns[c["q"].strip()]) for c in cases if c["q"].strip() in patterns]


def snapshot(engine: AgriEngine, cases: list) -> list:
    """Score every case once.

    Ranking does not depend on the confidence thresholds, so the sweep can
    reuse this and only re-apply MIN_SCORE / MIN_MARGIN -- which is what makes
    the grid search finish in seconds rather than minutes.
    """
    rows = []
    for case in cases:
        ranked = engine.rank(case["q"])
        top_score = ranked[0][1]
        margin = top_score - (ranked[1][1] if len(ranked) > 1 else 0.0)
        # The engine raises the margin bar for queries carrying almost no
        # content. Recording that floor here keeps the threshold sweep able to
        # vary MIN_SCORE / MIN_MARGIN while still measuring what the engine
        # actually does -- applying fixed thresholds in metrics() alone
        # silently ignored the rule and scored queries as answered that the
        # engine refuses.
        floor = engine.MIN_MARGIN
        if len(engine.content_stems(case["q"])) < engine.MIN_CONTENT_STEMS:
            floor = max(floor, engine.SHORT_QUERY_MARGIN)
        rows.append({
            "q": case["q"],
            "expected": case["expected"],
            "ids": [e.id for e, _ in ranked[:3]],
            "score": top_score,
            "margin": margin,
            "margin_floor": floor,
        })
    return rows


def metrics(rows: list, min_score: float, min_margin: float) -> dict:
    in_scope = [r for r in rows if r["expected"]]
    out_scope = [r for r in rows if not r["expected"]]

    top1 = top3 = answered = answered_right = 0
    confident_wrong, missed, wrong = [], [], []

    def is_confident(r):
        return (r["score"] >= min_score
                and r["margin"] >= max(min_margin, r.get("margin_floor", 0.0)))

    for r in in_scope:
        hit = r["ids"][0] == r["expected"]
        confident = is_confident(r)
        top1 += hit
        top3 += r["expected"] in r["ids"]
        if confident:
            answered += 1
            if hit:
                answered_right += 1
            else:
                confident_wrong.append(r)
        elif hit:
            missed.append(r)
        if not hit:
            wrong.append((r, confident))

    false_answer = [r for r in out_scope if is_confident(r)]
    n = len(in_scope) or 1
    return {
        "n_in_scope": len(in_scope),
        "n_out_scope": len(out_scope),
        "top1": top1 / n,
        "top3": top3 / n,
        "coverage": answered / n,
        "precision": answered_right / answered if answered else 0.0,
        "confident_wrong_rate": len(confident_wrong) / n,
        "oos_refusal": (len(out_scope) - len(false_answer)) / len(out_scope) if out_scope else 1.0,
        "_confident_wrong": confident_wrong,
        "_missed": missed,
        "_wrong": wrong,
        "_false_answer": false_answer,
    }


def evaluate(engine: AgriEngine, cases: list) -> dict:
    return metrics(snapshot(engine, cases), engine.MIN_SCORE, engine.MIN_MARGIN)


def report(m: dict, show_errors: bool = False) -> None:
    print(f"in-scope cases       {m['n_in_scope']}")
    print(f"out-of-scope cases   {m['n_out_scope']}")
    print()
    print(f"top-1 accuracy       {m['top1']:6.1%}   correct intent ranked first")
    print(f"top-3 accuracy       {m['top3']:6.1%}   correct intent in the top 3")
    print(f"coverage             {m['coverage']:6.1%}   answered confidently")
    print(f"answer precision     {m['precision']:6.1%}   of those, correct")
    print(f"CONFIDENT-WRONG      {m['confident_wrong_rate']:6.1%}   <-- the number that hurts farmers")
    print(f"out-of-scope refused {m['oos_refusal']:6.1%}   junk questions correctly declined")

    if not show_errors:
        return
    if m["_confident_wrong"]:
        print("\n--- confident but wrong ---")
        for r in m["_confident_wrong"]:
            print(f"  {r['q']}\n    want {r['expected']}  got {r['ids'][0]} "
                  f"(score {r['score']:.3f}, margin {r['margin']:.3f})")
    if m["_wrong"]:
        print("\n--- top-1 wrong (all, including low-confidence) ---")
        for r, confident in m["_wrong"]:
            print(f"  {'CONF' if confident else '    '} {r['q']}\n"
                  f"         want {r['expected']}  got {r['ids']}")
    if m["_false_answer"]:
        print("\n--- out-of-scope answered anyway ---")
        for r in m["_false_answer"]:
            print(f"  {r['q']}  ->  {r['ids'][0]} ({r['score']:.3f}, margin {r['margin']:.3f})")
    if m["_missed"]:
        print(f"\n--- right intent, too unsure to say it ({len(m['_missed'])}) ---")
        for r in m["_missed"]:
            print(f"  {r['q']}  ({r['ids'][0]}: score {r['score']:.3f}, margin {r['margin']:.3f})")


WEIGHT_GRID = [
    {"tight": 0.40, "loose": 0.30, "stems": 0.30},
    {"tight": 0.34, "loose": 0.33, "stems": 0.33},
    {"tight": 0.50, "loose": 0.25, "stems": 0.25},
    {"tight": 0.30, "loose": 0.20, "stems": 0.50},
    {"tight": 0.25, "loose": 0.25, "stems": 0.50},
    {"tight": 0.45, "loose": 0.15, "stems": 0.40},
    {"tight": 0.20, "loose": 0.40, "stems": 0.40},
    {"tight": 0.15, "loose": 0.35, "stems": 0.50},
]
CROP_GRID = [(0.10, 0.08), (0.12, 0.10), (0.18, 0.15), (0.25, 0.20), (0.30, 0.30), (0.35, 0.40)]
SCORE_GRID = (0.24, 0.28, 0.32, 0.36, 0.40, 0.44)
MARGIN_GRID = (0.02, 0.045, 0.07, 0.10, 0.14, 0.18)


def objective(m: dict) -> float:
    """Coverage is worth having, a confident mistake is worth four of it."""
    return (m["coverage"]
            - 4.0 * m["confident_wrong_rate"]
            - 1.5 * (1.0 - m["oos_refusal"])
            + 0.4 * m["top1"])


def tune(cases: list) -> None:
    engine = AgriEngine()
    best = None
    for weights in WEIGHT_GRID:
        for crop_bonus, crop_penalty in CROP_GRID:
            engine.WEIGHTS = weights
            engine.CROP_BONUS = crop_bonus
            engine.CROP_PENALTY = crop_penalty
            rows = snapshot(engine, cases)          # scored once per config
            for min_score in SCORE_GRID:
                for min_margin in MARGIN_GRID:
                    m = metrics(rows, min_score, min_margin)
                    obj = objective(m)
                    if best is None or obj > best[0]:
                        best = (obj, weights, crop_bonus, crop_penalty,
                                min_score, min_margin, m)

    obj, weights, cb, cp, ms, mm, m = best
    print(f"best configuration (objective {obj:.4f}):\n")
    print(f"    WEIGHTS      = {weights}")
    print(f"    CROP_BONUS   = {cb}")
    print(f"    CROP_PENALTY = {cp}")
    print(f"    MIN_SCORE    = {ms}")
    print(f"    MIN_MARGIN   = {mm}\n")
    report(m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="grid-search the settings")
    ap.add_argument("--errors", action="store_true", help="print every mistake")
    ap.add_argument("--dump", metavar="PATH",
                    help="write the per-case ranking as JSON, for the web-port verifier")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if accuracy has regressed below the floors")
    args = ap.parse_args()

    cases = load_cases()
    if args.tune:
        tune(cases)
        return 0
    engine = AgriEngine()

    leaked = check_leakage(engine, cases)
    if leaked:
        print(f"WARNING: {len(leaked)} test quer{'y is' if len(leaked) == 1 else 'ies are'} "
              f"a verbatim knowledge-base pattern -- these inflate every number below:")
        for q, intent in leaked:
            print(f"    {q}   (pattern of {intent})")
        print()

    rows = snapshot(engine, cases)
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False)
        print(f"wrote {args.dump} ({len(rows)} cases)")
        return 0
    m = metrics(rows, engine.MIN_SCORE, engine.MIN_MARGIN)
    report(m, show_errors=args.errors)

    if args.check:
        # Floors sit a little below the current numbers: they catch a real
        # regression without failing the build over one reworded test case.
        floors = [
            ("top-1 accuracy", m["top1"], 0.88, False),
            ("out-of-scope refusal", m["oos_refusal"], 0.80, False),
            ("confident-wrong rate", m["confident_wrong_rate"], 0.05, True),
        ]
        bad = []
        for name, value, floor, lower_is_better in floors:
            if (value > floor) if lower_is_better else (value < floor):
                bad.append(f"{name} {value:.1%} vs floor {floor:.1%}")
        if check_leakage(engine, cases):
            bad.append("test queries duplicate knowledge-base patterns")
        if bad:
            print("\nREGRESSED:")
            for b in bad:
                print("   ", b)
            return 1
        print("\nall floors met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
