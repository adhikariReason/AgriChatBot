# -*- coding: utf-8 -*-
"""Retrieval evaluation: does dense help for Nepali, and what does Roman Nepali cost?

    python -m retrieval.evaluate                 # the report (needs judgements.json)
    python -m retrieval.evaluate --pool          # build the judging sheet (see below)
    python -m retrieval.evaluate --record q01 3,5 [--note "..."]

Design
  queries.json   40 information needs, each written twice by hand: once in
                 Devanagari, once in Roman Nepali the way a farmer would type it
                 (split postpositions, inconsistent spellings).  Same question in
                 both scripts, so a script difference is not a topic difference.
  judgements.json  my relevance judgements.  For each question: the passages I
                 read and judged (`judged`) and which of them answer it
                 (`relevant`).  Judged by pooling: the top results of *every*
                 system and condition below are pooled, so no system is graded only
                 on its own picks.  A retrieved passage that is not in `judged`
                 counts as NOT relevant, and the report says how many there were.
  metric         hit@5: a relevant passage is among the top 5 (also hit@1, MRR@10).

Conditions (columns of the report)
  dense            BGE-M3 cosine
  lexical          BM25 over agrichat.nepali_text stems (the chatbot's content view)
  tfidf            engine-style TF-IDF (stems word n-grams + loose-key char n-grams)
  rrf              dense + lexical fused with Reciprocal Rank Fusion (k=60)
Query variants
  deva             Devanagari query, embedded directly
  roman raw        Roman query embedded as typed
  roman translit   Roman query -> Devanagari (translit.py) -> embedded
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agrichat.nepali_text import keys                 # noqa: E402
from retrieval import translit                         # noqa: E402
from retrieval.search import Searcher, rrf             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
QUERIES = os.path.join(HERE, "queries.json")
JUDGEMENTS = os.path.join(HERE, "judgements.json")
POOL = os.path.join(ART, "pool.json")
K = 5
DEPTH = 100


# ---------------------------------------------------------------------------
# engine-style TF-IDF baseline (the chatbot's scoring idea, applied to passages)
# ---------------------------------------------------------------------------
class EngineTfidf:
    """stems word 1-2grams (w=.59) + loose-key char 3-5grams (w=.41).

    The chatbot weighs tight/loose/stems at .15/.35/.50; the tight view is
    dropped here (it is nearly redundant with loose and doubles memory) and the
    remaining weights renormalised.
    """

    def __init__(self, art=ART):
        from sklearn.feature_extraction.text import TfidfVectorizer
        with open(os.path.join(art, "lexical_keys.json")) as fh:
            rows = json.load(fh)
        self.v_stem = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                                      tokenizer=str.split, token_pattern=None, lowercase=False)
        self.m_stem = self.v_stem.fit_transform([r["stems"] for r in rows])
        self.v_loose = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True,
                                      min_df=2, dtype=np.float32)
        self.m_loose = self.v_loose.fit_transform([r["loose"].replace(" ", "") for r in rows])

    def scores(self, query: str) -> np.ndarray:
        k = keys(query)
        s1 = (self.m_stem @ self.v_stem.transform([k["stems"]]).T).toarray().ravel()
        s2 = (self.m_loose @ self.v_loose.transform([k["loose"].replace(" ", "")]).T).toarray().ravel()
        return 0.59 * s1 + 0.41 * s2


# ---------------------------------------------------------------------------
# running the systems
# ---------------------------------------------------------------------------
def top_idx(scores, n=DEPTH):
    idx = np.argpartition(-scores, n - 1)[:n]
    return [int(i) for i in idx[np.argsort(-scores[idx])]]


class Systems:
    def __init__(self):
        self.s = Searcher()
        self.tf = EngineTfidf()
        self.cache = {}

    def lexical(self, q, tl=False):
        return top_idx(self.s.lexical_scores(q, tl))

    def dense(self, q, tl=True):
        return top_idx(self.s.dense_scores(q, tl))

    def tfidf(self, q, tl=False):
        return top_idx(self.tf.scores(self.s.prepare(q, tl)))

    def run(self, q, script):
        """All (system, variant) rankings for one query.  -> {name: [idx...]}"""
        r = {}
        if script == "roman_noisy":
            script = "roman"
        if script == "deva":
            d, l, t = self.dense(q), self.lexical(q), self.tfidf(q)
            r.update({"dense": d, "lexical": l, "tfidf": t, "rrf": [i for i, _ in rrf([d, l])]})
            return r
        d_raw, d_tl = self.dense(q, False), self.dense(q, True)
        l_raw, l_tl = self.lexical(q, False), self.lexical(q, True)
        t_raw, t_tl = self.tfidf(q, False), self.tfidf(q, True)
        r.update({
            "dense|raw": d_raw, "dense|translit": d_tl,
            "lexical|raw": l_raw, "lexical|translit": l_tl,
            "tfidf|raw": t_raw, "tfidf|translit": t_tl,
            "rrf|raw": [i for i, _ in rrf([d_raw, l_raw])],
            "rrf|translit": [i for i, _ in rrf([d_tl, l_raw])],          # shipped config
            "rrf|translit-both": [i for i, _ in rrf([d_tl, l_tl])],
        })
        return r


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# pooling + recording judgements
# ---------------------------------------------------------------------------
def build_pool(depth=5, oracle=6, only_new=False):
    """Write the judging sheet: per question, every passage any system put in its top `depth`."""
    sysm = Systems()
    queries = load_json(QUERIES)
    pool = {}
    for q in queries:
        cand = []
        for script in ("deva", "roman", "roman_noisy"):
            if script == "roman_noisy" and "roman_noisy" not in q:
                continue
            runs = sysm.run(q[script], script)
            for name, ranking in runs.items():
                for i in ranking[:depth]:
                    if i not in cand:
                        cand.append(i)
        # oracle candidates: passages containing ALL of the question's key terms,
        # densest first, so a relevant passage that *every* system missed can still be
        # judged (otherwise pooling would understate how often systems miss)
        terms = q.get("terms", [])
        if terms:
            scored = []
            for i, p in enumerate(sysm.s.passages):
                t = p["text"]
                if all(x in t for x in terms):
                    scored.append((sum(t.count(x) for x in terms) / len(t), i))
            for _, i in sorted(scored, reverse=True)[:oracle]:
                if i not in cand:
                    cand.append(i)
        pids = [sysm.s.passages[i]["pid"] for i in cand]
        if only_new:
            done = set((load_json(JUDGEMENTS, {"queries": {}})["queries"].get(q["id"]) or {}).get("judged", []))
            pids = [p for p in pids if p not in done]
        pool[q["id"]] = pids
    os.makedirs(ART, exist_ok=True)
    with open(POOL, "w", encoding="utf-8") as fh:
        json.dump(pool, fh, ensure_ascii=False)
    print("pool sizes:", {k: len(v) for k, v in pool.items()})
    return pool


def show_pool(qid, width=380):
    pool = load_json(POOL)
    queries = {q["id"]: q for q in load_json(QUERIES)}
    with open(os.path.join(ART, "passages.jsonl"), encoding="utf-8") as fh:
        by_pid = {}
        for line in fh:
            p = json.loads(line)
            by_pid[p["pid"]] = p
    q = queries[qid]
    print(f"=== {qid} [{q['topic']}]  {q['deva']}   |   {q['roman']}")
    for n, pid in enumerate(pool[qid], 1):
        p = by_pid[pid]
        print(f"[{n}] {p['title'][:45]} ({p['extraction']}{',noisy' if p['noisy'] else ''}) :: "
              f"{p['text'][:width]}")


def record(qid, relevant_numbers, note="", extend=False):
    pool = load_json(POOL)
    j = load_json(JUDGEMENTS, {"rubric": RUBRIC, "queries": {}})
    pids = pool[qid]
    rel = [pids[int(n) - 1] for n in relevant_numbers if str(n).strip()]
    if extend and qid in j["queries"]:
        old = j["queries"][qid]
        j["queries"][qid] = {"judged": old["judged"] + pids, "relevant": old["relevant"] + rel,
                             "note": (old.get("note", "") + " | " + note).strip(" |")}
    else:
        j["queries"][qid] = {"judged": pids, "relevant": rel, "note": note}
    with open(JUDGEMENTS, "w", encoding="utf-8") as fh:
        json.dump(j, fh, ensure_ascii=False, indent=1)
    print(qid, "recorded:", len(rel), "relevant of", len(pids), "judged")


RUBRIC = ("A passage is relevant if it contains information that would help answer the "
          "question as asked: it must be about the specific crop/animal/pest/topic named AND "
          "say something substantive about what was asked (how to manage, grow, prevent, "
          "what it is).  Passages that merely mention the topic in a list, table of contents, "
          "budget line or administrative report are not relevant.  Garbled OCR that I could "
          "not read as an answer is not relevant.")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def metrics(ranking_pids, relevant):
    rel = set(relevant)
    top = ranking_pids[:10]
    first = next((r for r, p in enumerate(top, 1) if p in rel), None)
    return {
        "hit1": float(bool(top[:1]) and top[0] in rel),
        "hit5": float(any(p in rel for p in top[:K])),
        "mrr": 1.0 / first if first else 0.0,
    }


def paired_ci(a, b, n=5000, seed=0):
    """Bootstrap 95% CI for mean(a)-mean(b) over queries."""
    rng = random.Random(seed)
    a, b = np.asarray(a), np.asarray(b)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(len(a)) for _ in range(len(a))]
        diffs.append(a[idx].mean() - b[idx].mean())
    diffs.sort()
    return float(a.mean() - b.mean()), diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def token_recovery(queries, field="roman"):
    """How much of the Devanagari query does transliterating the Roman one recover?"""
    def toks(s):
        return [t for t in re.findall(r"[ऀ-ॿ]+", s)]
    def norm(t):                       # ignore chandrabindu/anusvara and ZWJ noise
        return re.sub("[ँं‌‍]", "", t)
    out = {}
    for label, use_lex in (("lexicon+rules", True), ("rules only", False)):
        hit = tot = 0
        for q in queries:
            gold = {norm(t) for t in toks(q["deva"])}
            got = {norm(t) for t in toks(translit.to_devanagari(q[field], use_lexicon=use_lex))}
            hit += len(gold & got)
            tot += len(gold)
        out[label] = hit / tot
    return out


def report(out_path=None):
    queries = load_json(QUERIES)
    judg = load_json(JUDGEMENTS)
    if not judg:
        sys.exit("no judgements.json: run --pool, then --record for each question")
    J = judg["queries"]
    missing = [q["id"] for q in queries if q["id"] not in J]
    if missing:
        sys.exit(f"unjudged questions: {missing}")
    sysm = Systems()
    pids = [p["pid"] for p in sysm.s.passages]
    noisy = [p["noisy"] for p in sysm.s.passages]

    # per-query per-condition metric rows
    rows = {}       # (script, cond) -> list of metric dicts
    unjudged = {}   # cond -> number of top-5 passages not judged
    noisy_share = {}
    answerable = [q for q in queries if J[q["id"]]["relevant"]]
    for q in queries:
        rel = J[q["id"]]["relevant"]
        judged = set(J[q["id"]]["judged"])
        for script in ("deva", "roman", "roman_noisy"):
            for cond, ranking in sysm.run(q[script], script).items():
                rp = [pids[i] for i in ranking]
                rows.setdefault((script, cond), []).append(metrics(rp, rel))
                unjudged[(script, cond)] = unjudged.get((script, cond), 0) + sum(p not in judged for p in rp[:K])
                noisy_share.setdefault((script, cond), []).extend(noisy[i] for i in ranking[:K])

    def agg(key, sub=None):
        r = rows[key]
        if sub is not None:
            r = [m for m, q in zip(r, queries) if q["id"] in sub]
        return {k: 100 * np.mean([m[k] for m in r]) for k in ("hit1", "hit5", "mrr")}

    lines = []
    P = lines.append
    ans_ids = {q["id"] for q in answerable}
    P(f"questions: {len(queries)} (x2 scripts = {2 * len(queries)} queries); "
      f"{len(answerable)} have >=1 relevant passage in the judged pool")
    P(f"passages indexed: {len(pids)}")
    P("")
    for label, sub in (("ALL questions", None), ("ANSWERABLE questions only (>=1 relevant passage exists)", ans_ids)):
        P(f"--- hit@5 (%), {label} ---")
        P(f"{'':28s} {'dense':>8s} {'lexical':>8s} {'tfidf':>8s} {'RRF':>8s}")
        P(f"{'Devanagari query':28s} " + " ".join(f"{agg(('deva', c), sub)['hit5']:8.1f}" for c in ("dense", "lexical", "tfidf", "rrf")))
        P(f"{'Roman, as typed (no translit)':28s} " + " ".join(f"{agg(('roman', c + '|raw'), sub)['hit5']:8.1f}" for c in ("dense", "lexical", "tfidf", "rrf")))
        P(f"{'Roman -> Devanagari (translit)':28s} " + " ".join(f"{agg(('roman', c + '|translit'), sub)['hit5']:8.1f}" for c in ("dense", "lexical", "tfidf", "rrf")))
        P(f"{'SLOPPY Roman, as typed':28s} " + " ".join(f"{agg(('roman_noisy', c + '|raw'), sub)['hit5']:8.1f}" for c in ("dense", "lexical", "tfidf", "rrf")))
        P(f"{'SLOPPY Roman -> translit':28s} " + " ".join(f"{agg(('roman_noisy', c + '|translit'), sub)['hit5']:8.1f}" for c in ("dense", "lexical", "tfidf", "rrf")))
        P("   (lexical/tfidf 'translit' = transliterate the lexical query as well; RRF 'translit' = dense query")
        P("    transliterated, lexical query raw -- the configuration search.py ships)")
        P("")
    P("--- hit@1 and MRR@10 (%), answerable questions ---")
    for script, conds in (("deva", ["dense", "lexical", "tfidf", "rrf"]),
                          ("roman", ["dense|raw", "dense|translit", "lexical|raw", "lexical|translit",
                                     "tfidf|raw", "rrf|raw", "rrf|translit", "rrf|translit-both"]),
                          ("roman_noisy", ["dense|raw", "dense|translit", "lexical|raw", "rrf|raw", "rrf|translit"])):
        for c in conds:
            m = agg((script, c), ans_ids)
            P(f"  {script:5s} {c:20s} hit@1 {m['hit1']:5.1f}   MRR {m['mrr']:5.1f}")
    P("")
    P("--- what transliteration is worth (hit@5 points, answerable questions; 95% paired bootstrap CI) ---")
    sel = [i for i, q in enumerate(queries) if q["id"] in ans_ids]
    def col(key):
        return [rows[key][i]["hit5"] for i in sel]
    for label, a, b in [
        ("dense:  Roman translit - Roman raw", ("roman", "dense|translit"), ("roman", "dense|raw")),
        ("RRF:    Roman translit - Roman raw", ("roman", "rrf|translit"), ("roman", "rrf|raw")),
        ("dense:  Devanagari - Roman translit  (cost of Roman that remains)", ("deva", "dense"), ("roman", "dense|translit")),
        ("dense:  Devanagari - Roman raw       (total cost of Roman, no fix)", ("deva", "dense"), ("roman", "dense|raw")),
        ("RRF:    Devanagari - Roman translit", ("deva", "rrf"), ("roman", "rrf|translit")),
        ("lexical: Devanagari - Roman raw", ("deva", "lexical"), ("roman", "lexical|raw")),
        ("lexical: Roman translit - Roman raw", ("roman", "lexical|translit"), ("roman", "lexical|raw")),
        ("SLOPPY dense:  translit - raw", ("roman_noisy", "dense|translit"), ("roman_noisy", "dense|raw")),
        ("SLOPPY RRF:    translit - raw", ("roman_noisy", "rrf|translit"), ("roman_noisy", "rrf|raw")),
        ("SLOPPY vs clean Roman, dense translit", ("roman_noisy", "dense|translit"), ("roman", "dense|translit")),
        ("SLOPPY dense translit vs Devanagari", ("roman_noisy", "dense|translit"), ("deva", "dense")),
    ]:
        d, lo, hi = paired_ci(col(a), col(b))
        P(f"  {label:66s} {100 * d:+6.1f}  [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    P("")
    P("--- does dense beat lexical? (hit@5 points, answerable questions; 95% paired CI) ---")
    for label, a, b in [
        ("Devanagari: dense - lexical", ("deva", "dense"), ("deva", "lexical")),
        ("Devanagari: dense - tfidf", ("deva", "dense"), ("deva", "tfidf")),
        ("Devanagari: RRF - lexical", ("deva", "rrf"), ("deva", "lexical")),
        ("Devanagari: RRF - dense", ("deva", "rrf"), ("deva", "dense")),
        ("Roman: dense(translit) - lexical(raw)", ("roman", "dense|translit"), ("roman", "lexical|raw")),
        ("Roman: RRF(translit) - lexical(raw)", ("roman", "rrf|translit"), ("roman", "lexical|raw")),
        ("Roman: RRF(translit) - dense(translit)", ("roman", "rrf|translit"), ("roman", "dense|translit")),
    ]:
        d, lo, hi = paired_ci(col(a), col(b))
        P(f"  {label:66s} {100 * d:+6.1f}  [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    P("  -- same comparisons on hit@1 (harder, less saturated) --")
    def col1(key, m="hit1"):
        return [rows[key][i][m] for i in sel]
    for label, a, b in [
        ("Devanagari: dense - lexical   hit@1", ("deva", "dense"), ("deva", "lexical")),
        ("Devanagari: dense - tfidf     hit@1", ("deva", "dense"), ("deva", "tfidf")),
        ("Devanagari: RRF - dense       hit@1", ("deva", "rrf"), ("deva", "dense")),
        ("Devanagari: RRF - lexical     hit@1", ("deva", "rrf"), ("deva", "lexical")),
        ("Roman: dense(translit) - lexical(raw) hit@1", ("roman", "dense|translit"), ("roman", "lexical|raw")),
        ("Roman: dense(translit) - dense(raw)   hit@1", ("roman", "dense|translit"), ("roman", "dense|raw")),
        ("SLOPPY Roman: dense(translit) - lexical(raw) hit@1", ("roman_noisy", "dense|translit"), ("roman_noisy", "lexical|raw")),
    ]:
        d, lo, hi = paired_ci(col1(a), col1(b))
        P(f"  {label:66s} {100 * d:+6.1f}  [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    P("")
    P("--- transliteration quality: share of the Devanagari query's words recovered from the Roman query ---")
    for field in ("roman", "roman_noisy"):
        for k, v in token_recovery(queries, field).items():
            P(f"  {field:12s} {k:16s} {100 * v:5.1f}%")
    P("")
    P("--- share of top-5 results that are flagged noisy (OCR debris / malformed) ---")
    for key in (("deva", "dense"), ("deva", "lexical"), ("deva", "rrf"), ("roman", "rrf|translit"), ("roman_noisy", "rrf|translit")):
        P(f"  {key[0]:5s} {key[1]:16s} {100 * np.mean(noisy_share[key]):5.1f}%   (corpus base rate {100 * np.mean(noisy):.1f}%)")
    P("")
    tot_top5 = len(queries) * K
    P("--- judging coverage: top-5 passages not in the judged pool (counted as non-relevant) ---")
    worst = max(unjudged.items(), key=lambda kv: kv[1])
    P(f"  max over systems: {worst[1]} of {tot_top5} ({worst[0][0]} {worst[0][1]}); "
      f"mean {np.mean(list(unjudged.values())):.1f}")
    P("")
    P("--- per-question hit@5  (D=dense L=lexical R=rrf; upper = Devanagari, lower = Roman translit) ---")
    for i, q in enumerate(queries):
        def f(key):
            return "Y" if rows[key][i]["hit5"] else "."
        nrel = len(J[q["id"]]["relevant"])
        P(f"  {q['id']} rel={nrel:2d} deva D{f(('deva','dense'))} L{f(('deva','lexical'))} R{f(('deva','rrf'))}"
          f" | roman-raw D{f(('roman','dense|raw'))} L{f(('roman','lexical|raw'))} R{f(('roman','rrf|raw'))}"
          f" | roman-translit D{f(('roman','dense|translit'))} R{f(('roman','rrf|translit'))}"
          f" | sloppy-translit D{f(('roman_noisy','dense|translit'))} R{f(('roman_noisy','rrf|translit'))}  {q['topic']}")
    text = "\n".join(lines)
    print(text)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="store_true", help="build the judging sheet")
    ap.add_argument("--show", metavar="QID", nargs="+", help="print a question's pooled passages")
    ap.add_argument("--record", nargs=2, metavar=("QID", "NUMBERS"), help="e.g. q01 1,4,7  (use '-' for none)")
    ap.add_argument("--note", default="")
    ap.add_argument("--new", action="store_true", help="with --pool: only passages not yet judged; with --record: extend")
    ap.add_argument("--width", type=int, default=380)
    ap.add_argument("--out", default=os.path.join(HERE, "results.txt"))
    a = ap.parse_args()
    if a.pool:
        build_pool(only_new=a.new)
    elif a.show:
        for q in a.show:
            show_pool(q, a.width)
    elif a.record:
        nums = [] if a.record[1] == "-" else a.record[1].split(",")
        record(a.record[0], nums, a.note, extend=a.new)
    else:
        report(a.out)


if __name__ == "__main__":
    main()
