# -*- coding: utf-8 -*-
"""Build and save the passage index.

    python -m retrieval.index            # chunk, quality-filter, embed, save
    python -m retrieval.index --no-dense # everything except the embeddings

Steps
  1. chunk the 210 ``index_recommended`` documents          (chunk.py)
  2. quality filter: drop passages with OCR noise            (quality.py)
  3. drop exact duplicate passages
  4. lexical keys via agrichat.nepali_text.keys              (cached)
     + the transliteration lexicon (corpus vocabulary)       (translit.py)
  5. BGE-M3 dense embeddings, sharded + resumable            (embed.py)

Everything lands in retrieval/artifacts/ (git-ignored).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agrichat.nepali_text import keys            # noqa: E402  (imported, never modified)
from retrieval import chunk, quality             # noqa: E402

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
SHARD = 512

# mild-noise flag: kept, but search results can be filtered with clean_only
NOISY_STRAY, NOISY_MALFORMED = 0.03, 0.05


def path(name):
    return os.path.join(ART, name)


def build_passages():
    """Chunk + filter.  Returns (kept passages, report dict, dropped rows)."""
    allp = chunk.chunk_corpus()
    reasons = collections.Counter()
    by_ext = collections.defaultdict(collections.Counter)
    per_doc = collections.defaultdict(lambda: collections.Counter())
    kept, dropped, seen = [], [], set()
    dups = 0
    for p in allp:
        ok, reason, st = quality.verdict(p["text"])
        ext = p["extraction"]
        if ok:
            h = hashlib.md5(re.sub(r"\s+", "", p["text"]).encode()).hexdigest()
            if h in seen:
                ok, reason = False, "duplicate"
                dups += 1
            else:
                seen.add(h)
        if not ok:
            reasons[reason] += 1
            by_ext[ext][reason] += 1
            per_doc[p["doc_id"]]["dropped"] += 1
            dropped.append({"pid": p["pid"], "doc_id": p["doc_id"], "reason": reason,
                            "extraction": ext, "stats": {k: round(v, 3) for k, v in st.items()},
                            "preview": p["text"][:160]})
            continue
        p["noisy"] = bool(st["stray_ratio"] > NOISY_STRAY or st["malformed"] > NOISY_MALFORMED)
        p["quality"] = {k: round(v, 3) for k, v in st.items()}
        p["text"], p["debris_removed"] = quality.clean(p["text"])
        kept.append(p)
        by_ext[ext]["kept"] += 1
        per_doc[p["doc_id"]]["kept"] += 1
    report = {
        "documents": len({p["doc_id"] for p in allp}),
        "documents_with_kept_passages": len({p["doc_id"] for p in kept}),
        "passages_after_chunking": len(allp),
        "passages_after_quality_filter": len(kept),
        "dropped_total": len(dropped),
        "dropped_by_reason": dict(reasons),
        "by_extraction": {k: dict(v) for k, v in by_ext.items()},
        "kept_flagged_noisy": sum(p["noisy"] for p in kept),
        "thresholds": {k: getattr(quality, k) for k in dir(quality) if k.startswith("T_")} | {"MIN_DEVA": quality.MIN_DEVA},
        "noisy_flag": {"debris": NOISY_STRAY, "malformed": NOISY_MALFORMED},
        "debris_tokens_removed_from_kept": sum(p["debris_removed"] for p in kept),
        "kept_with_debris_removed": sum(p["debris_removed"] > 0 for p in kept),
    }
    return kept, report, dropped


def save_lexical(kept):
    rows = []
    for p in kept:
        k = keys(p["text"])
        rows.append({"stems": k["stems"], "loose": k["loose"]})
    with open(path("lexical_keys.json"), "w") as fh:
        json.dump(rows, fh)


def build_dense(kept, log):
    from retrieval import embed
    os.makedirs(path("emb"), exist_ok=True)
    texts = [p["text"] for p in kept]
    n = len(texts)
    t_start = time.time()
    done_before = 0
    for s in range(0, n, SHARD):
        f = path(f"emb/part_{s // SHARD:04d}.npy")
        if os.path.exists(f):
            done_before += 1
            continue
        t0 = time.time()
        part = embed.encode(texts[s:s + SHARD], batch_size=16)
        np.save(f + ".tmp.npy", part.astype(np.float16))
        os.replace(f + ".tmp.npy", f)
        el = time.time() - t_start
        log(f"shard {s // SHARD + 1}/{(n + SHARD - 1) // SHARD} "
            f"({len(part) / (time.time() - t0):.1f} passages/s, elapsed {el / 60:.1f} min)")
    parts = [np.load(path(f"emb/part_{i:04d}.npy")) for i in range((n + SHARD - 1) // SHARD)]
    mat = np.concatenate(parts).astype(np.float16)
    assert mat.shape[0] == n, (mat.shape, n)
    np.save(path("dense.npy"), mat)
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dense", action="store_true")
    args = ap.parse_args()
    os.makedirs(ART, exist_ok=True)
    logf = open(path("build.log"), "a")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    t0 = time.time()
    kept, report, dropped = build_passages()
    with open(path("passages.jsonl"), "w", encoding="utf-8") as fh:
        for p in kept:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(path("dropped.jsonl"), "w", encoding="utf-8") as fh:
        for d in dropped:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    log(f"chunked {report['passages_after_chunking']} -> kept {len(kept)}; dropped {report['dropped_by_reason']}")
    save_lexical(kept)
    from retrieval import translit
    translit.build_lexicon([p["text"] for p in kept])      # corpus vocabulary for roman->Devanagari
    report["lexical_keys_seconds"] = round(time.time() - t0, 1)
    if not args.no_dense:
        t1 = time.time()
        build_dense(kept, log)
        report["dense_build_seconds"] = round(time.time() - t1, 1)
        report["note"] = "dense_build_seconds covers only this run; resumed shards are skipped"
    with open(path("build_report.json"), "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
