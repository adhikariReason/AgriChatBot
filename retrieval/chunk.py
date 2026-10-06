# -*- coding: utf-8 -*-
"""Documents -> passages.

Only manifest records with ``index_recommended`` are chunked.  Every passage
carries the provenance a search result needs to be cited: doc id, source_url,
licence, source_org, extraction method and the page it starts on.

Chunking rules
  * 600-1000 characters (soft target ~800), never splitting mid-sentence unless
    a single sentence exceeds the maximum.
  * Paragraph (blank-line) boundaries are preferred break points.
  * Devanagari danda (।, ॥), the OCR'd pipe ("|") and ? ! end sentences.
  * ~150 characters (whole trailing sentences) are repeated at the start of the
    next passage so an answer that straddles a boundary is still retrievable.
  * "[page N]" markers are removed from the text and tracked as metadata.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "data", "corpus", "manifest.json")

MIN_CHARS, TARGET, MAX_CHARS, OVERLAP = 600, 800, 1000, 150

_PAGE = re.compile(r"^\[page (\d+)\]\s*$")
# sentence end: danda / double danda / ? / ! / a pipe that OCR used for danda /
# an ASCII full stop followed by space (abbreviations like डा. are rare enough).
_SENT_END = re.compile(r"(?<=[।॥?!])\s+|(?<=\s[|])\s+|(?<=[ऀ-ॿ]\.)\s+(?=[ऀ-ॿ])")


def load_records(only_recommended: bool = True):
    with open(MANIFEST, encoding="utf-8") as fh:
        recs = json.load(fh)
    return [r for r in recs if r.get("index_recommended")] if only_recommended else recs


def read_text(rec) -> str:
    path = rec["text_path"]
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    if not os.path.exists(path):
        alt = os.path.join(ROOT, "data", "corpus", "text", os.path.basename(path))
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"{path} -- run `python tools/unpack_corpus.py`")
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _units(text: str):
    """Yield (page, sentence, paragraph_break_before) units."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("​", "").replace("﻿", "")
    # U+FFFD marks a glyph the PDF font could not map; in this corpus it sits
    # where a vowel sign was lost, so dropping it repairs more than it harms.
    text = text.replace(chr(0xFFFD), "")
    page = 1
    para, para_page = [], 1
    paras = []

    def flush():
        nonlocal para
        if para:
            paras.append((para_page, " ".join(para)))
            para = []

    for line in text.split("\n"):
        m = _PAGE.match(line.strip())
        if m:
            flush()
            page = int(m.group(1))
            continue
        line = line.strip()
        if not line:
            flush()
            continue
        if not para:
            para_page = page
        para.append(line)
    flush()

    for pg, p in paras:
        p = re.sub(r"\s+", " ", p).strip()
        first = True
        for s in _SENT_END.split(p):
            s = s.strip()
            if not s:
                continue
            # hard-split absurdly long "sentences" (tables, runs without danda)
            while len(s) > MAX_CHARS:
                cut = s.rfind(" ", MIN_CHARS, MAX_CHARS)
                cut = cut if cut > 0 else MAX_CHARS
                yield pg, s[:cut].strip(), first
                first = False
                s = s[cut:].strip()
            if s:
                yield pg, s, first
                first = False


def chunk_text(text: str):
    """Return list of (start_page, passage_text)."""
    out = []
    cur, cur_len, n_new = [], 0, 0

    def emit():
        nonlocal cur, cur_len, n_new
        out.append((cur[0][0], " ".join(u[1] for u in cur)))
        # carry whole trailing sentences worth >= OVERLAP chars into the next chunk
        keep, n = [], 0
        for u in reversed(cur):
            if n >= OVERLAP:
                break
            keep.insert(0, u)
            n += len(u[1]) + 1
        if n > OVERLAP * 2.2:          # a single huge sentence: don't repeat it
            keep = []
        cur, cur_len, n_new = keep, sum(len(u[1]) + 1 for u in keep), 0

    for pg, s, para_start in _units(text):
        add = len(s) + 1
        if n_new and cur_len + add > MAX_CHARS:
            emit()
        elif n_new and para_start and cur_len >= TARGET:
            emit()                      # paragraph boundary near the target size
        if cur_len + add > MAX_CHARS:   # overlap would overflow: drop it
            cur, cur_len = [], 0
        cur.append((pg, s))
        cur_len += add
        n_new += 1
    if n_new:
        body = " ".join(u[1] for u in cur)
        if len(body) < MIN_CHARS and out and len(out[-1][1]) + len(body) + 1 <= MAX_CHARS + 200:
            # fold a short tail into the previous passage instead of a runt
            new_only = " ".join(u[1] for u in cur[len(cur) - n_new:])
            out[-1] = (out[-1][0], out[-1][1] + " " + new_only)
        else:
            out.append((cur[0][0], body))
    return out


def chunk_corpus(records=None):
    """All passages for the recommended records, with provenance."""
    records = records if records is not None else load_records()
    passages = []
    for rec in records:
        for i, (page, body) in enumerate(chunk_text(read_text(rec))):
            passages.append({
                "pid": f"{rec['id']}#{i}",
                "doc_id": rec["id"],
                "title": rec.get("title", ""),
                "source_url": rec["source_url"],
                "source_org": rec["source_org"],
                "licence": rec["licence"],
                "extraction": rec["extraction"],
                "agro_density": rec.get("agro_density"),
                "page": page,
                "text": body,
            })
    return passages


if __name__ == "__main__":
    ps = chunk_corpus()
    import statistics
    L = [len(p["text"]) for p in ps]
    print(len(ps), "passages; chars median", statistics.median(L), "min", min(L), "max", max(L))
