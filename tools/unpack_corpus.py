# -*- coding: utf-8 -*-
"""Restore data/corpus/text/ from the committed tarball.

The extracted text is committed as a single gzipped tarball rather than 462
loose files: it is 15MB compressed, and regenerating it means re-crawling
politely rate-limited government sites and re-running OCR over ~7,300 scanned
pages. data/corpus/raw/ (2.4GB of original PDFs) is NOT committed and is
re-downloadable with tools/collect.py.

Run:  python tools/unpack_corpus.py
"""
import os, sys, tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(ROOT, "data", "corpus", "text.tar.gz")

def main() -> int:
    if not os.path.isfile(ARCHIVE):
        print(f"missing {ARCHIVE}", file=sys.stderr)
        return 1
    with tarfile.open(ARCHIVE) as tf:
        members = [m for m in tf.getmembers()
                   if not (m.name.startswith("/") or ".." in m.name.split("/"))]
        tf.extractall(ROOT, members=members)
    print(f"restored {len(members)} entries into data/corpus/text/")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
