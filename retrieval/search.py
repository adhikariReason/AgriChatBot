# -*- coding: utf-8 -*-
"""Query -> ranked passages.  Hybrid: BGE-M3 dense + BM25 over the project's stems, fused with RRF.

    from retrieval.search import Searcher
    s = Searcher()
    for hit in s.search("dhan ma kira lagyo", k=5):
        print(hit["score"], hit["source_org"], hit["source_url"], hit["text"][:80])

    python -m retrieval.search "dhan ma kira lagyo"
    python -m retrieval.search --mode dense --no-translit "धानमा कीरा लाग्यो"

Query path
    Roman Nepali  -> translit.to_devanagari -> BGE-M3 (dense half)
    Devanagari    ->                          BGE-M3 (dense half)
    either script -> agrichat.nepali_text.keys -> BM25 (lexical half; the collapse
                     keys make script irrelevant, so it gets the raw query)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agrichat.nepali_text import keys            # noqa: E402  (imported, never modified)
from retrieval import translit                    # noqa: E402

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
RRF_K = 60


class BM25:
    """Okapi BM25 over whitespace tokens, uni+bigrams, sparse-matrix scoring."""

    def __init__(self, docs, k1=1.2, b=0.75):
        from sklearn.feature_extraction.text import CountVectorizer
        self.vec = CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                                   ngram_range=(1, 2))
        tf = self.vec.fit_transform(docs).tocsr().astype(np.float32)
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avg = dl.mean() if dl.size else 1.0
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        denom = tf.data + k1 * (1 - b + b * dl[rows] / avg)
        tf.data = (tf.data * (k1 + 1) / denom) * self.idf[tf.indices]
        self.W = tf.tocsc()

    def scores(self, stems: str) -> np.ndarray:
        q = self.vec.transform([stems])
        q.data[:] = 1.0
        return np.asarray(self.W.dot(q.T).todense()).ravel()


class Searcher:
    def __init__(self, art=ART, load_dense=True):
        self.art = art
        with open(os.path.join(art, "passages.jsonl"), encoding="utf-8") as fh:
            self.passages = [json.loads(l) for l in fh]
        with open(os.path.join(art, "lexical_keys.json")) as fh:
            lex = json.load(fh)
        self.bm25 = BM25([r["stems"] for r in lex])
        self.noisy = np.array([p["noisy"] for p in self.passages])
        self.dense = None
        if load_dense and os.path.exists(os.path.join(art, "dense.npy")):
            self.dense = np.load(os.path.join(art, "dense.npy")).astype(np.float32)
            assert len(self.dense) == len(self.passages)
        self._qcache = {}

    # -- query preparation -------------------------------------------------
    @staticmethod
    def prepare(query: str, use_translit: bool = True) -> str:
        """The string that gets embedded."""
        if use_translit and translit.is_roman(query):
            return translit.to_devanagari(query)
        return query

    def embed_query(self, text: str) -> np.ndarray:
        if text not in self._qcache:
            from retrieval import embed
            self._qcache[text] = embed.encode([text], batch_size=1)[0]
        return self._qcache[text]

    # -- rankers -----------------------------------------------------------
    def dense_scores(self, query, use_translit=True):
        if self.dense is None:
            raise RuntimeError("dense index not built: python -m retrieval.index")
        return self.dense @ self.embed_query(self.prepare(query, use_translit))

    def lexical_scores(self, query, use_translit=False):
        q = self.prepare(query, use_translit)
        return self.bm25.scores(keys(q)["stems"])

    @staticmethod
    def _top(scores, n, mask=None):
        s = scores.copy()
        if mask is not None:
            s[mask] = -np.inf
        n = min(n, len(s))
        idx = np.argpartition(-s, n - 1)[:n]
        idx = idx[np.argsort(-s[idx])]
        return [int(i) for i in idx if np.isfinite(s[i]) and (s[i] > 0 or True)]

    def rank(self, query, mode="hybrid", translit_dense=True, translit_lexical=False,
             pool=100, clean_only=False):
        """Return [(passage_index, score)] best first, full pool depth."""
        mask = self.noisy if clean_only else None
        if mode == "dense":
            sc = self.dense_scores(query, translit_dense)
            return [(i, float(sc[i])) for i in self._top(sc, pool, mask)]
        if mode == "lexical":
            sc = self.lexical_scores(query, translit_lexical)
            return [(i, float(sc[i])) for i in self._top(sc, pool, mask) if sc[i] > 0]
        d = [i for i, _ in self.rank(query, "dense", translit_dense, translit_lexical, pool, clean_only)]
        l = [i for i, _ in self.rank(query, "lexical", translit_dense, translit_lexical, pool, clean_only)]
        return rrf([d, l], k=RRF_K)

    def search(self, query, k=5, mode="hybrid", translit_dense=True, translit_lexical=False,
               clean_only=False, pool=100):
        out = []
        for rank, (i, score) in enumerate(self.rank(query, mode, translit_dense, translit_lexical,
                                                     pool, clean_only)[:k], 1):
            p = self.passages[i]
            out.append({
                "rank": rank, "score": score, "pid": p["pid"], "doc_id": p["doc_id"],
                "title": p["title"], "source_url": p["source_url"], "source_org": p["source_org"],
                "licence": p["licence"], "extraction": p["extraction"], "page": p["page"],
                "noisy": p["noisy"], "text": p["text"],
            })
        return out


def rrf(rankings, k=RRF_K):
    """Reciprocal Rank Fusion of several best-first index lists."""
    fused = {}
    for ranking in rankings:
        for r, i in enumerate(ranking, 1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + r)
    return sorted(fused.items(), key=lambda kv: -kv[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--mode", choices=["hybrid", "dense", "lexical"], default="hybrid")
    ap.add_argument("--no-translit", action="store_true", help="embed Roman queries as typed")
    ap.add_argument("--clean-only", action="store_true", help="skip passages flagged noisy")
    a = ap.parse_args()
    s = Searcher(load_dense=a.mode != "lexical")
    print("embedded query:", s.prepare(a.query, not a.no_translit))
    for h in s.search(a.query, a.k, a.mode, translit_dense=not a.no_translit, clean_only=a.clean_only):
        print(f"\n#{h['rank']}  {h['score']:.4f}  {h['source_org']} ({h['extraction']}) p.{h['page']}"
              f"{'  [noisy]' if h['noisy'] else ''}\n   {h['source_url']}\n   {h['text'][:300]}")


if __name__ == "__main__":
    main()
