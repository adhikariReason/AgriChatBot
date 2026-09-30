# retrieval/ -- hybrid passage search over the Nepali agriculture corpus

Dense (BGE-M3) + lexical (BM25 over the chatbot's own collapse keys), fused with
Reciprocal Rank Fusion, over the 210 documents in `data/corpus/manifest.json`
whose `index_recommended` is true.  Nothing outside `retrieval/` was modified.

```
Roman Nepali query --> translit.py (roman -> Devanagari) --> BGE-M3 --.
Devanagari query   ------------------------------------> BGE-M3 -----+-- RRF --> passages
either script      --> agrichat.nepali_text.keys --> BM25 -------------'
```

## Rebuild and query

```bash
pip install -r retrieval/requirements.txt
python tools/unpack_corpus.py             # only if data/corpus/text/ is missing
python -m retrieval.index                 # chunk, filter, embed, save  (~47 min on 4 CPU cores)
python -m retrieval.index --no-dense      # everything but the embeddings (~1.3 min)
python -m retrieval.search "dhan ma khairo phadke kira lagyo"
python -m retrieval.search --mode dense --no-translit "dhan ma kira lagyo"   # see what Roman costs
python -m retrieval.evaluate              # the measurement (~1.5 min) -> results.txt
```

```python
from retrieval.search import Searcher
s = Searcher()
for h in s.search("dhan ma kira lagyo", k=5):          # mode="hybrid" | "dense" | "lexical"
    print(h["score"], h["source_org"], h["licence"], h["source_url"], h["page"], h["noisy"])
```

Every hit carries `pid`, `doc_id`, `source_url`, `source_org`, `licence`,
`extraction` (`ocr`/`pdf-text`), `page` and a `noisy` flag.  `clean_only=True`
skips passages flagged noisy.  Everything built lives in `retrieval/artifacts/`
(git-ignored, ~150 MB).  Resumable: re-running `index` skips finished embedding
shards.

| file | |
|---|---|
| `chunk.py` | manifest -> passages, 600-1000 chars, sentence/paragraph aware (danda, `\|`, `?`, `!`), ~150 chars overlap, `[page N]` markers tracked as metadata, U+FFFD removed |
| `quality.py` | OCR-noise statistics, keep/drop verdict, debris-token cleaner |
| `translit.py` | roman -> Devanagari for queries |
| `embed.py` | BGE-M3 via plain `transformers` (CLS, L2-normalised, bf16 autocast on CPU) |
| `index.py` | builds and saves everything |
| `search.py` | `Searcher`, BM25, RRF |
| `evaluate.py`, `queries.json`, `judgements.json`, `results.txt` | the measurement, its 40 hand-written questions, my relevance judgements, its latest output |
| `build_report.json` | counts from the last build |

## Numbers from the last build

| | passages |
|---|---:|
| after chunking (210 docs) | 24,469 |
| dropped: exact duplicates | 1,396 |
| dropped: not Nepali prose (< 150 Devanagari letters; English abstracts, bare tables) | 2,563 |
| dropped: mostly English (> 50 % of letters Latin) | 699 |
| dropped: OCR / font debris (> 8 % of tokens are debris) | 3,046 |
| dropped: numeric tables, malformed Devanagari, fragments, symbol noise | 493 |
| **indexed** | **16,272** (8,109 OCR, 8,163 pdf-text; 198 of the 210 docs) |

All drops are listed with their reason and statistics in
`artifacts/dropped.jsonl`.  Of the indexed passages, 11,065 had debris tokens
(legacy-font running headers such as `s[lif tyf kz'kG5L 8fo/L @)*@`, `Bl`, `aT`,
`फोन/४19:54%0`) removed -- 49,466 tokens in total, recorded per passage in
`debris_removed` -- and 5,956 are flagged `noisy` (>= 3 % debris or >= 5 %
malformed Devanagari before cleaning).

Things worth knowing about the filter:

* The worst noise is **not** in the OCRed files.  The 51 "pdf-text" documents lose
  more passages (legacy Preeti-font text that decodes as Latin punctuation soup)
  than the 159 OCR ones.
* Quoted English is deliberately not debris (`(Powdery mildew)`, `FAO`, units,
  URLs, numbers).  An earlier, cruder "Latin ratio" rule threw away clean
  bilingual disease descriptions; that is why the rule is token-based.
* Spot audit (my reading, 25 + 25 random passages, so +-20 points): of passages
  dropped for debris/tables, ~18/25 were really garbage or contact/price tables
  and ~7/25 were readable prose -- the filter errs toward dropping.  Of kept
  passages ~5/25 were low-value lists (ToC dot leaders, staff rosters,
  bibliographies, contact tables).  **Not caught:** damage that stays inside
  Devanagari -- doubled letters (`ततथा`, `समययमा`), lost conjuncts (`राम्ो`),
  two-column OCR lines interleaved mid-sentence.  Those pass every statistic here.
* Identical text is deduplicated, but the eight AITC diary editions and the
  quarterly magazines repeat near-identical passages (different years, OCR
  variants).  They are not collapsed, so a top-5 often contains 3-4 copies of the
  same answer.  That inflates hit@5 and wastes result slots.

## Roman Nepali

BGE-M3 has seen Devanagari Nepali and essentially no Roman Nepali.
`translit.py` turns `dhan ma kira lagyo` into `धानमा कीरा लाग्यो`:

1. **Corpus lexicon.**  Every Devanagari word in the indexed passages is folded
   with the project's own `collapse_tight` / `collapse_loose`, so `dhan`, `dhaan`,
   `dhann` meet at the corpus's most frequent surface form `धान`; further fallbacks
   tolerate dropped final `a` (`ra`), e/i and o/u swaps (`bemari`), a dropped
   internal schwa (`upchar` -> `उपचार`) and word-final `-ey`.
2. **Phonetic rules** for words the corpus never saw.

Separate postpositions are glued back on (`dhan ma` -> `धानमा`), because that is how
the embedding model has seen them.  It does not edit `agrichat/nepali_text.py`;
it imports `normalize_roman`, `collapse_tight`, `collapse_loose`, `drop_weak_nasal`.
Known weak spots: English loan words spelled phonetically (`compost` ->
`चोम्पोस्त`, corpus has `कम्पोष्ट`), `pani` (water vs "also", forced to
`पानी`), names.

## Measurement

40 information needs (crops, pests, diseases, soil, livestock, policy), each
written three ways **by me**: Devanagari; Roman Nepali the way a careful typist
writes it; and a "sloppy" Roman variant (`fadke keera`, `xa`, `k`, `jaalo`, long
vowels, dropped spaces).  Same question in every script, so a script effect is not
a topic effect.  Relevance was judged by me by pooling: the top 5 of every system
and every query variant (plus six passages containing all key terms) were shown
blind to which system returned them, and I marked which ones answer the question
(`judgements.json`; rubric inside).  A retrieved passage that I did not judge would
count as non-relevant; there were none.  Passages that *no* system returned cannot
be known, which bounds recall for everyone equally (one case I found by grep and
added: q38, see its note).  39/40 questions have at least one relevant passage.

hit@5 (%), 39 answerable questions (`results.txt` has everything):

| query | dense | BM25 (stems) | TF-IDF (engine style) | RRF |
|---|---:|---:|---:|---:|
| Devanagari | **87.2** | 74.4 | 74.4 | 79.5 |
| Roman, as typed, no translit | 35.9 | 74.4 | 74.4 | 64.1 |
| Roman -> Devanagari (translit) | **89.7** | 74.4 | 76.9 | 82.1 |
| sloppy Roman, as typed | 33.3 | 64.1 | 76.9 | 71.8 |
| sloppy Roman -> Devanagari | **89.7** | 66.7 | 79.5 | 87.2 |

hit@1 (%): dense 61.5 (Devanagari) vs BM25 33.3, TF-IDF 43.6, RRF 66.7.

* Transliteration is worth **+53.8 points** of dense hit@5 (95 % CI +38 to +69;
  +43.6 on hit@1) and +17.9 on RRF.  With it, Roman is within 2.6 points of
  Devanagari (one question), even for sloppy spelling (+0.0 against careful Roman).
  Without it, Roman dense retrieval is worse than useless next to lexical search.
  Dense without translit still gets ~35 % only because many queries contain
  English/Latin-script terms the model knows (`rabies`, `kiwi`, `citrus greening`).
* The lexical half does not care about script: the collapse keys make Devanagari and
  Roman hit@5 identical (74.4 / 74.4; hit@1 differs, 33 vs 44, which is noise-sized).  Transliterating the lexical query too changes
  nothing, so `search.py` gives the lexical half the raw query.
* **Dense beats the existing lexical approach**, but the evidence is thin on
  hit@5 (+12.8 points, CI -2.6 to +28.2) and clear on hit@1 (+28.2, CI +10 to +46).
  Forty questions cannot separate them at hit@5.
* **RRF did not beat dense alone** on hit@5 (79.5 vs 87.2, CI -23 to +8) and is level
  on hit@1.  It beats lexical-only almost everywhere but never beats
  dense-with-translit (sloppy Roman: 87.2 vs 89.7).  On
  this corpus the BM25 half adds noise from OCR-damaged tokens as often as it adds
  exact-match rescues.  RRF is still the safer default where query terms are
  brand names or numbers (not tested here).

Caveats: one judge (me, who also built the system); 40 questions; the near-duplicate
diaries make hit@5 easy; queries are my phrasing, not farmers'; passages were
judged on their first 200-380 characters.  Read the CIs, not
the point estimates.

## Model and cost

* `BAAI/bge-m3` (MIT), `pytorch_model.bin` 2.27 GB; 2.2 GB in `~/.cache/huggingface`.
  (torch CPU wheel + transformers add ~0.6 GB of pip cache.)
* Build: 1.3 min chunk + filter + lexical keys, then 45.5 min of embedding for 16,272
  passages (5-7.5 passages/s, 4 cores, bf16 autocast -- AVX512-bf16/AMX CPU; fp32 is
  roughly 4x slower; bf16 vs fp32 cosine 0.9999).  Average passage is ~290 tokens;
  a few exceed 512 and are truncated.
* Index: dense matrix 32 MB (fp16), passages 47 MB, lexical keys 26 MB, lexicon 6 MB.
  Plain numpy cosine over 16k x 1024 is instant; FAISS is not needed.
* Query: ~0.2 s to embed a query on CPU; BM25 ~5 ms.
