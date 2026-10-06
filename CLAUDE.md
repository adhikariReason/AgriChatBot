# AgriChatBot — working notes

Nepali-language question-answering for farmers in Nepal. Questions arrive in
**Devanagari or Roman Nepali** and must reach the same answer. Read `README.md`
for the architecture; this file is the operational stuff that is easy to get
wrong.

## Run it

```bash
pip install -r requirements.txt
python tools/unpack_corpus.py     # first time only: restores data/corpus/text/
python chatbot.py                 # CLI
python web/server.py              # console on :8000
```

## Two commands to re-run after edits

These are not optional — CI fails on both, and one of them has already caught
a real bug.

| After touching | Run |
|---|---|
| `data/kb/` or `data/diagnostic/` | `python -m agrichat.export_web` |
| `agrichat/` | `node eval/verify_web_port.js` (see below) |

```bash
python eval/evaluate.py --dump /tmp/py.json
python eval/test_diagnostic.py --dump /tmp/py_diag.json
node eval/verify_web_port.js /tmp/py.json /tmp/py_diag.json
```

The web console ships a **JavaScript port** of the engine in `web/agri.js`.
Any change to `agrichat/nepali_text.py`, `engine.py`, `diagnostic.py` or
`conversation.py` must be mirrored there. The verifier replays Python's results
through the port and fails on drift. It caught two engine fixes that had been
left out of the browser build — do not skip it.

## Full check before pushing

```bash
python eval/test_nepali_text.py      # Devanagari / Roman convergence
python eval/test_diagnostic.py       # triage conversations
python eval/evaluate.py --check      # accuracy against regression floors
```

## Things that will bite you

**Do not reach for an English NLP tool.** `nltk.word_tokenize` splits
Devanagari badly and `WordNetLemmatizer` does nothing for Nepali. That is what
the previous version used and why it did not work. The text layer is
hand-written and tested; extend it rather than replacing it.

**Crop aliases match per token, and prefix matching is Devanagari-only.** A
substring search over the whole utterance read `phone ko battery kasto hunxa`
as a question about धान, because भात collapses to `bat`, which sits inside
`battery`. Devanagari glues morphemes (`धानबाली`) so a prefix there is
legitimate; Roman is space-separated and never needs it.

**One intent per question, not per crop.** An intent that bundles season,
spacing, fertiliser and pests will answer *every* question about that crop with
all of it. If an answer needs more than about three section headings, it is
probably two intents.

**Adding patterns has side effects.** Every addition can hijack another
intent's queries. Re-run the evaluation after any knowledge-base edit — a fix
for one query has broken another more than once here.

**Test queries must not duplicate knowledge-base patterns.** A verbatim copy is
a guaranteed hit that inflates every number. `evaluate.py` warns on this; six
had crept in and were rephrased.

**Short queries inflate cosine similarity.** `kun tarkari` — two words —
scored 0.758. Queries with fewer than two content stems face a much higher
margin bar.

## Safety posture — keep this

Chemical advice stays in the hand-curated knowledge base. It points at the
label rate, the pre-harvest interval, and the local कृषि ज्ञान केन्द्र for
confirming the diagnosis, and never reads as a prescription. This is not
excess caution: OCR measurably corrupts exactly the tables and pesticide
trade-name registers in `data/corpus/` where doses live, and Nepal has 24
banned pesticides that older third-party material still recommends. Do not
generate dose advice from the corpus.

`data/corpus/pesticides/banned_pesticides.json` is a **hand transcription**
whose Bikram Sambat dates are unverified. It is a deny-list, not an
allow-list, and not safe as a sole gate.

## Known weakness

Out-of-scope refusal (~81%) is the worst number. Domains that *border*
agriculture slip through: a gold price matches market prices, a bank loan
matches soil testing. There is no "is this about farming at all?" gate, only a
confidence threshold, and a neighbouring domain can clear it honestly. That
gate is the next real piece of work.

## Measurement honesty

The headline numbers are a **regression guard, not a field accuracy claim**:
the same author wrote the knowledge base and most of the test set, and
`--tune` picks thresholds against that same set. Cases marked
`source: real-user` in `eval/testset.json` came from actual use and are worth
more than the rest — two of them found a bug 160 synthetic cases missed. Add
every real reported failure there.

## Not wired in

`retrieval/` is a standalone, measured corpus index (BGE-M3 + BM25 over 16,272
passages). Its finding: transliterating Roman queries to Devanagari before
embedding is worth **+54 points** of dense hit@5, because the embedding model
effectively cannot read Roman Nepali. Integrating it costs ~2.9 GB of
dependencies and 0.2 s/query, and moves answers from vetted text to corpus
passages — see the safety note above. Nothing in `agrichat/` imports it.

## Deploy

`web/` publishes to https://adhikarireason.github.io/AgriChatBot/ from `main`
after tests pass. Off `main`, a manual workflow dispatch deploys; the
`github-pages` environment must allow the branch or the job fails at the gate
with zero steps.
