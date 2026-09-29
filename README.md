# AgriChatBot · कृषि सहयोगी च्याटबोट

A Nepali-language question-answering assistant for farmers in Nepal.

It answers in Nepali, and it understands questions in **either script** —
Devanagari or Roman Nepali — including the inconsistent spellings people
actually type:

```
तपाईं: धानमा मरुवा रोग लाग्यो, के गर्ने?
तपाईं: dhan ma maruwa rog lagyo k garne
तपाईं: dhaan ma maruva rog lageko cha
```

All three reach the same answer.

## Quick start

```bash
pip install -r requirements.txt

python chatbot.py                                  # interactive
python chatbot.py "bakhra lai kun khop lagaune"    # one question
python chatbot.py --debug "आलुमा कीरा लाग्यो"       # show match scores
```

No training step. The index is built from `data/kb/*.json` at startup in
well under a second, so editing the knowledge base takes effect on the next
run.

## What it knows

45 intents across the topics Nepali farmers actually ask about:

| Area | Covered |
|---|---|
| अन्नबाली | धान, मकै, गहुँ, कोदो, फापर — समय, बीउ दर, मल मात्रा, रोग, कीरा |
| तरकारी | आलु, गोलभेँडा, काउली, बन्दा, खुर्सानी, प्याज, लसुन, बेर्ना, टनेल खेती |
| नगदे बाली | अदुवा, बेसार, अलैंची, केरा |
| माटो र मल | माटो परीक्षण, कम्पोस्ट, जीवामृत, रासायनिक मल, बाली फेरबदल |
| सिँचाइ | थोपा सिँचाइ, मल्चिङ |
| पशुपन्छी | बाख्रा, गाई–भैंसी, कुखुरा, माछा, मौरी, घाँस बाली |
| नीति र बजार | कृषि अनुदान, बाली बीमा, कृषि ज्ञान केन्द्र, बजार मूल्य, भण्डारण |

Fertiliser rates, spacing, vaccination schedules and subsidy procedures are
grounded in Nepali extension sources (MoALD, PMAMP, NARC reporting, provincial
agriculture directorates); each entry carries its `sources`.

## Accuracy

Measured on `eval/testset.json` — 149 held-out queries in both scripts, none
copied verbatim from the knowledge base, including 12 out-of-scope questions:

```bash
python eval/evaluate.py            # report
python eval/evaluate.py --errors   # every mistake, with scores
python eval/evaluate.py --tune     # grid-search the thresholds
```

The metric that matters is **confident-wrong**: how often the bot hands a
farmer a confident answer to a question they did not ask. A low-confidence
miss becomes "did you mean one of these?", which costs the user a line; a
confident miss can cost them a season. The engine therefore refuses to answer
unless both the absolute score and the margin over the runner-up clear a bar.

Current numbers:

```
top-1 accuracy        91.2%   correct intent ranked first
top-3 accuracy        96.4%   correct intent in the top 3
coverage              89.1%   answered confidently
answer precision      96.7%   of those, correct
CONFIDENT-WRONG        2.9%   <-- the number that hurts farmers
out-of-scope refused  91.7%   junk questions correctly declined
```

> **Read these as a regression guard, not a field accuracy claim.** The test
> set was written by the same author as the knowledge base, and `--tune`
> selects thresholds against this same set, so both the phrasing and the
> settings are optimistic. Real farmer phrasing will be messier. Before
> deploying, collect actual farmer questions and re-measure on those.

## How it works

The hard part is that Roman Nepali has no standard orthography — छ appears as
`cha`, `chha`, `chh` or `xa` — and Devanagari glues postpositions onto nouns
(`धानमा`) where Roman typists split them (`dhan ma`). So both scripts are
pushed into one comparable space before anything else happens:

```
Devanagari ──┐                      ┌── collapse_tight   precision key
             ├── detailed roman ────┤
Roman ───────┘                      └── collapse_loose   recall key
                                     └── stems           content words
```

`agrichat/nepali_text.py` does the transliteration, the lossy phonetic
collapsing (श/ष/स → s, ब/व → b, retroflex → dental, aspiration dropped in the
recall key) and a Nepali suffix stemmer that peels off `-मा -को -लाई -हरू` and
verb endings — which is what makes `धानमा` and `dhan ma` meet at `dhan`.

`agrichat/engine.py` indexes all three views with TF-IDF (character n-grams on
the two phonetic keys, word n-grams on the stems), scores a query against every
pattern, and max-pools per intent. On top of that:

- **Crop-entity matching.** "धानमा कीरा" and "आलुमा कीरा" are lexically almost
  identical; only the crop word separates them. Naming a crop boosts intents
  about that crop and penalises intents about a different one.
- **Explicit refusal.** Below the score or margin bar, the bot lists its best
  guesses instead of asserting one.

## Layout

```
chatbot.py               CLI
agrichat/nepali_text.py  transliteration, phonetic collapsing, stemming
agrichat/kb.py           knowledge-base loading and validation
agrichat/engine.py       retrieval, crop matching, confidence
data/kb/*.json           the knowledge base (edit this to add topics)
eval/testset.json        held-out queries with gold labels
eval/evaluate.py         metrics and threshold tuning
```

### Adding a topic

Append an entry to any file in `data/kb/`:

```json
{
  "id": "unique_snake_case_id",
  "crops": ["धान"],
  "patterns": ["देवनागरीमा प्रश्न", "roman nepali ma prasna"],
  "answer": "नेपालीमा जवाफ",
  "followups": ["another_intent_id"],
  "sources": ["https://..."]
}
```

Give it patterns in **both** scripts — the transliterator handles spelling
variation, but it cannot guess that a farmer says "गाभा मर्‍यो" for stem borer.
`agrichat/kb.py` rejects duplicate ids, empty patterns and dangling
`followups` at load time. Add a few queries to `eval/testset.json` and re-run
the evaluation to confirm the new intent does not cannibalise an existing one.

## Scope and safety

The bot gives general guidance. It deliberately does **not** hand out
pesticide dose-and-spray schedules as if they were prescriptions: entries that
touch chemicals point the farmer at the label rate, the pre-harvest interval,
and the nearest कृषि ज्ञान केन्द्र for confirming the diagnosis first. Wrong
agro-chemical advice is expensive at best and dangerous at worst.

## Note on the previous version

This replaces a Keras bag-of-words classifier (8 English intents,
`nltk.word_tokenize` + `WordNetLemmatizer`). That pipeline could not work for
Nepali: the lemmatizer is English-only, the tokenizer splits Devanagari badly,
and exact word matching cannot survive Nepali morphology or Roman Nepali
spelling variation. The TF-IDF retrieval approach here is both more accurate
on this data and small enough to run without TensorFlow.
