/*
 * Browser port of the AgriChatBot retrieval engine.
 *
 * The Python package stays the source of truth.  Pattern keys are collapsed
 * by Python and shipped in kb-bundle.js, so the only logic reproduced here is
 * collapsing the *query* plus the TF-IDF scoring -- and
 * eval/verify_web_port.js replays the whole eval set through this file and
 * fails if it disagrees with Python.  Keep the two in step: change
 * agrichat/nepali_text.py, mirror it here, re-run the verifier.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.AgriChat = factory();
})(typeof self !== "undefined" ? self : globalThis, function () {
  "use strict";

  /* ---------------------------------------------------------------- text */

  var CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "w",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "k", "ख़": "kh", "ग़": "g", "ज़": "j", "ड़": "d", "ढ़": "dh", "फ़": "ph"
  };
  var VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऍ": "e", "ऑ": "o"
  };
  var MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॅ": "e", "ॉ": "o"
  };
  var HALANT = "्", ANUSVARA = "ं", CHANDRABINDU = "ँ";
  var VISARGA = "ः", NUKTA = "़", AVAGRAHA = "ऽ";
  var SKIP = {};
  SKIP[CHANDRABINDU] = SKIP[NUKTA] = SKIP[AVAGRAHA] = SKIP["‌"] = SKIP["‍"] = true;
  var DIGITS = {};
  "०१२३४५६७८९".split("").forEach(function (d, i) { DIGITS[d] = String(i); });

  var DEVA = /[ऀ-ॿ]/;

  function normalizeUnicode(text) {
    text = (text || "").normalize("NFC");
    return text.replace(/[​‌‍﻿]/g, "");
  }
  function hasDevanagari(text) { return DEVA.test(text || ""); }

  function devaToRoman(text) {
    text = normalizeUnicode(text);
    var chars = Array.from(text), out = [], n = chars.length;
    for (var i = 0; i < n; i++) {
      var ch = chars[i];
      if (SKIP[ch]) continue;
      if (CONSONANTS[ch]) {
        out.push(CONSONANTS[ch]);
        var nxt = i + 1 < n ? chars[i + 1] : "";
        if (MATRAS[nxt] || nxt === HALANT) continue;
        if (nxt === "" || !DEVA.test(nxt)) continue;   // word-final schwa
        if (nxt === ANUSVARA || nxt === VISARGA || nxt === CHANDRABINDU) continue;
        out.push("a");
      } else if (MATRAS[ch]) out.push(MATRAS[ch]);
      else if (VOWELS[ch]) out.push(VOWELS[ch]);
      else if (ch === ANUSVARA) out.push("n");
      else if (ch === VISARGA) out.push("h");
      else if (ch === HALANT) continue;
      else if (DIGITS[ch]) out.push(DIGITS[ch]);
      else out.push(ch);
    }
    return out.join("");
  }

  var ROMAN_FIXES = [
    [/chhh+/g, "chh"], [/x/g, "chh"], [/z/g, "j"], [/f/g, "ph"],
    [/v/g, "w"], [/q/g, "k"], [/c(?!h)/g, "ch"], [/ck/g, "k"], [/sch/g, "sh"],
    [/\by\b/g, "yo"], [/\bk\b/g, "ke"], [/\bm\b/g, "ma"], [/\bn\b/g, "na"],
    [/\bh\b/g, "ho"], [/\bcha\b/g, "chha"], [/\bchan\b/g, "chhan"], [/\bth\b/g, "the"]
  ];
  function normalizeRoman(text) {
    text = normalizeUnicode(text).toLowerCase().replace(/['`’]/g, "");
    for (var i = 0; i < ROMAN_FIXES.length; i++)
      text = text.replace(ROMAN_FIXES[i][0], ROMAN_FIXES[i][1]);
    return text;
  }

  var TIGHT_RULES = [
    [/aa+/g, "a"], [/ee|ii|y(?=[aeiou])|i+/g, "i"], [/oo|uu|u+/g, "u"],
    [/ay\b/g, "e"], [/ai|ei/g, "e"], [/au|ou/g, "o"],
    [/sh|ss/g, "s"], [/ng|ny|nn/g, "n"], [/w/g, "b"],
    [/([kgcjtdpbmnrlsh])\1+/g, "$1"]
  ];
  var LOOSE_RULES = [
    [/chh|ch/g, "c"], [/kh/g, "k"], [/gh/g, "g"], [/jh/g, "j"],
    [/th/g, "t"], [/dh/g, "d"], [/ph/g, "p"], [/bh/g, "b"]
  ];

  function toRoman(text) {
    text = normalizeUnicode(text);
    if (!hasDevanagari(text)) return normalizeRoman(text);
    var parts = text.match(/[ऀ-ॿ]+|[^ऀ-ॿ]+/g) || [];
    return normalizeRoman(parts.map(function (tok) {
      return hasDevanagari(tok) ? devaToRoman(tok) : normalizeRoman(tok);
    }).join(""));
  }
  function clean(text) {
    return text.replace(/[^a-z0-9\s]+/g, " ").replace(/\s+/g, " ").trim();
  }
  function apply(text, rules) {
    for (var i = 0; i < rules.length; i++) text = text.replace(rules[i][0], rules[i][1]);
    return text;
  }
  function collapseTight(text) { return clean(apply(toRoman(text), TIGHT_RULES)); }
  function collapseLoose(text) {
    var r = apply(apply(toRoman(text), LOOSE_RULES), TIGHT_RULES);
    return clean(r.replace(/([kgcjtdpbmnrlsh])\1+/g, "$1"));
  }

  /* ---------------------------------------------------------- morphology */

  var SUFFIXES = ["harulai", "haruko", "haruka", "haruki", "harule", "haruma",
    "harusanga", "haru", "haru", "bhanda", "bhandapani", "dekhi", "samma",
    "sanga", "sangai", "bata", "madhye", "pachhi", "aghi", "tira", "jasto",
    "jastai", "lai", "ko", "ka", "ki", "le", "ma", "mai", "kai", "sit", "siti"];
  var VERB_SUFFIXES = ["chhau", "chhan", "chhin", "chhu", "chha", "chh",
    "cau", "can", "cin", "cu", "ca", "c", "yeko", "eko", "eka", "eki", "ieko",
    "ne", "nu", "na", "yo", "e"];
  var RISKY = set(["na", "ne", "nu", "yo", "e"]);
  var MIN_STEM = 3, MIN_STEM_RISKY = 4;
  var PROTECTED = set(["kahile", "kasari", "kati", "kaha", "kahan", "kun",
    "kina", "kasto", "kata", "kaho", "kehi", "kahi"]);
  var BARE_POSTPOSITIONS = set(["ko", "ka", "ki", "le", "lai", "bata", "sanga",
    "dekhi", "samma", "haru", "madhye", "bhanda", "tira", "sit"]);
  var STOPWORDS = set(["ra", "ani", "tara", "pani", "ho", "hoina", "hos",
    "huns", "hun", "hunch", "yo", "tyo", "yi", "ti", "yas", "tyas", "yasto",
    "tyasto", "mero", "hamro", "tapai", "timro", "malai", "hamilai", "tapaile",
    "ma", "mai", "cha", "ca", "chha", "bhaneko", "bhanne", "arko", "aru",
    "sakinch", "sakch", "parch", "garch", "mailey", "la", "ni", "nai",
    "ch", "chh", "c", "chan", "can", "chhan"]);

  function set(list) {
    var o = Object.create(null);
    list.forEach(function (k) { o[k] = true; });
    return o;
  }
  function endsWith(s, suf) { return s.length >= suf.length && s.slice(s.length - suf.length) === suf; }

  function stripSuffixes(word) {
    if (PROTECTED[word]) return word;
    var strippedPostposition = false, changed = true, i;
    while (changed) {
      changed = false;
      for (i = 0; i < SUFFIXES.length; i++) {
        var suf = SUFFIXES[i];
        if (endsWith(word, suf) && word.length - suf.length >= MIN_STEM) {
          word = word.slice(0, word.length - suf.length);
          changed = true; strippedPostposition = true;
          break;
        }
      }
    }
    if (!strippedPostposition) {
      for (i = 0; i < VERB_SUFFIXES.length; i++) {
        var v = VERB_SUFFIXES[i];
        var floor = RISKY[v] ? MIN_STEM_RISKY : MIN_STEM;
        if (endsWith(word, v) && word.length - v.length >= floor) {
          word = word.slice(0, word.length - v.length);
          break;
        }
      }
    }
    if (word.length > MIN_STEM && endsWith(word, "a")) word = word.slice(0, -1);
    return word;
  }

  var WEAK_NASAL = /n(?=[kgcjtdpbmrlsh])/g;
  function dropWeakNasal(text) { return text.replace(WEAK_NASAL, ""); }

  function tokenize(text) {
    return dropWeakNasal(collapseTight(text)).split(" ").filter(Boolean);
  }
  function stems(text) {
    var out = [];
    tokenize(text).forEach(function (token) {
      if (STOPWORDS[token] || BARE_POSTPOSITIONS[token]) return;
      var stem = stripSuffixes(token);
      if (stem.length < 2 || STOPWORDS[stem] || BARE_POSTPOSITIONS[stem]) return;
      out.push(stem);
    });
    return out;
  }
  function flat(text) { return text.replace(/ /g, ""); }

  function keys(text) {
    return {
      tight: collapseTight(text),
      loose: dropWeakNasal(collapseLoose(text)),
      stems: stems(text).join(" ")
    };
  }

  /* -------------------------------------------------------------- tf-idf */

  // Mirrors sklearn's TfidfVectorizer: sublinear tf, smooth idf, L2 norm.
  function charNgrams(doc, minN, maxN) {
    doc = doc.replace(/\s+/g, " ");
    var grams = [], len = doc.length, hi = Math.min(maxN, len);
    for (var n = minN; n <= hi; n++)
      for (var i = 0; i + n <= len; i++) grams.push(doc.slice(i, i + n));
    return grams;
  }
  function wordNgrams(doc, minN, maxN) {
    var tokens = doc.match(/\b\w\w+\b/g) || [];   // sklearn's default token_pattern
    var grams = tokens.slice(), hi = Math.min(maxN, tokens.length);
    for (var n = Math.max(minN, 2); n <= hi; n++)
      for (var i = 0; i + n <= tokens.length; i++) grams.push(tokens.slice(i, i + n).join(" "));
    return grams;
  }
  function counts(grams) {
    var c = Object.create(null);
    for (var i = 0; i < grams.length; i++) c[grams[i]] = (c[grams[i]] || 0) + 1;
    return c;
  }

  function Vectorizer(analyzer) {
    this.analyzer = analyzer;
    this.idf = Object.create(null);
    this.docs = [];
  }
  Vectorizer.prototype.fit = function (documents) {
    var df = Object.create(null), n = documents.length, i, g;
    var allCounts = documents.map(function (d) { return counts(this.analyzer(d)); }, this);
    for (i = 0; i < allCounts.length; i++)
      for (g in allCounts[i]) df[g] = (df[g] || 0) + 1;
    for (g in df) this.idf[g] = Math.log((1 + n) / (1 + df[g])) + 1;
    this.docs = allCounts.map(this._vector, this);
    return this;
  };
  Vectorizer.prototype._vector = function (c) {
    var vec = Object.create(null), norm = 0, g, v;
    for (g in c) {
      if (this.idf[g] === undefined) continue;          // unseen at fit time
      v = (1 + Math.log(c[g])) * this.idf[g];           // sublinear tf
      vec[g] = v; norm += v * v;
    }
    norm = Math.sqrt(norm);
    if (norm > 0) for (g in vec) vec[g] /= norm;
    return vec;
  };
  Vectorizer.prototype.transform = function (doc) {
    return this._vector(counts(this.analyzer(doc)));
  };
  Vectorizer.prototype.similarities = function (queryVec, out) {
    // rows are L2-normalized, so the dot product is the cosine
    for (var i = 0; i < this.docs.length; i++) {
      var doc = this.docs[i], s = 0, g;
      for (g in queryVec) if (doc[g] !== undefined) s += queryVec[g] * doc[g];
      out[i] = s;
    }
    return out;
  };

  /* -------------------------------------------------------------- engine */

  function Engine(bundle) {
    this.bundle = bundle;
    this.entries = bundle.entries;
    this.config = bundle.config;
    this.byId = {};
    this.entries.forEach(function (e) { this.byId[e.id] = e; }, this);

    var tight = [], loose = [], stemDocs = [];
    this.owner = [];
    this.entries.forEach(function (entry, idx) {
      entry.keys.forEach(function (k) {
        this.owner.push(idx);
        tight.push(k[0]); loose.push(k[1]); stemDocs.push(k[2]);
      }, this);
    }, this);

    var charAnalyzer = function (d) { return charNgrams(d, 3, 5); };
    var wordAnalyzer = function (d) { return wordNgrams(d, 1, 2); };
    this.vec = {
      tight: new Vectorizer(charAnalyzer).fit(tight),
      loose: new Vectorizer(charAnalyzer).fit(loose),
      stems: new Vectorizer(wordAnalyzer).fit(stemDocs)
    };
    this._scratch = new Float64Array(tight.length);
    this._patternScores = new Float64Array(tight.length);
  }

  // Matching is per token, not a substring search over the whole utterance:
  // भात (rice) collapses to "bat", which sits inside "battery", and a raw
  // substring search read "phone ko battery kasto hunxa" as a question about
  // धान. Prefix matching is allowed only for Devanagari tokens, where
  // morphemes are glued together (धानबाली, धानमा).
  Engine.prototype.detectCrops = function (text) {
    var exact = Object.create(null);
    tokenize(text).forEach(function (t) { exact[flat(keys(t).loose)] = true; });
    stems(text).forEach(function (t) { exact[flat(keys(t).loose)] = true; });
    delete exact[""];

    var devaTokens = [];
    ((text || "").match(/[\u0900-\u097F]+/g) || []).forEach(function (run) {
      devaTokens.push(flat(keys(run).loose));
      stems(run).forEach(function (t) { devaTokens.push(flat(keys(t).loose)); });
    });
    devaTokens = devaTokens.filter(Boolean);

    var found = [];
    var aliases = this.bundle.cropAliases;
    for (var crop in aliases) {
      var list = aliases[crop];
      for (var i = 0; i < list.length; i++) {
        var alias = list[i];
        if (alias.length < 3) continue;
        var hit = exact[alias] || devaTokens.some(function (t) { return t.indexOf(alias) === 0; });
        if (hit) { found.push(crop); break; }
      }
    }
    return found;
  };

  // Short queries inflate cosine similarity -- "kun tarkari" scored 0.758 on
  // two words. A question carrying almost no content has not really been
  // asked, whatever it scores.
  Engine.prototype.contentStems = function (text) {
    return stems(text).filter(function (s) { return !PROTECTED[s]; });
  };

  Engine.prototype.rank = function (text) {
    var k = keys(text);
    var query = { tight: flat(k.tight), loose: flat(k.loose), stems: k.stems };
    var total = this._patternScores, scratch = this._scratch, i;
    total.fill(0);
    var weights = this.config.weights;
    for (var name in weights) {
      if (!query[name] || !query[name].trim()) continue;
      this.vec[name].similarities(this.vec[name].transform(query[name]), scratch);
      var w = weights[name];
      for (i = 0; i < total.length; i++) total[i] += w * scratch[i];
    }

    var perIntent = new Float64Array(this.entries.length);
    for (i = 0; i < total.length; i++) {
      var idx = this.owner[i];
      if (total[i] > perIntent[idx]) perIntent[idx] = total[i];
    }

    var crops = this.detectCrops(text);
    if (crops.length) {
      var cropSet = set(crops);
      for (i = 0; i < this.entries.length; i++) {
        var entryCrops = this.entries[i].crops;
        if (!entryCrops.length) continue;
        var overlap = entryCrops.some(function (c) { return cropSet[c]; });
        perIntent[i] += overlap ? this.config.cropBonus : -this.config.cropPenalty;
      }
    }

    var ranked = [];
    for (i = 0; i < this.entries.length; i++)
      ranked.push({ entry: this.entries[i], score: perIntent[i] });
    ranked.sort(function (a, b) { return b.score - a.score; });
    return ranked;
  };

  Engine.prototype.answer = function (text) {
    var ranked = this.rank(text);
    var top = ranked[0];
    var margin = top.score - (ranked.length > 1 ? ranked[1].score : 0);
    var minMargin = this.config.minMargin;
    if (this.contentStems(text).length < (this.config.minContentStems || 0)) {
      minMargin = Math.max(minMargin, this.config.shortQueryMargin || 0);
    }
    return {
      entry: top.entry,
      score: top.score,
      margin: margin,
      confident: top.score >= this.config.minScore && margin >= minMargin,
      suggestions: ranked.slice(0, 3).filter(function (r) { return r.score > 0.12; })
        .map(function (r) { return r.entry; }),
      crops: this.detectCrops(text),
      ranked: ranked
    };
  };


  /* ---------------------------------------------------------- diagnostic */

  // Mirror of agrichat/diagnostic.py and agrichat/conversation.py.  Keep the
  // two in step; the rules themselves live in the bundle, so only this logic
  // is duplicated.
  var WILDCARD = "*";
  var SLOT_ORDER = ["crop", "part", "symptom"];
  var QUESTION_WORDS = ["कति", "कसरी", "कहिले", "कहाँ", "किन", "कुन", "के गर्ने",
                        "kati", "kasari", "kahile", "kaha", "kina", "kun"];

  function Diagnostic(engine, spec) {
    this.engine = engine;
    this.spec = spec;
    this.rules = spec.rules;

    this.alias = {};
    Object.keys(spec.slots).forEach(function (slot) {
      var table = Object.create(null);
      spec.slots[slot].options.forEach(function (option) {
        option.aliases.forEach(function (a) {
          var k = flat(collapseLoose(a));
          if (k) table[k] = option.value;
        });
      });
      this.alias[slot] = table;
    }, this);

    this.triggers = spec.triggers
      .map(function (t) { return flat(collapseLoose(t)); })
      .filter(Boolean);
  }

  Diagnostic.prototype.looksLikeSymptom = function (text) {
    var blob = flat(collapseLoose(text));
    return this.triggers.some(function (t) { return blob.indexOf(t) !== -1; });
  };

  Diagnostic.prototype.extract = function (text, into) {
    var slots = Object.assign({}, into || {});
    var blob = flat(collapseLoose(text));

    if (slots.crop === undefined) {
      var crops = this.engine.detectCrops(text);
      if (crops.length) slots.crop = crops.slice().sort()[0];
    }
    ["part", "symptom"].forEach(function (slot) {
      if (slots[slot] !== undefined) return;
      var best = null, table = this.alias[slot];
      for (var key in table) {
        if (key && blob.indexOf(key) !== -1) {
          if (best === null || key.length > best[0].length) best = [key, table[key]];
        }
      }
      if (best) slots[slot] = best[1];
    }, this);
    return slots;
  };

  Diagnostic.prototype.matching = function (slots) {
    var out = this.rules.filter(function (rule) {
      return SLOT_ORDER.every(function (slot) {
        var known = slots[slot], want = rule[slot] === undefined ? WILDCARD : rule[slot];
        return known === undefined || want === WILDCARD || want === known;
      });
    });
    // crop-specific knowledge beats the generic fallbacks
    if (slots.crop) {
      var specific = out.filter(function (r) { return (r.crop || WILDCARD) !== WILDCARD; });
      if (specific.length) return specific;
    }
    return out;
  };

  Diagnostic.prototype.resolve = function (slots) {
    var candidates = this.matching(slots).slice();
    if (!candidates.length) return null;
    var spec = function (rule) {
      return SLOT_ORDER.filter(function (s) {
        return (rule[s] === undefined ? WILDCARD : rule[s]) !== WILDCARD;
      }).length;
    };
    candidates.sort(function (a, b) { return spec(b) - spec(a); });
    return candidates[0].intent;
  };

  Diagnostic.prototype.nextQuestion = function (slots) {
    var candidates = this.matching(slots);
    if (!candidates.length) return null;
    var outcomes = {};
    candidates.forEach(function (r) { outcomes[r.intent] = true; });
    if (Object.keys(outcomes).length <= 1) return null;   // asking cannot help

    for (var i = 0; i < SLOT_ORDER.length; i++) {
      var slot = SLOT_ORDER[i];
      if (slots[slot] !== undefined) continue;
      var values = {};
      candidates.forEach(function (r) { values[r[slot] === undefined ? WILDCARD : r[slot]] = true; });
      var keys_ = Object.keys(values);
      if (keys_.length === 1 && keys_[0] === WILDCARD) continue;
      if (slot === "crop") {
        return { slot: "crop", text: this.spec.crop_question, options: this.spec.crop_examples };
      }
      var cfg = this.spec.slots[slot];
      var live = {};
      candidates.forEach(function (r) { live[r[slot] === undefined ? WILDCARD : r[slot]] = true; });
      var options = cfg.options.filter(function (o) {
        return live[o.value] || live[WILDCARD];
      }).map(function (o) { return o.label; });
      return { slot: slot, text: cfg.question, options: options };
    }
    return null;
  };

  function Conversation(engine, diagnostic) {
    this.engine = engine;
    this.diagnostic = diagnostic || new Diagnostic(engine, engine.bundle.diagnostic);
    this.reset();
  }
  Conversation.prototype.reset = function () {
    this.slots = {};
    this.pending = null;
    this.asked = [];
  };

  Conversation.prototype.send = function (text) {
    text = (text || "").trim();
    if (!text) return { kind: "unknown", text: "कृपया आफ्नो प्रश्न लेख्नुहोस्।", options: [], slots: {} };

    if (this.pending !== null) {
      var cont = this._continue(text);
      if (cont !== null) return cont;
      this.reset();
    }

    var result = this.engine.answer(text);
    var isSymptom = this.diagnostic.looksLikeSymptom(text);
    var slots = isSymptom ? this.diagnostic.extract(text) : {};
    var underspecified = isSymptom && slots.symptom === undefined;

    if (result.confident && !underspecified) {
      this.reset();
      return { kind: "answer", text: this._answerText(result.entry), intent: result.entry.id,
               options: [], slots: {}, score: result.score, confident: true, result: result };
    }
    if (isSymptom) {
      this.slots = slots;
      return this._advance(result);
    }
    return { kind: "clarify", text: null, options: [], slots: {}, result: result };
  };

  Conversation.prototype._continue = function (text) {
    var self = this;
    if (QUESTION_WORDS.some(function (w) { return text.indexOf(w) !== -1; })
        && this.engine.answer(text).confident) return null;

    var before = JSON.stringify(this.slots);
    this.slots = this.diagnostic.extract(text, this.slots);
    if (JSON.stringify(this.slots) === before) {
      if (this.engine.answer(text).confident || !this.diagnostic.looksLikeSymptom(text)) return null;
      return { kind: "question", text: "माफ गर्नुहोस्, बुझिनँ। " + this.pending.text,
               options: this.pending.options, slots: Object.assign({}, this.slots) };
    }
    return this._advance(null);
  };

  Conversation.prototype._advance = function (result) {
    var question = this.diagnostic.nextQuestion(this.slots);
    if (question !== null && this.asked.indexOf(question.slot) === -1) {
      this.pending = question;
      this.asked.push(question.slot);
      return { kind: "question", text: question.text, options: question.options,
               slots: Object.assign({}, this.slots) };
    }
    var intentId = this.diagnostic.resolve(this.slots);
    var entry = intentId ? this.engine.byId[intentId] : null;
    if (!entry) {
      this.reset();
      if (result) return { kind: "clarify", text: null, options: [], slots: {}, result: result };
      return { kind: "unknown", options: [], slots: {},
               text: "माफ गर्नुहोस्, यो समस्या पहिचान गर्न सकिनँ। नजिकको कृषि ज्ञान केन्द्रमा बोट वा पात देखाउनुहोस्।" };
    }
    var slots = Object.assign({}, this.slots);
    this.reset();
    return { kind: "answer", text: this._answerText(entry, slots), intent: entry.id,
             options: [], slots: slots, confident: true };
  };

  Conversation.prototype._answerText = function (entry, slots) {
    var out = "";
    if (slots) {
      var said = [slots.crop, slots.part, slots.symptom].filter(Boolean).join(" · ");
      if (said) out += "[" + said + "]\n\n";
    }
    out += entry.answer;
    var follow = (entry.followups || []).slice(0, 3)
      .map(function (id) { return this.engine.byId[id]; }, this).filter(Boolean);
    if (follow.length) {
      out += "\n\nसम्बन्धित विषय:\n" + follow.map(function (e) { return "  • " + e.label; }).join("\n");
    }
    return out;
  };

  return {
    Engine: Engine,
    Diagnostic: Diagnostic,
    Conversation: Conversation,
    keys: keys, stems: stems, flat: flat,
    collapseTight: collapseTight, collapseLoose: collapseLoose,
    dropWeakNasal: dropWeakNasal, scriptHasDevanagari: hasDevanagari
  };
});
