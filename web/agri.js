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

  Engine.prototype.detectCrops = function (text) {
    var blob = flat(keys(text).loose);
    var tokenKeys = Object.create(null);
    stems(text).forEach(function (t) { tokenKeys[flat(keys(t).loose)] = true; });
    var found = [];
    var aliases = this.bundle.cropAliases;
    for (var crop in aliases) {
      var list = aliases[crop];
      for (var i = 0; i < list.length; i++) {
        var alias = list[i];
        if (alias.length < 3) continue;
        if (tokenKeys[alias] || blob.indexOf(alias) !== -1) { found.push(crop); break; }
      }
    }
    return found;
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
    return {
      entry: top.entry,
      score: top.score,
      margin: margin,
      confident: top.score >= this.config.minScore && margin >= this.config.minMargin,
      suggestions: ranked.slice(0, 3).filter(function (r) { return r.score > 0.12; })
        .map(function (r) { return r.entry; }),
      crops: this.detectCrops(text),
      ranked: ranked
    };
  };

  return {
    Engine: Engine,
    keys: keys, stems: stems, flat: flat,
    collapseTight: collapseTight, collapseLoose: collapseLoose,
    dropWeakNasal: dropWeakNasal, scriptHasDevanagari: hasDevanagari
  };
});
