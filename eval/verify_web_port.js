/*
 * Check that the browser engine agrees with the Python one.
 *
 * The web build reimplements query collapsing and TF-IDF scoring in
 * JavaScript.  Two implementations of the same thing drift, so this replays
 * the whole eval set through the JS engine and compares, case by case,
 * against Python's ranking.
 *
 * Run:
 *     python eval/evaluate.py --dump /tmp/py_ranking.json
 *     python eval/test_diagnostic.py --dump /tmp/py_diag.json
 *     node eval/verify_web_port.js /tmp/py_ranking.json /tmp/py_diag.json
 *
 * Exits non-zero if the two disagree on any top-1 intent, or if the JS
 * accuracy comes out below Python's.
 */
"use strict";

const fs = require("fs");
const path = require("path");

const ROOT = path.dirname(__dirname);
require(path.join(ROOT, "web", "kb-bundle.js"));
const AgriChat = require(path.join(ROOT, "web", "agri.js"));

const SCORE_TOLERANCE = 1e-6;

function main() {
  const referencePath = process.argv[2];
  if (!referencePath) {
    console.error("usage: node eval/verify_web_port.js <python-ranking.json>");
    return 2;
  }
  const reference = JSON.parse(fs.readFileSync(referencePath, "utf8"));
  const engine = new AgriChat.Engine(globalThis.AGRI_KB);
  const cfg = engine.config;

  let top1Mismatch = [];
  let scoreDrift = [];
  let jsTop1 = 0, pyTop1 = 0, jsConfidentWrong = 0, inScope = 0;
  let maxDelta = 0;

  for (const ref of reference) {
    const got = engine.answer(ref.q);
    const gotIds = got.ranked.slice(0, 3).map((r) => r.entry.id);

    if (gotIds[0] !== ref.ids[0]) {
      top1Mismatch.push({ q: ref.q, python: ref.ids[0], js: gotIds[0] });
    }
    const delta = Math.abs(got.score - ref.score);
    maxDelta = Math.max(maxDelta, delta);
    if (delta > SCORE_TOLERANCE) {
      scoreDrift.push({ q: ref.q, python: ref.score, js: got.score, delta });
    }

    if (ref.expected) {
      inScope++;
      if (gotIds[0] === ref.expected) jsTop1++;
      if (ref.ids[0] === ref.expected) pyTop1++;
      const confident = got.score >= cfg.minScore && got.margin >= cfg.minMargin;
      if (confident && gotIds[0] !== ref.expected) jsConfidentWrong++;
    }
  }

  const pct = (n) => ((100 * n) / inScope).toFixed(1) + "%";
  console.log(`cases                ${reference.length} (${inScope} in scope)`);
  console.log(`top-1 agreement      ${reference.length - top1Mismatch.length}/${reference.length}`);
  console.log(`max score delta      ${maxDelta.toExponential(2)}`);
  console.log(`python top-1         ${pct(pyTop1)}`);
  console.log(`js top-1             ${pct(jsTop1)}`);
  console.log(`js confident-wrong   ${pct(jsConfidentWrong)}`);

  let failed = false;
  if (top1Mismatch.length) {
    failed = true;
    console.log(`\nFAIL: ${top1Mismatch.length} top-1 disagreement(s)`);
    for (const m of top1Mismatch.slice(0, 20)) {
      console.log(`  ${m.q}\n    python ${m.python}\n    js     ${m.js}`);
    }
  }
  if (scoreDrift.length) {
    console.log(`\nnote: ${scoreDrift.length} case(s) differ in score beyond ${SCORE_TOLERANCE}`);
    for (const d of scoreDrift.slice(0, 5)) {
      console.log(`  ${d.q}: python ${d.python.toFixed(6)} js ${d.js.toFixed(6)}`);
    }
  }
  if (jsTop1 < pyTop1) {
    failed = true;
    console.log(`\nFAIL: js accuracy (${pct(jsTop1)}) is below python (${pct(pyTop1)})`);
  }
  // ---- diagnostic transcripts -------------------------------------------

  const diagPath = process.argv[3];
  if (diagPath) {
    const transcripts = JSON.parse(fs.readFileSync(diagPath, "utf8"));
    const convoMismatch = [];
    for (const ref of transcripts) {
      const convo = new AgriChat.Conversation(engine);
      let questions = 0, intent = null, reply = null;
      for (const turn of ref.turns) {
        reply = convo.send(turn);
        if (reply.kind === "answer") { intent = reply.intent; break; }
        questions++;
      }
      if (intent !== ref.intent || questions !== ref.questions) {
        convoMismatch.push({ label: ref.label, py: [ref.intent, ref.questions],
                             js: [intent, questions] });
      }
    }
    console.log(`\ndiagnostic transcripts ${transcripts.length - convoMismatch.length}/${transcripts.length} match`);
    if (convoMismatch.length) {
      failed = true;
      console.log(`FAIL: ${convoMismatch.length} diagnostic disagreement(s)`);
      for (const m of convoMismatch) {
        console.log(`  ${m.label}\n    python ${JSON.stringify(m.py)}\n    js     ${JSON.stringify(m.js)}`);
      }
    }
  }

  if (!failed) console.log("\nweb port matches the Python engine");
  return failed ? 1 : 0;
}

process.exit(main());
