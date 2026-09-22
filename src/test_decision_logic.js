// src/test_decision_logic.js
//
// Iteration 5 decision-layer tests. Two parts:
//   1. Unit tests: extracts the === DECISION-BEGIN/END === block verbatim from
//      web/index.html and exercises decideVerdict boundaries (threshold edges,
//      priority over uncertainty, malformed inputs).
//   2. Product tests: replays every case in
//      docs/rejection_experiment/product_test_outputs.json (REAL Variant B
//      Keras outputs on real experiment images) through the SAME extracted
//      decideVerdict and asserts the expected states.
//
// Run from the repo root:  node src/test_decision_logic.js
// Exits non-zero on any failure.

'use strict';

const fs = require('fs');
const path = require('path');

const HTML = fs.readFileSync(path.join(__dirname, '..', 'web', 'index.html'), 'utf8');
const PRODUCT_JSON = path.join(__dirname, '..', 'docs', 'rejection_experiment',
                               'product_test_outputs.json');

function extractDecisionBlock() {
  const begin = HTML.indexOf('// === DECISION-BEGIN');
  const end = HTML.indexOf('// === DECISION-END');
  if (begin < 0 || end < 0 || end < begin) {
    throw new Error('DECISION markers not found in web/index.html');
  }
  return HTML.slice(begin, end);
}

function makeDecide() {
  const block = extractDecisionBlock();
  return new Function(block + '\nreturn decideVerdict;')();
}

let passCount = 0;
let failCount = 0;

function check(name, actual, expected) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  if (ok) { passCount++; console.log('PASS ' + name); }
  else {
    failCount++;
    console.log('FAIL ' + name + '  expected=' + JSON.stringify(expected) +
                '  actual=' + JSON.stringify(actual));
  }
}

function expectThrows(name, fn) {
  try {
    fn();
    failCount++;
    console.log('FAIL ' + name + '  (expected TypeError, none thrown)');
  } catch (e) {
    if (e instanceof TypeError) {
      passCount++;
      console.log('PASS ' + name);
    } else {
      failCount++;
      console.log('FAIL ' + name + '  (threw ' + e.constructor.name + ' instead of TypeError)');
    }
  }
}

function runUnitCases(decide) {
  const dominant = [0.98, 0.01, 0.005, 0.005];      // top 0.98, margin 0.97
  const ambiguous = [0.55, 0.25, 0.10, 0.10];       // top < 0.60 -> uncertain
  const weakMargin = [0.55, 0.20, 0.15, 0.10];      // top >= 0.60? no: 0.55 < 0.60
  const marginCase = [0.45, 0.25, 0.20, 0.10];      // top 0.45 < 0.60 -> uncertain

  // --- rejection threshold boundary (explicit, per the iteration brief) ---
  check('unit: reject exactly at 0.8774 -> unsupported',
        decide(dominant, 0.8774).state, 'unsupported');
  check('unit: reject just above 0.8774 -> unsupported',
        decide(dominant, 0.87740001).state, 'unsupported');
  check('unit: reject just below 0.8774 (dominant bins) -> supported',
        decide(dominant, 0.87739999).state, 'supported');

  // --- uncertainty boundaries (existing rule, unchanged semantics) ---
  check('unit: top exactly MIN_TOP, margin exactly MIN_MARGIN -> supported',
        decide([0.60, 0.10, 0.10, 0.10], 0.0).state, 'supported');
  check('unit: top just below MIN_TOP -> uncertain',
        decide([0.5999999, 0.10, 0.20, 0.10], 0.0).state, 'uncertain');
  check('unit: low top AND low margin -> uncertain',
        decide(marginCase, 0.0).state, 'uncertain');
  check('unit: ambiguous example shape -> uncertain',
        decide(ambiguous, 0.0).state, 'uncertain');
  check('unit: reject=0 with weak-margin bins -> uncertain (weakMargin top=0.55)',
        decide(weakMargin, 0.0).state, 'uncertain');

  // --- priority: rejection beats uncertainty ---
  check('unit: ambiguous bins + reject 0.95 -> unsupported (priority)',
        decide(ambiguous, 0.95).state, 'unsupported');
  check('unit: argmax correctness (hazardous index 2)',
        decide([0.05, 0.05, 0.80, 0.10], 0.0).best, 2);
  check('unit: argmax correctness (general trash index 3)',
        decide([0.05, 0.05, 0.10, 0.80], 0.0).best, 3);

  // --- malformed / degenerate inputs (hardened decision layer) ---
  expectThrows('unit: 2-length probs throws', function () { decide([0.5, 0.5], 0.0); });
  expectThrows('unit: 5-length probs throws',
               function () { decide([0.25, 0.25, 0.25, 0.25, 0.0], 0.0); });
  expectThrows('unit: NaN in bins throws',
               function () { decide([0.9, NaN, 0.05, 0.05], 0.0); });
  expectThrows('unit: NaN reject throws',
               function () { decide(dominant, NaN); });
  expectThrows('unit: non-array probs throws',
               function () { decide('0.9,0.1', 0.0); });
  expectThrows('unit: Infinity reject throws',
               function () { decide(dominant, Infinity); });

  // --- returned verdict shape sanity ---
  const v = decide(dominant, 0.1);
  check('unit: verdict fields present',
        Object.keys(v).sort(), ['best', 'margin', 'reject', 'second', 'state', 'top']);
}

function runProductCases(decide) {
  if (!fs.existsSync(PRODUCT_JSON)) {
    failCount++;
    console.log('FAIL product cases: ' + PRODUCT_JSON + ' missing ' +
                '(run src/product_test_outputs.py first)');
    return;
  }
  const payload = JSON.parse(fs.readFileSync(PRODUCT_JSON, 'utf8'));
  console.log('product cases from ' + payload.model +
              ' (threshold ' + payload.reject_threshold + '):');

  const groups = {};
  for (const c of payload.cases) {
    const verdict = decide(c.bins, c.reject);
    if (c.expected !== null) {
      check('product[' + c.id + '] (' + c.group + ')', verdict.state, c.expected);
    } else {
      // Ground truth known to be weakly detected (collages): assert the
      // decision layer returned a valid state and log the outcome; the
      // measured detection rate is reported, never asserted as perfect.
      if (['supported', 'uncertain', 'unsupported'].indexOf(verdict.state) >= 0) {
        passCount++;
        console.log('PASS product[' + c.id + '] (' + c.group + ') state=' +
                    verdict.state + ' (informational; ground truth: ' +
                    c.ground_truth + ', reject=' + c.reject.toFixed(4) + ')');
      } else {
        failCount++;
        console.log('FAIL product[' + c.id + '] invalid state ' + verdict.state);
      }
    }
    (groups[c.group] = groups[c.group] || []).push(verdict.state);
  }

  console.log('--- group state summary ---');
  for (const g of Object.keys(groups).sort()) {
    const counts = {};
    groups[g].forEach(function (s) { counts[s] = (counts[s] || 0) + 1; });
    console.log('  ' + g + ': ' + JSON.stringify(counts));
  }
}

function main() {
  console.log('extracting decision block from web/index.html ...');
  const decide = makeDecide();
  console.log('--- unit cases ---');
  runUnitCases(decide);
  console.log('--- product cases ---');
  runProductCases(decide);
  console.log('---');
  console.log('TOTAL: ' + passCount + ' passed, ' + failCount + ' failed');
  // Do NOT call process.exit() here: it truncates pending stdout writes on
  // Windows (output gets lost when piped). Setting exitCode flushes normally.
  process.exitCode = failCount > 0 ? 1 : 0;
}

if (require.main === module) {
  main();
}
