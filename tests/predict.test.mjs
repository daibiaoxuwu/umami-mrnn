// Checks predict.js against predictions from the original Keras models.
// Run: node tests/predict.test.mjs
import { readFileSync } from 'node:fs';
import { descriptors, predict, UMAMI_CUTOFF } from '../predict.js';

const here = new URL('.', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('../model/manifest.json', here)));
const data = new Float32Array(readFileSync(new URL('../model/weights.bin', here)).buffer.slice(0));
const w = {};
for (const [key, { shape, offset }] of Object.entries(manifest)) {
  const i = key.lastIndexOf('.');
  (w[key.slice(0, i)] ??= {})[key.slice(i + 1)] = data.subarray(offset, offset + shape.reduce((a, b) => a * b));
}

const fixture = JSON.parse(readFileSync(new URL('fixture.json', here)));
let worst = 0, failures = 0;
for (const row of fixture) {
  if (row.descriptors) {
    const d = descriptors(row.seq);
    const maxDiff = Math.max(...d.map((v, i) => Math.abs(v - row.descriptors[i])));
    if (d.length !== 1093 || maxDiff > 1e-9) { failures++; console.log('descriptor mismatch', row.seq, d.length, maxDiff); }
  }
  const p = predict(w, row.seq);
  const diff = Math.max(Math.abs(p.bp - row.bp), Math.abs(p.rnn - row.rnn), Math.abs(p.threshold - row.threshold));
  worst = Math.max(worst, diff);
  if (diff > 1e-2 || p.umami !== row.threshold < UMAMI_CUTOFF) { failures++; console.log('mismatch', row, p); }
}
console.log(`${fixture.length} peptides, max abs difference ${worst.toExponential(2)}, ${failures} failures`);
process.exit(failures ? 1 : 0);
