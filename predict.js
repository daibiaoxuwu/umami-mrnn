// Umami-MRNN inference in plain JavaScript (browser and Node).
// Ensemble of an MLP on sequence descriptors and an RNN on per-residue
// property codes, ported from the original Keras models.

export const AA = 'ACDEFGHIKLMNPQRSTVWY';
export const MIN_LEN = 2;
export const MAX_LEN = 39;
const BP_WEIGHT = 0.67;
export const UMAMI_CUTOFF = 40; // predicted threshold (mmol/L) at or above this = non-umami

// ---------- MLP features (iFeature descriptors, 1093 values) ----------

const PROPERTIES = [ // [group1, group2, group3] per physicochemical property
  ['RKEDQN', 'GASTPHY', 'CLVIMFW'],            // hydrophobicity_PRAM900101
  ['QSTNGDE', 'RAHCKMV', 'LYPFIW'],            // hydrophobicity_ARGP820101
  ['QNGSWTDERA', 'HMCKV', 'LPFYI'],            // hydrophobicity_ZIMJ680101
  ['KPDESNQT', 'GRHA', 'YMFWLCVI'],            // hydrophobicity_PONP930101
  ['KDEQPSRNTG', 'AHYMLV', 'FIWC'],            // hydrophobicity_CASG920101
  ['RDKENQHYP', 'SGTAW', 'CVLIMF'],            // hydrophobicity_ENGD860101
  ['KERSQD', 'NTPG', 'AYHWVMFLIC'],            // hydrophobicity_FASG890101
  ['GASTPDC', 'NVEQIL', 'MHKFRYW'],            // normwaalsvolume
  ['LIFWCMVY', 'PATGS', 'HQRKNED'],            // polarity
  ['GASDT', 'CPNVEQIL', 'KMHFRYW'],            // polarizability
  ['KR', 'ANCQGHILMFPSTWYV', 'DE'],            // charge
  ['EALMQKRH', 'VIYCWFT', 'GNPSD'],            // secondarystruct
  ['ALFCGIVW', 'RKQEND', 'MSPTHY'],            // solventaccess
];
const CODONS = { A: 4, C: 2, D: 2, E: 2, F: 2, G: 4, H: 2, I: 3, K: 2, L: 6,
                 M: 1, N: 2, P: 4, Q: 2, R: 6, S: 6, T: 4, V: 4, W: 1, Y: 2 };

const count = (set, seq) => [...seq].filter(c => set.includes(c)).length;

const aac = seq => [...AA].map(a => count(a, seq) / seq.length);

const ctdc = seq => PROPERTIES.flatMap(([g1, g2]) => {
  const c1 = count(g1, seq) / seq.length, c2 = count(g2, seq) / seq.length;
  return [c1, c2, 1 - c1 - c2];
});

// Position (% of length) of the first, 25%, 50%, 75% and last residue of each group.
const ctdd = seq => PROPERTIES.flatMap(groups => groups.flatMap(set => {
  const n = count(set, seq);
  if (n === 0) return [0, 0, 0, 0, 0];
  return [1, Math.floor(0.25 * n), Math.floor(0.5 * n), Math.floor(0.75 * n), n].map(cutoff => {
    cutoff = Math.max(cutoff, 1);
    for (let i = 0, seen = 0; i < seq.length; i++)
      if (set.includes(seq[i]) && ++seen === cutoff) return (i + 1) / seq.length * 100;
  });
}));

const ctdt = seq => {
  const pairs = seq.length - 1;
  return PROPERTIES.flatMap(([g1, g2, g3]) => {
    const t = [0, 0, 0]; // 1<->2, 1<->3, 2<->3 transitions
    for (let j = 0; j < pairs; j++) {
      const a = seq[j], b = seq[j + 1];
      const between = (x, y) => (x.includes(a) && y.includes(b)) || (y.includes(a) && x.includes(b));
      if (between(g1, g2)) t[0]++;
      else if (between(g1, g3)) t[1]++;
      else if (between(g2, g3)) t[2]++;
    }
    return t.map(v => v / pairs);
  });
};

const dpc = seq => {
  const v = new Array(400).fill(0);
  for (let j = 0; j < seq.length - 1; j++) v[AA.indexOf(seq[j]) * 20 + AA.indexOf(seq[j + 1])]++;
  const total = seq.length - 1;
  return v.map(x => x / total);
};

// Dipeptide deviation from the mean expected from codon usage.
const dde = (seq, composition) => composition.map((observed, k) => {
  const a = AA[Math.floor(k / 20)], b = AA[k % 20];
  const mean = (CODONS[a] / 61) * (CODONS[b] / 61);
  const variance = mean * (1 - mean) / (seq.length - 1);
  return (observed - mean) / Math.sqrt(variance);
});

export function descriptors(seq) {
  const pairs = dpc(seq);
  return [...aac(seq), ...ctdc(seq), ...ctdd(seq), ...ctdt(seq), ...dde(seq, pairs), ...pairs];
}

// ---------- RNN features (10-bit overlapping property code per residue) ----------

const OPF = { A: '0000100110', R: '1101000000', N: '1000000100', D: '1011000100', C: '1000100110',
              Q: '1000000000', E: '1011000000', G: '0000100110', H: '1101101000', I: '0000110000',
              L: '0000110000', K: '1101100000', M: '0000100000', F: '0000101000', P: '0000000101',
              S: '1000000110', T: '1000100100', W: '1000101000', Y: '1000101000', V: '0000110100' };
// The original model padded to 39 residues with a masked value, so only the
// real residues reach the RNN.
export const residueCodes = seq => [...seq].map(c => [...OPF[c]].map(Number));

// ---------- networks ----------

// y = x W + b, with W stored row-major as [in][out]
function dense(x, { kernel, bias }, relu = false) {
  const out = Float64Array.from(bias);
  for (let i = 0; i < x.length; i++) {
    if (x[i] === 0) continue;
    for (let j = 0, row = i * out.length; j < out.length; j++) out[j] += x[i] * kernel[row + j];
  }
  return relu ? out.map(v => Math.max(v, 0)) : out;
}

// h_t = tanh(x_t K + h_{t-1} R + b); returns every hidden state.
function simpleRnn(xs, { kernel, recurrent_kernel, bias }) {
  let h = new Float64Array(bias.length);
  return xs.map(x => {
    const a = dense(x, { kernel, bias }), r = dense(h, { kernel: recurrent_kernel, bias: a });
    return (h = r.map(Math.tanh));
  });
}

export async function loadModel(baseUrl = new URL('./model/', import.meta.url)) {
  const read = async name => {
    const res = await fetch(new URL(name, baseUrl));
    if (!res.ok) throw new Error(`Failed to load ${name}`);
    return res;
  };
  const manifest = await (await read('manifest.json')).json();
  const data = new Float32Array(await (await read('weights.bin')).arrayBuffer());
  const w = {};
  for (const [key, { shape, offset }] of Object.entries(manifest)) {
    const [layer, name] = [key.slice(0, key.lastIndexOf('.')), key.slice(key.lastIndexOf('.') + 1)];
    (w[layer] ??= {})[name] = data.subarray(offset, offset + shape.reduce((a, b) => a * b));
  }
  return w;
}

export function validate(seq) {
  if (!/^[A-Z]+$/.test(seq) || [...seq].some(c => !AA.includes(c)))
    return 'contains letters that are not standard amino acids';
  if (seq.length < MIN_LEN || seq.length > MAX_LEN)
    return `length must be ${MIN_LEN}-${MAX_LEN} residues`;
  return null;
}

// Returns { bp, rnn, threshold, umami } for one peptide sequence.
export function predict(w, sequence) {
  const seq = sequence.trim().toUpperCase();
  const error = validate(seq);
  if (error) throw new Error(error);

  const x = descriptors(seq);
  const bp = dense(dense(dense(x, w['bp.d1'], true), w['bp.d2'], true), w['bp.d3'])[0];

  const states = simpleRnn(simpleRnn(residueCodes(seq), w['rnn.r1']), w['rnn.r2']);
  const rnn = dense(states.at(-1), w['rnn.d'])[0];

  const threshold = BP_WEIGHT * bp + (1 - BP_WEIGHT) * rnn;
  return { bp, rnn, threshold, umami: threshold < UMAMI_CUTOFF };
}

// Parse either FASTA (">name" lines) or one sequence per line.
export function parseInput(text) {
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  if (!lines.some(l => l.startsWith('>'))) return lines.flatMap(l => l.split(/\s+/)).map(s => ({ name: s, seq: s }));
  const records = [];
  for (const l of lines) {
    if (l.startsWith('>')) records.push({ name: l.slice(1).trim(), seq: '' });
    else if (records.length) records.at(-1).seq += l.replace(/\s+/g, '');
    else records.push({ name: l, seq: l });
  }
  return records;
}
