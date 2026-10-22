// BeamDial router in the browser. A line-for-line port of beamdial/features.py and beamdial/router.py,
// checked against the Python version on every task in the dataset (scripts/check_js_parity.py).
// Nothing is sent anywhere: features, weights and the pick all stay in this page.

const DIM = 1 << 14;
const KEYWORDS = ["prove", "find all", "how many", "probability", "maximum", "minimum", "integer", "prime", "sum",
  "remainder", "polynomial", "triangle", "sequence", "expected", "function", "return", "list", "string",
  "sort", "dictionary", "recursion", "optimal", "efficient", "edge case", "regex", "matrix"];
export const LEVELS = ["low", "medium", "high", "xhigh", "max"];

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
const enc = new TextEncoder();
function crc32(str) {
  const bytes = enc.encode(str);
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

const TOKEN = /\\[a-zA-Z]+|[a-zA-Z_]+|\d+|[^\sa-zA-Z\d]/gu;
const pyLen = (s) => Array.from(s).length; // Python len() counts code points

export function featurize(text) {
  text = text || "";
  const low = text.toLowerCase();
  const toks = (text.match(TOKEN) || []).filter((t) => t.trim() !== "").map((t) => t.toLowerCase());
  const counts = new Map();
  const grams = toks.slice();
  for (let i = 0; i + 1 < toks.length; i++) grams.push(toks[i] + " " + toks[i + 1]);
  for (const g of grams) {
    const h = crc32(g);
    const idx = h % DIM;
    const sign = (h >>> 20) & 1 ? 1 : -1;
    counts.set(idx, (counts.get(idx) || 0) + sign);
  }
  const feats = new Map();
  let norm = 0;
  for (const [i, v] of counts) {
    if (!v) continue;
    const f = Math.sign(v) * Math.log1p(Math.abs(v));
    feats.set(i, f);
    norm += f * f;
  }
  norm = Math.sqrt(norm) || 1;
  for (const [i, v] of feats) feats.set(i, v / norm);

  const nums = (text.match(/\d{1,12}/g) || []).map(Number);
  const n = pyLen(text);
  const digits = (text.match(/\d/g) || []).length;
  const dense = [
    Math.log1p(n) / 10,
    Math.log1p(low.split(/\s+/).filter(Boolean).length) / 8,
    digits / Math.max(n, 1),
    Math.log1p((text.match(/\\[a-zA-Z]+/g) || []).length) / 5,
    Math.log1p((text.match(/\$/g) || []).length) / 5,
    text.includes("```") ? 1 : 0,
    text.includes("def ") ? 1 : 0,
    Math.log1p((text.match(/\n/g) || []).length) / 5,
    Math.log1p((text.match(/\?/g) || []).length) / 3,
    nums.length ? Math.log1p(Math.max(...nums)) / 30 : 0,
    ...KEYWORDS.map((k) => (low.includes(k) ? 1 : 0)),
  ];
  dense.forEach((v, i) => { if (v) feats.set(DIM + i, v); });
  return feats;
}

const sigmoid = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

export function family(prompt) {
  const p = prompt.toLowerCase();
  const codeKeys = ["```", "def ", "python", "function", "return ", "implement", "class "];
  const codeHits = codeKeys.filter((k) => p.includes(k)).length;
  if (codeHits >= 1 && (p.includes("python") || p.includes("def ") || p.includes("```") || codeHits >= 2)) return "code";
  const mathKeys = ["\\", "$", "\\boxed", "solve", "compute", "how many", "find the", "integer", "probability", "equation",
    "prove", "sum of", "remainder", "triangle", "polynomial", "evaluate", "simplify"];
  const mathHits = mathKeys.filter((k) => p.includes(k)).length;
  const digits = (p.match(/\d/g) || []).length;
  return mathHits >= 1 || digits >= 4 ? "math" : "other";
}

export class Router {
  constructor(w) {
    this.w = w;
    this.levels = w.levels;
    this.coef = {};
    for (const l of this.levels) this.coef[l] = new Map(Object.entries(w.coef[l]).map(([k, v]) => [Number(k), v]));
    this.intercept = w.intercept;
    this.tokenTable = w.tokens || {};
  }
  probs(prompt) {
    const x = featurize(prompt);
    const out = {};
    for (const l of this.levels) {
      let z = this.intercept[l];
      const c = this.coef[l];
      for (const [i, v] of x) z += v * (c.get(i) || 0);
      out[l] = sigmoid(z);
    }
    let best = 0;
    for (const l of this.levels) { best = Math.max(best, out[l]); out[l] = best; }
    return out;
  }
  tokens(prompt) {
    const t = this.tokenTable[family(prompt)] || this.tokenTable.all || {};
    return Object.fromEntries(this.levels.map((l) => [l, Number(t[l] || 0)]));
  }
  explain(prompt, target) {
    target = target ?? this.w.default_target ?? 0.85;
    const pr = this.probs(prompt), tk = this.tokens(prompt);
    let pick = this.levels.find((l) => pr[l] >= target);
    if (!pick) {
      const top = Math.max(...Object.values(pr));
      pick = this.levels.find((l) => pr[l] >= top - 1e-9);
    }
    const fam = family(prompt);
    return { effort: pick, p: pr[pick], probs: pr, tokens: tk, family: fam, inDomain: fam !== "other" };
  }
}
