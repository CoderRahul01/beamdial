"""Prompt features for the BeamDial router. Pure Python, no dependencies, deterministic across machines.

Hashed word unigrams and bigrams (log term frequency, L2 normalised) plus a few dense signals.
"""
import math
import re
import zlib

DIM = 1 << 14
_TOKEN = re.compile(r"\\[a-zA-Z]+|[a-zA-Z_]+|\d+|[^\sa-zA-Z\d]")
KEYWORDS = [
    "prove", "find all", "how many", "probability", "maximum", "minimum", "integer", "prime", "sum",
    "remainder", "polynomial", "triangle", "sequence", "expected", "function", "return", "list", "string",
    "sort", "dictionary", "recursion", "optimal", "efficient", "edge case", "regex", "matrix",
]
DENSE = ["len_chars", "len_words", "digits_frac", "latex_cmds", "dollars", "code_fence", "has_def",
         "lines", "questions", "max_number"] + ["kw_" + k.replace(" ", "_") for k in KEYWORDS]


def _h(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))


def featurize(text: str) -> dict:
    """Return {index: value}. Indices < DIM are hashed n-grams; DIM + i are dense features."""
    text = text or ""
    low = text.lower()
    toks = [t.lower() for t in _TOKEN.findall(text) if not t.isspace()]
    counts = {}
    grams = toks + [a + " " + b for a, b in zip(toks, toks[1:])]
    for g in grams:
        h = _h(g)
        idx = h % DIM
        sign = 1.0 if (h >> 20) & 1 else -1.0
        counts[idx] = counts.get(idx, 0.0) + sign
    feats = {i: math.copysign(math.log1p(abs(v)), v) for i, v in counts.items() if v}
    norm = math.sqrt(sum(v * v for v in feats.values())) or 1.0
    feats = {i: v / norm for i, v in feats.items()}

    nums = [int(n) for n in re.findall(r"\d{1,12}", text)]
    dense = [
        math.log1p(len(text)) / 10,
        math.log1p(len(low.split())) / 8,
        sum(c.isdigit() for c in text) / max(len(text), 1),
        math.log1p(len(re.findall(r"\\[a-zA-Z]+", text))) / 5,
        math.log1p(text.count("$")) / 5,
        1.0 if "```" in text else 0.0,
        1.0 if "def " in text else 0.0,
        math.log1p(text.count("\n")) / 5,
        math.log1p(text.count("?")) / 3,
        math.log1p(max(nums)) / 30 if nums else 0.0,
    ] + [1.0 if k in low else 0.0 for k in KEYWORDS]
    for i, v in enumerate(dense):
        if v:
            feats[DIM + i] = v
    return feats


N_FEATURES = DIM + len(DENSE)
