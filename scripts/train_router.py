"""Train the BeamDial router from graded runs.

One logistic model per reasoning level predicts P(correct | prompt). We also write out-of-fold
predictions two ways, so the reported numbers are honest:
  cv5  : 5-fold CV, folds stratified by source (in-distribution)
  loso : leave-one-source-group-out (the router never saw that kind of task, a tougher test)

python scripts/train_router.py data/graded.jsonl
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from beamdial.features import N_FEATURES, featurize  # noqa: E402
from beamdial.router import LEVELS  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from graders import load_tasks  # noqa: E402

GROUPS = {"gsm8k": "gsm8k", "math_l1": "math", "math_l2": "math", "math_l3": "math", "math_l4": "math",
          "math_l5": "math", "amc23": "competition", "aime24": "competition", "olympiad": "competition",
          "humaneval": "humaneval", "mbpp": "mbpp",
          # v0.2 hard set
          "aime26": "frontier", "hmmt26": "frontier", "apex": "frontier", "apex_short": "frontier",
          "longctx_8k": "longctx", "longctx_32k": "longctx"}
C_REG = 0.5


def load(graded_path):
    tasks = load_tasks(ROOT)
    acc = defaultdict(lambda: defaultdict(list))
    toks = defaultdict(lambda: defaultdict(list))
    for r in map(json.loads, open(graded_path)):
        acc[r["task_id"]][r["effort"]].append(float(r["correct"]))
        toks[r["task_id"]][r["effort"]].append(r["usage"]["completion_tokens"])
    ids = sorted(t for t in acc if all(l in acc[t] for l in LEVELS))
    return tasks, ids, acc, toks


def matrix(texts):
    rows, cols, vals = [], [], []
    for i, t in enumerate(texts):
        for j, v in featurize(t).items():
            rows.append(i)
            cols.append(j)
            vals.append(v)
    return csr_matrix((vals, (rows, cols)), shape=(len(texts), N_FEATURES))


def fit(X, y):
    """Labels are mean correctness over samples; binarise at 0.5. Constant labels -> intercept only."""
    yb = (np.asarray(y) >= 0.5).astype(int)
    if yb.min() == yb.max():
        p = min(max(yb.mean(), 0.02), 0.98)
        return None, float(np.log(p / (1 - p)))
    m = LogisticRegression(C=C_REG, max_iter=2000, solver="liblinear")
    m.fit(X, yb)
    return m, float(m.intercept_[0])


def predict(model, b, X):
    if model is None:
        return np.full(X.shape[0], 1 / (1 + np.exp(-b)))
    return model.predict_proba(X)[:, 1]


def main():
    graded = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "graded.jsonl")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks, ids, acc, toks = load(graded)
    X = matrix([tasks[i]["prompt"] for i in ids])
    Y = {l: [float(np.mean(acc[i][l])) for i in ids] for l in LEVELS}
    src = [tasks[i]["source"] for i in ids]
    grp = [GROUPS[s] for s in src]
    print(f"{len(ids)} tasks with all {len(LEVELS)} levels")

    oof = {i: {"cv5": {}, "loso": {}} for i in ids}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(np.zeros(len(ids)), src):
        for l in LEVELS:
            m, b = fit(X[tr], [Y[l][k] for k in tr])
            for k, p in zip(te, predict(m, b, X[te])):
                oof[ids[k]]["cv5"][l] = float(p)
    for g in sorted(set(grp)):
        te = [k for k in range(len(ids)) if grp[k] == g]
        tr = [k for k in range(len(ids)) if grp[k] != g]
        for l in LEVELS:
            m, b = fit(X[tr], [Y[l][k] for k in tr])
            for k, p in zip(te, predict(m, b, X[te])):
                oof[ids[k]]["loso"][l] = float(p)
    (out_dir / "oof.json").write_text(json.dumps(oof))

    coef, intercept = {}, {}
    for l in LEVELS:
        m, b = fit(X, Y[l])
        intercept[l] = round(b, 6)
        coef[l] = {} if m is None else {str(j): round(float(w), 5) for j, w in enumerate(m.coef_[0]) if abs(w) > 1e-4}

    fam = {i: tasks[i]["family"] for i in ids}
    table = {}
    for f in ("math", "code", "all"):
        sel = [i for i in ids if f == "all" or fam[i] == f]
        # mean, not median: the mean is what you pay for on average across many requests
        table[f] = {l: int(np.mean([np.mean(toks[i][l]) for i in sel])) for l in LEVELS} if sel else {}
    weights = dict(version="0.1.0", model="reflection/Beam-1.0-501B-A23B", levels=LEVELS, coef=coef,
                   intercept=intercept, tokens=table, default_target=0.85, n_train_tasks=len(ids),
                   note="Trained on the BeamDial dataset (math + Python code). Out-of-domain prompts get rougher estimates.")
    dst = ROOT / "beamdial" / "router_weights.json"
    dst.write_text(json.dumps(weights, separators=(",", ":")))
    print(f"router weights -> {dst} ({dst.stat().st_size / 1024:.0f} KB); oof -> {out_dir / 'oof.json'}")


if __name__ == "__main__":
    main()
