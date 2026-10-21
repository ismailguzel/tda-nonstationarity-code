"""
Which TDA-90 features carry a detector — exploratory analysis
(feature-group ablation, Section 5.5 of the paper).

For each characteristic given (default V, where TDA-90 beats TSFresh-774, and P, where it
loses), three views, all on the main run's data and folds:

1. Coefficients. The logistic-regression coefficients saved by src/detectors.py
   (standardised scale), mean over the 5 folds, and in how many folds the sign agrees.
   TDA-90 features are correlated, so a coefficient alone can mislead; 2 and 3 check it.
2. Blocks. TDA-90 has six blocks — the paper features of window Rips H0 / H1, mean / std
   over windows; windowed sublevel H0; whole-series sublevel H0. The detector is refitted
   (logistic regression, same pipeline and folds as src/detectors.py) on each block alone
   and on TDA-90 without each block; balanced accuracy at levels 1 and 4.
3. Single features. ROC AUC of each feature alone, direction-free (max(AUC, 1 - AUC)),
   among the series of level 1 and of level 4 — for TDA-90 and, as contrast, TSFresh-774.

    python src/interpret.py --run runs/<tag>/main --features data/<tag>/features --out runs/<tag>/interpret
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detectors import LETTERS, labels, load, make  # noqa: E402

BLOCKS = {
    "paper H0, mean over windows": lambda n: n.startswith("pwin_H0__") and n.endswith("__mean"),
    "paper H0, std over windows": lambda n: n.startswith("pwin_H0__") and n.endswith("__std"),
    "paper H1, mean over windows": lambda n: n.startswith("pwin_H1__") and n.endswith("__mean"),
    "paper H1, std over windows": lambda n: n.startswith("pwin_H1__") and n.endswith("__std"),
    "windowed sublevel H0": lambda n: n.startswith("win_sub_H0__"),
    "whole-series sublevel H0": lambda n: n.startswith("topo__sub_H0__"),
}


def fit_score(X, y, lev, tr, te, seed):
    m = make("lr", seed).fit(X[tr], y[tr])
    pred, prob = m.predict(X[te]), m.predict_proba(X[te])[:, 1]
    out = {}
    for k in (1, 2, 3, 4):
        s = lev[te] == k
        if len(set(y[te][s])) == 2:
            out[f"L{k}"] = balanced_accuracy_score(y[te][s], pred[s])
            out[f"AUC L{k}"] = roc_auc_score(y[te][s], prob[s])
    return out


def single_auc(X, names, y, lev, k):
    s = lev == k
    a = np.array([roc_auc_score(y[s], X[s, i]) if X[s, i].std() > 0 else 0.5 for i in range(X.shape[1])])
    return pd.Series(np.maximum(a, 1 - a), index=names)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chars", nargs="+", default=["V", "P"], choices=LETTERS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    _, _, cls, X, names = load(Path(args.features), ["TDA-90", "TSFresh-774"])
    Y = labels(cls)
    lev = np.array([0 if c == "stationary" else c.count("+") + 1 for c in cls])
    T, tn = X["TDA-90"], names["TDA-90"]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=args.seed).split(cls, cls))
    block_of = {n: b for n in tn for b, f in BLOCKS.items() if f(n)}
    assert len(block_of) == len(tn) == 90, "every TDA-90 feature belongs to exactly one block"
    coefs = pd.read_csv(Path(args.run) / "lr_coefs.csv")
    lines = ["=== Interpretation of the TDA-90 detectors — exploratory, not in the analysis plan"]

    for c in args.chars:
        j = LETTERS.index(c)
        y = Y[:, j]
        lines.append(f"\n##### {c}")

        C = coefs[(coefs.set == "TDA-90") & (coefs.char == c)].pivot(index="feature", columns="fold", values="coef")
        S = pd.DataFrame({"mean coef": C.mean(1), "sd": C.std(1),
                          "same sign in folds": (np.sign(C).eq(np.sign(C.mean(1)), axis=0)).sum(1),
                          "block": [block_of[n] for n in C.index]})
        S = S.reindex(S["mean coef"].abs().sort_values(ascending=False).index)
        lines += ["\n  1. largest logistic-regression coefficients (standardised features)", "  " + S.head(12).round(3).to_string().replace("\n", "\n  "),
                  "\n  sum of |mean coef| per block", "  " + S.groupby("block")["mean coef"].apply(lambda v: v.abs().sum()).sort_values(ascending=False).round(2).to_string().replace("\n", "\n  ")]

        sets = {"TDA-90 (all 90)": np.arange(90)}
        for b in BLOCKS:
            idx = np.array([i for i, n in enumerate(tn) if block_of[n] == b])
            sets[f"only {b} ({len(idx)})"] = idx
            sets[f"without {b}"] = np.setdiff1d(np.arange(90), idx)
        res = Parallel(n_jobs=-1)(delayed(fit_score)(T[:, idx], y, lev, tr, te, args.seed)
                                  for idx in sets.values() for tr, te in folds)
        R = pd.DataFrame(res); R["set"] = np.repeat(list(sets), len(folds))
        B = R.groupby("set", sort=False)[["L1", "L4", "AUC L1", "AUC L4"]].mean()
        B.to_csv(out / f"blocks_{c}.csv")
        lines += ["\n  2. blocks: logistic regression on a block alone / without it (mean of 5 folds)", "  " + B.round(3).to_string().replace("\n", "\n  ")]

        for k in (1, 4):
            a = single_auc(T, tn, y, lev, k).sort_values(ascending=False)
            b = single_auc(X["TSFresh-774"], names["TSFresh-774"], y, lev, k).sort_values(ascending=False)
            a.to_csv(out / f"single_auc_{c}_L{k}_tda90.csv"); b.to_csv(out / f"single_auc_{c}_L{k}_tsfresh.csv")
            lines += [f"\n  3. best single features, level {k} (direction-free AUC)",
                      "     TDA-90:      " + "; ".join(f"{n} {v:.3f}" for n, v in a.head(5).items()),
                      "     TSFresh-774: " + "; ".join(f"{n} {v:.3f}" for n, v in b.head(5).items())]
        S.to_csv(out / f"coefs_{c}.csv")

    text = "\n".join(lines)
    (out / "interpret.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
