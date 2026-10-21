"""
Equal feature budget comparison (H9-H11 of the paper): the eight detectors at an
equal feature budget.

For each detector and each fold, mutual information between every feature of the pool
TSFresh-774 + TDA-90 and the detector's label is estimated on the training fold only. From
that one ranking:
    TSFresh top-k   the k highest-ranked TSFresh-774 features
    union top-k     the k highest-ranked features of the whole pool
Each is fitted with the classifiers of src/detectors.py and scored the same way.

The fold assignment is read from the main detectors' oof.parquet, so the folds are the same
by construction. --per-class draws a subsample (with its folds) for a
quick test of the code path; its numbers mean nothing.

Outputs in --out: folds.csv, oof.parquet (as src/detectors.py), selected.csv (fold, char,
set, rank, feature, family, mi), timing.csv.

    python src/detectors_budget.py --features data/<tag>/features --folds-from runs/<tag>/main/oof.parquet --out runs/<tag>/budget
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detectors import LETTERS, fit_one, labels, load  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--folds-from", required=True, help="the main detectors' oof.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--models", nargs="+", default=["lr", "hgb"], choices=["lr", "hgb"])
    ap.add_argument("--per-class", type=int, default=None, help="test only: subsample per class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    parts, key, cls, X, names = load(Path(args.features), ["TSFresh-774", "TDA-90"])
    pool = np.hstack([X["TSFresh-774"], X["TDA-90"]])
    pool_names = names["TSFresh-774"] + names["TDA-90"]
    family = np.array(["TSFresh"] * len(names["TSFresh-774"]) + ["TDA"] * len(names["TDA-90"]))
    F = pd.read_parquet(args.folds_from, columns=["part", "series_id", "fold"])
    fold = pd.MultiIndex.from_frame(F[["part", "series_id"]]).get_indexer(key)
    assert (fold >= 0).all(), "series missing from the main run"
    fold = F["fold"].to_numpy()[fold]
    rows_kept = np.arange(len(cls))
    if args.per_class:
        rng = np.random.default_rng(args.seed)
        rows_kept = np.sort(np.concatenate([rng.choice(np.where(cls == c)[0], args.per_class, replace=False) for c in np.unique(cls)]))
    pool, cls, fold, key = pool[rows_kept], cls[rows_kept], fold[rows_kept], key[rows_kept]
    Y = labels(cls)
    lev = np.array([0 if c == "stationary" else c.count("+") + 1 for c in cls])
    print(f"{len(cls)} series, pool {pool.shape[1]} features (TSFresh {int((family == 'TSFresh').sum())}, "
          f"TDA {int((family == 'TDA').sum())}), top-{args.k}, folds of the main run", flush=True)

    rows, sel, timing = [], [], []
    oof = pd.DataFrame({"part": key.get_level_values("part"), "series_id": key.get_level_values("series_id"),
                        "class": cls, "level": lev, "fold": fold})
    t0 = time.time()
    for f in sorted(set(fold)):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        for j, c in enumerate(LETTERS):
            t = time.time()
            mi = mutual_info_classif(pool[tr], Y[tr, j], random_state=args.seed, n_jobs=-1)
            order = np.argsort(mi)[::-1]
            chosen = {"union top-%d" % args.k: order[: args.k],
                      "TSFresh top-%d" % args.k: order[family[order] == "TSFresh"][: args.k]}
            t_mi = time.time() - t
            for sname, idx in chosen.items():
                sel += [dict(fold=f, char=c, set=sname, rank=r + 1, feature=pool_names[i], family=family[i], mi=mi[i])
                        for r, i in enumerate(idx)]
                for model in args.models:
                    t = time.time()
                    pred, prob, _ = fit_one(model, args.seed, pool[np.ix_(tr, idx)], Y[tr, j], pool[np.ix_(te, idx)])
                    oof.loc[te, f"{sname}|{model}|{c}"] = prob
                    for k in sorted(set(lev[te])):
                        m = lev[te] == k
                        y = Y[te][m, j]
                        if len(set(y)) == 2:
                            rows.append(dict(set=sname, model=model, level=k, char=c, fold=f,
                                             bal_acc=balanced_accuracy_score(y, pred[m]), auc=roc_auc_score(y, prob[m])))
                    timing.append(dict(fold=f, char=c, set=sname, model=model, seconds=time.time() - t, mi_seconds=t_mi))
            share = (family[chosen["union top-%d" % args.k]] == "TDA").mean()
            print(f"  fold {f + 1} {c}: MI {t_mi:.0f}s, TDA share of union top-{args.k} {share:.2f} ({time.time() - t0:.0f}s)", flush=True)

    pd.DataFrame(rows).to_csv(out / "folds.csv", index=False)
    oof.to_parquet(out / "oof.parquet", index=False)
    pd.DataFrame(sel).to_csv(out / "selected.csv", index=False)
    pd.DataFrame(timing).to_csv(out / "timing.csv", index=False)
    print(f"Saved -> {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
