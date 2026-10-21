"""
The eight detectors: one binary classifier per characteristic (S V T R M Q P C), trained on
every level at once, scored per level.

    feature sets   TDA-90, TSFresh-774, catch22; with --sets also the combined sets of the
                   complementarity comparison (TSFresh-774+TDA-90, TSFresh-774+catch22)
    classifiers    lr   logistic regression on signed-log, standardised features (L2, C = 1)
                   hgb  scikit-learn HistGradientBoosting, defaults
                   both class-weighted ("balanced")
    MiniRocket     aeon's transform fitted per fold, scaler, ridge (class-weighted), as in
                   aeon's MiniRocketClassifier; recorded as classifier "ridge"
    protocol       5-fold stratified by class, seed 42; balanced accuracy at the classifier's
                   own threshold and ROC AUC, among the series of each level

Outputs in --out:
    folds.csv       set, model, level, char, fold, bal_acc, auc
    oof.parquet     out-of-fold score of every series for every set x model x characteristic
                    (probability, or the ridge decision value), with class, level and fold
    lr_coefs.csv    logistic-regression coefficients (standardised scale) per set, char, fold
    timing.csv      seconds per fold and set x model

    python src/detectors.py --features data/<tag>/features --raw data/<tag>/raw --out runs/<tag>/main
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
LETTERS = ["S", "V", "T", "R", "M", "Q", "P", "C"]
NAME = {"stochastic_trend": "S", "volatility": "V", "trend": "T", "trend_shift": "R",
        "mean_shift": "M", "variance_shift": "Q", "point_anomaly": "P", "collective_anomaly": "C"}
SETS = ["TDA-90", "TSFresh-774", "catch22", "MiniRocket"]
COMBINED = ["TSFresh-774+TDA-90", "TSFresh-774+catch22"]     # complementarity comparison (H5-H8)
FILES = {"TDA-90": "tda90", "TSFresh-774": "tsfresh", "catch22": "catch22"}


def signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


def make(name: str, seed: int):
    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler
        return make_pipeline(FunctionTransformer(signed_log), StandardScaler(),
                             LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000))
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)


def fit_one(name, seed, Xtr, ytr, Xte):
    m = make(name, seed).fit(Xtr, ytr)
    coef = m[-1].coef_.ravel() if name == "lr" else None
    return m.predict(Xte), m.predict_proba(Xte)[:, 1], coef


def load(features: Path, sets: list[str]):
    parts = sorted(p.name for p in features.iterdir() if (p / "labels.parquet").exists())
    L = pd.concat([pd.read_parquet(features / p / "labels.parquet").assign(part=p) for p in parts])
    key = pd.MultiIndex.from_arrays([L["part"], L.index], names=["part", "series_id"])
    out, names, cache = {}, {}, {}

    def one(s):
        if s not in cache:
            frames = [pd.read_parquet(features / p / f"{FILES[s]}.parquet") for p in parts]
            cols = sorted(set.intersection(*(set(f.columns) for f in frames))) if s == "TSFresh-774" else list(frames[0].columns)
            X = pd.concat([f[cols] for f in frames]).to_numpy(float)
            cache[s] = (np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), cols)
        return cache[s]

    for s in sets:
        if s == "MiniRocket":
            continue
        comps = [one(c) for c in s.split("+")]          # a combined set: its components side by side
        out[s] = np.hstack([c[0] for c in comps])
        names[s] = [n for c in comps for n in c[1]]
    return parts, key, L["primary_category"].to_numpy(), out, names


def labels(cls: np.ndarray) -> np.ndarray:
    return np.array([[int(l in ({NAME[p] for p in c.split("+")} if c != "stationary" else set())) for l in LETTERS]
                     for c in cls])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--raw", required=True, help="raw parts, for MiniRocket")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", default=SETS, choices=SETS + COMBINED)
    ap.add_argument("--models", nargs="+", default=["lr", "hgb"], choices=["lr", "hgb"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    parts, key, cls, X, names = load(Path(args.features), args.sets)
    Y = labels(cls)
    lev = np.array([0 if c == "stationary" else c.count("+") + 1 for c in cls])
    print(f"{len(cls)} series, {len(set(cls))} classes, parts {parts}; "
          + ", ".join(f"{s} {v.shape[1]}" for s, v in X.items()), flush=True)
    if "MiniRocket" in args.sets:
        raw = pd.concat([pd.read_parquet(Path(args.raw) / f"{p}.parquet", columns=["series_id", "time", "data"]).assign(part=p)
                         for p in parts]).sort_values(["part", "series_id", "time"])
        R = np.stack([g.to_numpy(float) for _, g in raw.groupby(["part", "series_id"])["data"]])
        assert len(R) == len(cls)
        sd = R.std(1, keepdims=True)
        R = ((R - R.mean(1, keepdims=True)) / np.where(sd > 0, sd, 1.0))[:, None, :]
        del raw

    rows, coefs, timing = [], [], []
    oof = pd.DataFrame({"part": key.get_level_values("part"), "series_id": key.get_level_values("series_id"),
                        "class": cls, "level": lev, "fold": -1})

    def score(model, name, f, te, j, pred, prob):
        oof.loc[te, f"{name}|{model}|{LETTERS[j]}"] = prob
        for k in sorted(set(lev[te])):
            m = lev[te] == k
            y = Y[te][m, j]
            if len(set(y)) == 2:
                rows.append(dict(set=name, model=model, level=k, char=LETTERS[j], fold=f,
                                 bal_acc=balanced_accuracy_score(y, pred[m]), auc=roc_auc_score(y, prob[m])))

    t0 = time.time()
    for f, (tr, te) in enumerate(StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(cls, cls)):
        oof.loc[te, "fold"] = f
        for model in args.models:
            for name, Xs in X.items():
                t = time.time()
                res = Parallel(n_jobs=len(LETTERS) if model == "lr" else 1)(
                    delayed(fit_one)(model, args.seed, Xs[tr], Y[tr, j], Xs[te]) for j in range(len(LETTERS)))
                for j, (pred, prob, coef) in enumerate(res):
                    score(model, name, f, te, j, pred, prob)
                    if coef is not None:
                        coefs += [dict(set=name, char=LETTERS[j], fold=f, feature=n, coef=c) for n, c in zip(names[name], coef)]
                timing.append(dict(fold=f, set=name, model=model, seconds=time.time() - t))
                print(f"  fold {f + 1}/{args.folds} {name} {model} ({time.time() - t:.0f}s)", flush=True)
        if "MiniRocket" in args.sets:
            from aeon.transformations.collection.convolution_based import MiniRocket
            from sklearn.linear_model import RidgeClassifierCV
            from sklearn.preprocessing import StandardScaler
            t = time.time()
            mr = MiniRocket(random_state=args.seed, n_jobs=-1).fit(R[tr])
            Ztr = mr.transform(R[tr]); sc = StandardScaler(with_mean=False).fit(Ztr)
            Ztr, Zte = sc.transform(Ztr), sc.transform(mr.transform(R[te]))
            for j in range(len(LETTERS)):
                rc = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight="balanced").fit(Ztr, Y[tr, j])
                score("ridge", "MiniRocket", f, te, j, rc.predict(Zte), rc.decision_function(Zte))
            del Ztr, Zte
            timing.append(dict(fold=f, set="MiniRocket", model="ridge", seconds=time.time() - t))
            print(f"  fold {f + 1}/{args.folds} MiniRocket ridge ({time.time() - t:.0f}s)", flush=True)
        print(f"  fold {f + 1}/{args.folds} done ({time.time() - t0:.0f}s)", flush=True)

    pd.DataFrame(rows).to_csv(out / "folds.csv", index=False)
    oof.to_parquet(out / "oof.parquet", index=False)
    pd.DataFrame(coefs).to_csv(out / "lr_coefs.csv", index=False)
    pd.DataFrame(timing).to_csv(out / "timing.csv", index=False)
    print(f"Saved -> {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
