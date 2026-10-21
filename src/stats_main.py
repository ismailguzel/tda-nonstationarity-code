"""
The tables and tests of the main comparison (H1-H4 of the paper), from the outputs of
src/detectors.py. Written before the final run; every test is reported whatever it shows.

Corrected tests: Nadeau & Bengio (2003) — the variance of the k fold differences is
inflated by (1/k + n_test/n_train) because the training sets overlap; t with k - 1 df,
two-sided. Every comparison is TDA-90 against another set under the same classifier
(MiniRocket under its own ridge).

    T1  mean of the eight detectors per level, every set x classifier, balanced accuracy
        and AUC, with corrected SE
    T2  drop from level 1 to level 4 (primary measure), and TDA-90 minus each other set
    T3  level-1 and level-4 differences, TDA-90 minus each other set
    T4  every detector at levels 1 and 4; TDA-90 minus each other set per detector at level 4
    T5  all eight right (exact set) per level, from the out-of-fold scores

    python src/stats_main.py runs/<tag>/main
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REF = "TDA-90"
LETTERS = ["S", "V", "T", "R", "M", "Q", "P", "C"]


def nadeau_bengio(d: np.ndarray, test_over_train: float) -> tuple[float, float, float]:
    J = len(d)
    var = d.var(ddof=1) * (1 / J + test_over_train)
    t = d.mean() / np.sqrt(var) if var > 0 else np.inf
    return d.mean(), np.sqrt(var), 2 * stats.t.sf(abs(t), J - 1)


def fmt(m, se, p=None):
    s = f"{m:+.3f} ({se:.3f})" if p is not None else f"{m:.3f} ({se:.3f})"
    return s if p is None else s + f" p={p:.3f}" + ("**" if p < 0.01 else "*" if p < 0.05 else "")


def main() -> None:
    run = Path(sys.argv[1])
    D = pd.read_csv(run / "folds.csv")
    k = D.fold.nunique()
    r = 1 / (k - 1)                                    # n_test / n_train for k-fold
    keys = list(D[["set", "model"]].drop_duplicates().itertuples(index=False, name=None))
    refs = [kk for kk in keys if kk[0] == REF]
    pairs = [(rk, ok) for rk in refs for ok in keys if ok[0] != REF and (ok[1] == rk[1] or ok[0] == "MiniRocket")]
    lines = [f"=== {run}: {D.level.max()} levels, {k} folds; Nadeau-Bengio corrected SE, two-sided p (* < 0.05, ** < 0.01)"]
    tables = {}

    M8 = D[D.level.between(1, 4)].groupby(["set", "model", "level", "fold"])[["bal_acc", "auc"]].mean()
    for metric in ("bal_acc", "auc"):
        x = M8[metric]
        rows = []
        for s, m in keys:
            v = x.xs((s, m), level=["set", "model"])
            row = {"set": s, "model": m}
            for lv in (1, 2, 3, 4):
                f = v.xs(lv, level="level").to_numpy()
                row[f"L{lv}"] = fmt(f.mean(), f.std(ddof=1) * np.sqrt(1 / k + r))
            drop = (v.xs(4, level="level") - v.xs(1, level="level")).to_numpy()
            row["drop L1->L4"] = fmt(drop.mean(), drop.std(ddof=1) * np.sqrt(1 / k + r))
            rows.append(row)
        T1 = pd.DataFrame(rows); tables[f"T1_T2_mean_of_eight_{metric}"] = T1
        lines += [f"\nT1/T2  mean of the eight detectors, {metric}", T1.to_string(index=False)]

        rows = []
        for (rs, rm), (os_, om) in pairs:
            a, b = x.xs((rs, rm), level=["set", "model"]), x.xs((os_, om), level=["set", "model"])
            row = {"comparison": f"{rs} ({rm}) - {os_} ({om})"}
            dd = ((a.xs(4, level="level") - a.xs(1, level="level")) - (b.xs(4, level="level") - b.xs(1, level="level"))).sort_index().to_numpy()
            row["drop difference"] = fmt(*nadeau_bengio(dd, r))
            for lv in (1, 4):
                row[f"level {lv}"] = fmt(*nadeau_bengio((a.xs(lv, level="level") - b.xs(lv, level="level")).sort_index().to_numpy(), r))
            rows.append(row)
        T23 = pd.DataFrame(rows); tables[f"T2_T3_differences_{metric}"] = T23
        lines += [f"\nT2/T3  TDA-90 minus each set ({metric}); drop difference > 0 = TDA-90 degrades less", T23.to_string(index=False)]

    for lv in (1, 4):
        P = D[D.level == lv].groupby(["set", "model", "char"]).bal_acc.mean().unstack()[LETTERS]
        tables[f"T4_per_detector_level{lv}"] = P.reset_index()
        lines += [f"\nT4  balanced accuracy per detector, level {lv}", P.round(3).to_string()]
    rows = []
    x = D[D.level == 4].set_index(["set", "model", "char", "fold"]).bal_acc
    for (rs, rm), (os_, om) in pairs:
        row = {"comparison": f"{rs} ({rm}) - {os_} ({om})"}
        for c in LETTERS:
            d = (x.xs((rs, rm, c), level=["set", "model", "char"]) - x.xs((os_, om, c), level=["set", "model", "char"])).sort_index().to_numpy()
            row[c] = fmt(*nadeau_bengio(d, r))
        rows.append(row)
    T4 = pd.DataFrame(rows); tables["T4_per_detector_level4_differences"] = T4
    lines += ["\nT4  per detector at level 4, TDA-90 minus each set (balanced accuracy)", T4.to_string(index=False)]

    O = pd.read_parquet(run / "oof.parquet")
    rows = []
    for s, m in keys:
        cols = [f"{s}|{m}|{c}" for c in LETTERS]
        thr = 0.0 if m == "ridge" else 0.5
        pred = (O[cols].to_numpy() > thr).astype(int)
        truth = np.array([[int(c in ({n for n in cl.split("+")} if cl != "stationary" else set()))
                           for c in ["stochastic_trend", "volatility", "trend", "trend_shift", "mean_shift",
                                     "variance_shift", "point_anomaly", "collective_anomaly"]] for cl in O["class"]])
        ok = (pred == truth).all(1)
        rows.append({"set": s, "model": m, **{f"L{lv}": ok[O.level == lv].mean() for lv in sorted(O.level.unique())}})
    T5 = pd.DataFrame(rows); tables["T5_exact_set"] = T5
    lines += ["\nT5  all eight right, by level (out-of-fold)", T5.round(3).to_string(index=False)]

    tdir = run / "tables"; tdir.mkdir(exist_ok=True)
    for n, t in tables.items():
        t.to_csv(tdir / f"{n}.csv", index=False)
    text = "\n".join(lines)
    (run / "stats.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
