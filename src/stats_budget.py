"""
The tables and tests of the equal-feature-budget comparison (H9-H11 of the paper).
Reads the main detectors (TSFresh-774 alone), the combined sets (TSFresh-774 + TDA-90, if
present) and the budget run (TSFresh top-100, union top-100). Everything is reported whatever it shows.

    T10  mean of the eight detectors per level: TSFresh-774, TSFresh top-100, union top-100,
         TSFresh-774 + TDA-90
    T11  union top-100 minus TSFresh top-100 at each level, and L4 - L1      (H9, H10)
    T12  share of TDA-90 features in each detector's union top-100            (H11)
    T13  union top-100 minus TSFresh top-100 per detector at level 4

    python src/stats_budget.py runs/<tag>/main runs/<tag>/complementarity runs/<tag>/budget
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_main import LETTERS, fmt, nadeau_bengio  # noqa: E402


def main() -> None:
    final, v2, v3 = map(Path, sys.argv[1:4])
    V3 = pd.read_csv(v3 / "folds.csv")
    sets = sorted(V3.set.unique())
    TS = next(s for s in sets if s.startswith("TSFresh top"))
    UN = next(s for s in sets if s.startswith("union top"))
    models = sorted(V3.model.unique())
    parts = [pd.read_csv(final / "folds.csv").query("set == 'TSFresh-774' and model in @models"), V3]
    if (v2 / "folds.csv").exists():
        parts.append(pd.read_csv(v2 / "folds.csv").query("set == 'TSFresh-774+TDA-90' and model in @models"))
    D = pd.concat(parts)
    order = ["TSFresh-774", TS, UN, "TSFresh-774+TDA-90"]
    k = D.fold.nunique(); r = 1 / (k - 1)
    lines = [f"=== equal feature budget (addendum v3) ({TS} vs {UN}); folds of the final run; "
             f"Nadeau-Bengio corrected SE, two-sided p (* < 0.05, ** < 0.01)"]
    tables = {}
    M8 = D[D.level.between(1, 4)].groupby(["set", "model", "level", "fold"])[["bal_acc", "auc"]].mean()

    for metric in ("bal_acc", "auc"):
        x = M8[metric]
        rows = []
        for m in models:
            for s in order:
                if (s, m) not in x.index.droplevel(["level", "fold"]):
                    continue
                v = x.xs((s, m), level=["set", "model"])
                rows.append({"model": m, "set": s, **{f"L{lv}": fmt(v.xs(lv, level="level").mean(),
                                                                     v.xs(lv, level="level").std(ddof=1) * np.sqrt(1 / k + r))
                                                       for lv in (1, 2, 3, 4)}})
        T10 = pd.DataFrame(rows); tables[f"T10_mean_of_eight_{metric}"] = T10
        lines += [f"\nT10  mean of the eight detectors, {metric}", T10.to_string(index=False)]
        rows = []
        for m in models:
            d = x.xs((UN, m), level=["set", "model"]) - x.xs((TS, m), level=["set", "model"])
            row = {"model": m, **{f"L{lv}": fmt(*nadeau_bengio(d.xs(lv, level="level").sort_index().to_numpy(), r)) for lv in (1, 2, 3, 4)}}
            row["L4 - L1"] = fmt(*nadeau_bengio((d.xs(4, level="level") - d.xs(1, level="level")).sort_index().to_numpy(), r))
            rows.append(row)
        T11 = pd.DataFrame(rows); tables[f"T11_union_minus_tsfresh_{metric}"] = T11
        lines += [f"\nT11  {UN} minus {TS} ({metric})", T11.to_string(index=False)]

    S = pd.read_csv(v3 / "selected.csv")
    U = S[S.set == UN]
    T12 = (U.assign(tda=U.family == "TDA").groupby(["char", "fold"]).tda.mean().groupby("char")
           .agg(["mean", "min", "max"]).reindex(LETTERS).rename(columns={"mean": "TDA share", "min": "min over folds", "max": "max over folds"}))
    tables["T12_tda_share"] = T12.reset_index()
    lines += [f"\nT12  share of TDA-90 features in each detector's {UN}", T12.round(3).to_string()]
    top = U[U.family == "TDA"].groupby(["char", "feature"]).size().rename("folds").reset_index()
    top = top.sort_values(["char", "folds"], ascending=[True, False]).groupby("char").head(5)
    lines += ["  TDA-90 features most often selected, per detector (folds out of 5):"]
    for c in LETTERS:
        t = top[top.char == c]
        lines.append(f"    {c}: " + "; ".join(f"{n} ({v})" for n, v in zip(t.feature, t.folds)))

    y = D[D.level == 4].set_index(["set", "model", "char", "fold"])
    rows = []
    for metric in ("bal_acc", "auc"):
        for m in models:
            row = {"metric": metric, "model": m}
            for c in LETTERS:
                g = y[metric].xs((UN, m, c), level=["set", "model", "char"]) - y[metric].xs((TS, m, c), level=["set", "model", "char"])
                row[c] = fmt(*nadeau_bengio(g.sort_index().to_numpy(), r))
            rows.append(row)
    T13 = pd.DataFrame(rows); tables["T13_per_detector_level4"] = T13
    lines += [f"\nT13  {UN} minus {TS} per detector at level 4", T13.to_string(index=False)]

    tdir = v3 / "tables"; tdir.mkdir(exist_ok=True)
    for n, t in tables.items():
        t.to_csv(tdir / f"{n}.csv", index=False)
    text = "\n".join(lines)
    (v3 / "stats.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
