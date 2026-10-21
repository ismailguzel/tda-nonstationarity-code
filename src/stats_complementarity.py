"""
The tables and tests of the complementarity comparison (H5-H8 of the paper): what TDA-90 adds on top
of TSFresh-774, with catch22 as the control. Reads the main detectors (TSFresh-774 alone) and
the combined sets, checks that both used the same fold assignment, and
reports everything whatever it shows.

    gain(set) = mean of the eight detectors of set  -  that of TSFresh-774 alone,
                same classifier, same fold

    T6  mean of the eight detectors per level: TSFresh-774, +TDA-90, +catch22
    T7  gain at each level; gain L4 - gain L1                       (H5, H6)
    T8  gain from TDA-90 minus gain from catch22, level 4             (H7)
    T9  gain from TDA-90 per detector, level 4                        (H8)

Nadeau–Bengio corrected t-tests as in src/stats.py.

    python src/stats_complementarity.py runs/<tag>/main runs/<tag>/complementarity
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_main import LETTERS, fmt, nadeau_bengio  # noqa: E402

BASE, TDA, CTRL = "TSFresh-774", "TSFresh-774+TDA-90", "TSFresh-774+catch22"


def main() -> None:
    final, v2 = Path(sys.argv[1]), Path(sys.argv[2])
    a = pd.read_parquet(final / "oof.parquet", columns=["part", "series_id", "fold"])
    b = pd.read_parquet(v2 / "oof.parquet", columns=["part", "series_id", "fold"])
    same = a.merge(b, on=["part", "series_id"], suffixes=("_final", "_v2"))
    assert len(same) == len(a) == len(b) and (same.fold_final == same.fold_v2).all(), "fold assignments differ"

    V = pd.read_csv(v2 / "folds.csv")
    models = sorted(V.model.unique())
    D = pd.concat([pd.read_csv(final / "folds.csv").query("set == @BASE and model in @models"), V])
    k = D.fold.nunique(); r = 1 / (k - 1)
    lines = [f"=== complementarity (addendum v2): TSFresh-774 alone vs + TDA-90 and + catch22; fold assignment identical "
             f"({len(a)} series); Nadeau-Bengio corrected SE, two-sided p (* < 0.05, ** < 0.01)"]
    tables = {}
    M8 = D[D.level.between(1, 4)].groupby(["set", "model", "level", "fold"])[["bal_acc", "auc"]].mean()

    for metric in ("bal_acc", "auc"):
        x = M8[metric]
        rows = []
        for m in models:
            for s in (BASE, TDA, CTRL):
                v = x.xs((s, m), level=["set", "model"])
                rows.append({"model": m, "set": s, **{f"L{lv}": fmt(v.xs(lv, level="level").mean(),
                                                                     v.xs(lv, level="level").std(ddof=1) * np.sqrt(1 / k + r))
                                                       for lv in (1, 2, 3, 4)}})
        T6 = pd.DataFrame(rows); tables[f"T6_mean_of_eight_{metric}"] = T6
        lines += [f"\nT6  mean of the eight detectors, {metric}", T6.to_string(index=False)]

        rows = []
        for m in models:
            base = x.xs((BASE, m), level=["set", "model"])
            gains = {s: x.xs((s, m), level=["set", "model"]) - base for s in (TDA, CTRL)}
            for s, g in gains.items():
                row = {"model": m, "added": s.split("+")[1]}
                for lv in (1, 2, 3, 4):
                    row[f"gain L{lv}"] = fmt(*nadeau_bengio(g.xs(lv, level="level").sort_index().to_numpy(), r))
                row["gain L4 - gain L1"] = fmt(*nadeau_bengio((g.xs(4, level="level") - g.xs(1, level="level")).sort_index().to_numpy(), r))
                rows.append(row)
            d = (gains[TDA] - gains[CTRL]).xs(4, level="level").sort_index().to_numpy()
            rows.append({"model": m, "added": "TDA-90 minus catch22", "gain L4": fmt(*nadeau_bengio(d, r))})
        T78 = pd.DataFrame(rows).fillna(""); tables[f"T7_T8_gains_{metric}"] = T78
        lines += [f"\nT7/T8  gain over TSFresh-774 alone ({metric}); last row per model: TDA-90's gain minus catch22's at level 4",
                  T78.to_string(index=False)]

    rows = []
    y = D[D.level == 4].set_index(["set", "model", "char", "fold"])
    for metric in ("bal_acc", "auc"):
        for m in models:
            row = {"metric": metric, "model": m}
            for c in LETTERS:
                g = (y[metric].xs((TDA, m, c), level=["set", "model", "char"]) - y[metric].xs((BASE, m, c), level=["set", "model", "char"]))
                row[c] = fmt(*nadeau_bengio(g.sort_index().to_numpy(), r))
            rows.append(row)
    T9 = pd.DataFrame(rows); tables["T9_gain_per_detector_level4"] = T9
    lines += ["\nT9  gain from TDA-90 per detector at level 4", T9.to_string(index=False)]

    tdir = v2 / "tables"; tdir.mkdir(exist_ok=True)
    for n, t in tables.items():
        t.to_csv(tdir / f"{n}.csv", index=False)
    text = "\n".join(lines)
    (v2 / "stats.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
