"""Original vs TSFresh-771: every TSFresh-dependent hypothesis test, same NB formula as the paper.

    python experiments/checks/tsfresh771_compare_2026-10-22.py   # -> results/paper/sensitivity_tsfresh771/comparison.csv
"""
import sys
import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
ORIG = REPO / "results/paper"
def summ(df, s, model, level, metric):
    d = df[(df.set == s) & (df.model == model) & (df.level == level)]
    return d.groupby("fold")[metric].mean().sort_index().to_numpy()
def nb(x):
    x = np.asarray(x, float); k = len(x); m = x.mean(); s2 = x.var(ddof=1)
    se = np.sqrt((1 / k + 1 / 4) * s2); t = m / se if se > 0 else np.inf
    return m, se, 2 * stats.t.sf(abs(t), k - 1)
def tests(main, comp, bud):
    R = []
    for model in ["lr", "hgb"]:
        for met in ["bal_acc", "auc"]:
            T = lambda s, L, src=main: summ(src, s, model, L, met)
            drop = lambda s, src=main: T(s, 4, src) - T(s, 1, src)
            R.append(("H1 drop TDA-90 - TSFresh", model, met, *nb(drop("TDA-90") - drop("TSFresh-774"))))
            R.append(("H2 L4 TDA-90 - TSFresh", model, met, *nb(T("TDA-90", 4) - T("TSFresh-774", 4))))
            g = lambda L, s="TSFresh-774+TDA-90": summ(comp, s, model, L, met) - T("TSFresh-774", L)
            R.append(("H5 gain TDA-90, L4", model, met, *nb(g(4))))
            R.append(("H6 gain L4 - gain L1", model, met, *nb(g(4) - g(1))))
            R.append(("H7 gain TDA - gain catch22, L4", model, met, *nb(g(4) - g(4, "TSFresh-774+catch22"))))
            b = lambda L: summ(bud, "union top-100", model, L, met) - summ(bud, "TSFresh top-100", model, L, met)
            R.append(("H9 union - TSFresh top-100, L4", model, met, *nb(b(4))))
            R.append(("H10 budget diff L4 - L1", model, met, *nb(b(4) - b(1))))
            for c in "VPTR":
                if met != "bal_acc": continue
                d = lambda s, src: src[(src.set == s) & (src.model == model) & (src.level == 4) & (src.char == c)].sort_values("fold")[met].to_numpy()
                R.append((f"H8 gain {c}, L4", model, met, *nb(d("TSFresh-774+TDA-90", comp) - d("TSFresh-774", main))))
            for s, src in [("TSFresh-774", main), ("TSFresh-774+TDA-90", comp), ("TSFresh-774+catch22", comp), ("TSFresh top-100", bud), ("union top-100", bud)]:
                for L in (1, 4):
                    R.append((f"mean {s} L{L}", model, met, summ(src, s, model, L, met).mean(), np.nan, np.nan))
    return pd.DataFrame(R, columns=["test", "model", "metric", "mean", "se", "p"])
o_main = pd.read_csv(f"{ORIG}/main/folds.csv"); o_comp = pd.read_csv(f"{ORIG}/complementarity/folds.csv"); o_bud = pd.read_csv(f"{ORIG}/budget/folds.csv")
O = tests(o_main, o_comp, o_bud)
J = sys.argv[1] if len(sys.argv) > 1 else ORIG / "sensitivity_tsfresh771"   # output of tsfresh771_sensitivity
n_main = pd.read_csv(f"{J}/main/folds.csv"); n_comp = pd.read_csv(f"{J}/complementarity/folds.csv"); n_bud = pd.read_csv(f"{J}/budget/folds.csv")
n_main = pd.concat([o_main[o_main.set != "TSFresh-774"], n_main])          # TDA-90, catch22, MiniRocket unchanged
N = tests(n_main, n_comp, n_bud)
C = O.merge(N, on=["test", "model", "metric"], suffixes=("_orig", "_771"))
C["diff"] = C["mean_771"] - C["mean_orig"]
sig = lambda p: "" if np.isnan(p) else ("**" if p < 0.01 else "*" if p < 0.05 else "n.s.")
C["sig_orig"] = C.p_orig.map(sig); C["sig_771"] = C.p_771.map(sig)
C["same_decision"] = C.sig_orig.map(lambda s: s != "n.s.") == C.sig_771.map(lambda s: s != "n.s.")
pd.options.display.width = 250
print(C.round(4).to_string())
C.to_csv(f"{J}/comparison.csv", index=False)
t = C[C.test.str.startswith("H")]
print(f"\nhypothesis tests: {len(t)}; same significance decision: {int(t.same_decision.sum())}; max |diff| {t['diff'].abs().max():.4f}")
m = C[C.test.str.startswith("mean")]
print(f"set means: max |diff| {m['diff'].abs().max():.4f}")
