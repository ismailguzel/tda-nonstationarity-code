"""Compare the R cross-check (r_tests.csv) with the Python run (decisions.parquet): agreement
of decisions and statistics per test; R-based balanced accuracy by level; the cross-
sensitivity of each test to every single cause (level 1)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from detectors import labels, LETTERS
M = pd.read_csv("meta.csv"); R = pd.read_csv("r_tests.csv")
P = pd.read_parquet("decisions.parquet")
assert len(M) == len(R) == len(P) and (M.series_id.to_numpy() == P.series_id.to_numpy()).all() \
    and (M.part.to_numpy() == P.part.to_numpy()).all()
A = 0.05
rdec = {"S": R.S_stat > R.S_cv5, "V": R.V_p < A, "T": R.T_p < A, "R": R.R_stat > R.R_crit,
        "M": R.M_p < A, "Q": R.Q_p < A, "P": R.P_p < A}
pstat = {"V": P.V_stat, "T": P.T_stat, "R": P.R_stat, "M": P.M_stat, "Q": P.Q_stat, "P": P.P_stat}
out = {"R_crit": float(R.R_crit.iloc[0])}
lines = [f"R sup-LM 5% critical value {out['R_crit']:.3f} (Python 5.679)", "",
         "test  decision agreement  python flag rate  R flag rate  max|stat diff|  corr(stat)"]
for k in "SVTRMQP":
    pd_ = P[f"{k}_dec"].astype(bool).to_numpy(); rd = rdec[k].to_numpy()
    agree = (pd_ == rd).mean()
    if k in pstat:
        rs = R[f"{k}_stat"].to_numpy(); ps = pstat[k].to_numpy()
        md, cr = np.nanmax(np.abs(rs - ps)), np.corrcoef(rs, ps)[0, 1]
    else:
        md, cr = np.nan, np.nan
    out[k] = dict(agree=agree, py_rate=pd_.mean(), r_rate=rd.mean(), maxdiff=md, corr=cr)
    lines.append(f"{k:4s}  {agree:18.4f}  {pd_.mean():16.3f}  {rd.mean():11.3f}  {md:14.2e}  {cr:10.5f}")
lines.append(f"\nM, textbook OLS-CUSUM with i.i.d. scale (strucchange): flag rate on the stationary class "
             f"{(R.M_iid_p[M.level == 0] < A).mean():.3f}; LRV version {(R.M_p[M.level == 0] < A).mean():.3f}")
Y = labels(M.primary_category.to_numpy()); lev = M.level.to_numpy(); fold = M.fold.to_numpy()
rows = []
for src, D in (("python", {k: P[f"{k}_dec"].astype(bool).to_numpy() for k in "SVTRMQP"}),
               ("R", {k: v.to_numpy() for k, v in rdec.items()})):
    for k in "SVTRMQP":
        j = LETTERS.index(k)
        for f in range(5):
            for lv in (1, 2, 3, 4):
                m = (fold == f) & (lev == lv); y = Y[m, j]
                if len(set(y)) == 2:
                    rows.append(dict(src=src, char=k, fold=f, level=lv, bal_acc=balanced_accuracy_score(y, D[k][m])))
B = pd.DataFrame(rows)
mean7 = B.groupby(["src", "level", "fold"]).bal_acc.mean().groupby(["src", "level"]).mean().unstack().round(3)
lines += ["", "mean of seven, balanced accuracy by level", mean7.to_string(),
          "", "per cause, balanced accuracy L1 / L4",
          B[B.level.isin([1, 4])].groupby(["src", "char", "level"]).bal_acc.mean().unstack().round(3).to_string()]
# cross-sensitivity at level 1: share flagged by each test, per single-cause class (+ stationary)
cl = M.primary_category.to_numpy()
single = ["stationary", "stochastic_trend", "volatility", "trend", "trend_shift", "mean_shift",
          "variance_shift", "point_anomaly", "collective_anomaly"]
X = pd.DataFrame({k: [P.loc[cl == c, f"{k}_dec"].mean() for c in single] for k in "SVTRMQP"}, index=single)
XR = pd.DataFrame({k: [rdec[k][cl == c].mean() for c in single] for k in "SVTRMQP"}, index=single)
X.to_csv("cross_sensitivity_L1_python.csv"); XR.to_csv("cross_sensitivity_L1_R.csv")
lines += ["", "share flagged by each test, single-cause classes (python)", X.round(3).to_string(),
          "", "same, R", XR.round(3).to_string()]
B.to_csv("bal_acc_python_vs_R.csv", index=False)
open("compare.txt", "w").write("\n".join(lines) + "\n"); json.dump(out, open("compare.json", "w"), indent=1, default=float)
print("\n".join(lines))
