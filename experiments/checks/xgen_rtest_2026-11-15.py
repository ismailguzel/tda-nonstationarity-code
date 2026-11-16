"""Classical reference for trend reversals in the cross-generator check: the paper's sup-LM
trend-shift test (stored 5 % critical value, results/paper/tests/critical_values.json) applied to
the linear-trend series of arm A. Output: results/paper/xgen/r_test.txt."""
import json
from pathlib import Path
import pandas as pd
R = Path(__file__).resolve().parents[2]
t = pd.read_csv(R / "results/paper/xgen/tests.csv")
w = pd.read_csv(R / "results/paper/xgen/windows.csv")
crit = json.loads((R / "results/paper/tests/critical_values.json").read_text())["R_sup_lm_5pct"]
a = w.merge(t, on="id").query("arm == 'A' and base == 'LT'")
f = a["R_stat"] > crit
se, sp = f[a.R == 1].mean(), (~f[a.R == 0]).mean()
L = [f"sup-LM trend-shift test, 5% critical value {crit:.3f}; arm A, linear-trend base ({len(a)} series)",
     f"sensitivity {int(f[a.R == 1].sum())}/{int((a.R == 1).sum())}  specificity {int((~f[a.R == 0]).sum())}/{int((a.R == 0).sum())}  BA {(se + sp) / 2:.3f}",
     "share flagged by injection: " + str(a.assign(f=f).groupby("injection")["f"].mean().round(2).to_dict())]
(R / "results/paper/xgen/r_test.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
