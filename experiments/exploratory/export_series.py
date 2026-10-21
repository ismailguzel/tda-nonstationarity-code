"""Write the standardised series and their metadata for the R cross-check (same order, same
standardisation and same folds as experiments/exploratory/stat_tests.py)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO / "src"))
feats, raw = REPO / "data/paper/features", REPO / "data/paper/raw"
parts = sorted(p.name for p in feats.iterdir() if (p / "labels.parquet").exists())
Ls, Zs = [], []
for p in parts:
    L = pd.read_parquet(feats / p / "labels.parquet")
    R = pd.read_parquet(raw / f"{p}.parquet", columns=["series_id", "time", "data"]).sort_values(["series_id", "time"])
    A = np.stack([g.to_numpy(float) for _, g in R.groupby("series_id")["data"]])
    ids = pd.Index(sorted(R.series_id.unique()))
    A = A[ids.get_indexer(L.index)]
    Ls.append(L.assign(part=p)); Zs.append(A)
L = pd.concat(Ls); Z = np.vstack(Zs)
sd = Z.std(1, keepdims=True); Z = (Z - Z.mean(1, keepdims=True)) / np.where(sd > 0, sd, 1.0)
cls = L["primary_category"].to_numpy()
meta = L.reset_index()[["part", "series_id", "primary_category"]]
meta["level"] = [0 if c == "stationary" else c.count("+") + 1 for c in cls]
meta["fold"] = -1
for f, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(cls, cls)):
    meta.loc[te, "fold"] = f
meta.to_csv("meta.csv", index=False)
np.savetxt("series.csv", Z, delimiter=",", fmt="%.10g")
print(Z.shape, meta.level.value_counts().sort_index().to_dict())
