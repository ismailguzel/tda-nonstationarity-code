"""Realised share of GARCH(1,1) series with alpha + beta >= 1 (Appendix A note). Read-only."""
import glob, json
from pathlib import Path
import pandas as pd
D = str(Path(__file__).resolve().parents[2] / "data/paper/raw")
cols = ["series_id", "time", "volatility_type", "volatility_alpha", "volatility_beta"]
M = pd.concat([pd.read_parquet(f, columns=cols).query("time == time.min()").assign(part=f[-14:-8])
               for f in sorted(glob.glob(D + "/all72?.parquet"))])
assert len(M) == 36000, len(M)
for c in ("volatility_alpha", "volatility_beta"):
    M[c] = pd.to_numeric(M[c].astype(str).replace({"nan": None, "None": None}), errors="coerce")
G = M[M.volatility_type == "garch"]
s = G.volatility_alpha + G.volatility_beta
out = {"n_garch": int(len(G)), "n_missing": int(s.isna().sum()),
       "n_ge_1": int((s >= 1).sum()), "share_ge_1": float((s >= 1).mean()),
       "sum_min": float(s.min()), "sum_median": float(s.median()), "sum_max": float(s.max())}
print(json.dumps(out, indent=1)); open("garch_share.json", "w").write(json.dumps(out, indent=1))
