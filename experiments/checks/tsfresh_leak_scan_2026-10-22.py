"""TSFresh leakage scan (sensitivity check of the TSFresh imputation), read-only on the data.

1. Invariance test: for a sample of raw series, TSFresh is extracted (exactly as in
   src/features.py: z-score with pandas std, EfficientFCParameters, unit-root tests dropped)
   on x, on 1000*x + 500 and on 0.001*x - 7, WITHOUT imputation. In exact arithmetic every
   feature of the z-scored series is unchanged; a column that changes carries the raw level or
   scale of the series through floating-point residue.
2. Non-finite columns: NaN/inf rate before imputation, by class and by part.
3. For flagged columns, on the stored features of all 36 000 series (as the detectors see
   them): single-feature AUC (max(AUC, 1-AUC)) for each cause and for each generation part
   (one-vs-rest), and Spearman correlation with log raw SD and raw |mean|/SD.
"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from features import LEAKAGE_PATTERNS          # noqa: E402
from detectors import labels, LETTERS          # noqa: E402
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

FEAT, RAW = REPO / "data/paper/features", REPO / "data/paper/raw"
parts = sorted(p.name for p in FEAT.iterdir() if (p / "labels.parquet").exists())
frames = [pd.read_parquet(FEAT / p / "tsfresh.parquet") for p in parts]
cols = sorted(set.intersection(*(set(f.columns) for f in frames)))
print(f"parts {parts}; TSFresh columns common to all parts: {len(cols)}", flush=True)

def extract(ts):
    g = ts.groupby("series_id")["data"]
    ts = ts.assign(data=(ts["data"] - g.transform("mean")) / g.transform("std").replace(0, 1.0))
    F = extract_features(ts, column_id="series_id", column_sort="time", column_value="data",
                         default_fc_parameters=EfficientFCParameters(), disable_progressbar=True, pivot=True, n_jobs=8)
    F = F.sort_index()
    return F[[c for c in F.columns if not any(p in c.lower() for p in LEAKAGE_PATTERNS)]]

rng = np.random.default_rng(42)
samp, raws = [], {}
for p in parts:
    R = pd.read_parquet(RAW / f"{p}.parquet", columns=["series_id", "time", "data", "primary_category"]).sort_values(["series_id", "time"])
    raws[p] = R
    pick = rng.choice(R["series_id"].unique(), 50, replace=False)
    samp.append(R[R["series_id"].isin(pick)].assign(series_id=lambda d: p + ":" + d["series_id"].astype(str))[["series_id", "time", "data"]])
S = pd.concat(samp)
t = time.time()
F0 = extract(S)[cols]
F1 = extract(S.assign(data=S["data"] * 1000 + 500))[cols]
F2 = extract(S.assign(data=S["data"] * 0.001 - 7))[cols]
print(f"invariance sample: {F0.shape[0]} series, 3 extractions ({time.time() - t:.0f}s)", flush=True)

def changed(A, B):
    a, b = A.to_numpy(float), B.to_numpy(float)
    both_nan = np.isnan(a) & np.isnan(b)
    diff = ~np.isclose(a, b, rtol=1e-6, atol=1e-9) & ~both_nan
    return diff.mean(0)
inv = pd.DataFrame({"frac_changed_x1000": changed(F0, F1), "frac_changed_x0.001": changed(F0, F2),
                    "nonfinite_rate_sample": (~np.isfinite(F0.to_numpy(float))).mean(0)}, index=cols)
flag = inv[(inv.iloc[:, :2].max(1) > 0) | (inv["nonfinite_rate_sample"] > 0)].copy()
print(f"\nColumns not invariant to level/scale of the raw series, or non-finite: {len(flag)}")
print(flag.sort_values("frac_changed_x1000", ascending=False).to_string())

# all-series checks on stored features
L = pd.concat([pd.read_parquet(FEAT / p / "labels.parquet").assign(part=p) for p in parts])
X = pd.concat([f[cols] for f in frames]).reset_index(drop=True)
X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
Y = labels(L["primary_category"].to_numpy())
part = L["part"].to_numpy()
rawstat = pd.concat([raws[p].groupby("series_id")["data"].agg(["mean", "std"]).loc[L[L.part == p].index] for p in parts])
lsd = np.log(rawstat["std"].to_numpy()); rmean = np.abs(rawstat["mean"].to_numpy()) / rawstat["std"].to_numpy()

def a(y, s):
    if np.nanstd(s) == 0: return 0.5
    v = roc_auc_score(y, s); return max(v, 1 - v)
rows = []
for c in flag.index:
    s = X[c].to_numpy(float)
    r = {"column": c, "n_distinct": int(pd.Series(s).nunique()), "sd": float(np.std(s))}
    for j, l in enumerate(LETTERS): r[f"AUC_{l}"] = a(Y[:, j], s)
    r["AUC_part_max"] = max(a((part == p).astype(int), s) for p in parts)
    r["rho_lograwsd"] = spearmanr(s, lsd).correlation
    r["rho_rawmean_over_sd"] = spearmanr(s, rmean).correlation
    rows.append(r)
Rf = pd.DataFrame(rows).set_index("column")
pd.options.display.width = 250
print("\nStored features (36 000 series): single-feature AUC by cause and by part, Spearman with raw scale")
print(Rf.round(3).to_string())
inv.to_csv(REPO / "experiments/checks/tsfresh_leak_scan_2026-10-22_invariance.csv")
Rf.to_csv(REPO / "experiments/checks/tsfresh_leak_scan_2026-10-22_flagged_auc.csv")
# reference: how informative are ordinary columns about the part?
ref = [a((part == p).astype(int), X[c].to_numpy(float)) for c in rng.choice([c for c in cols if c not in flag.index], 40, replace=False) for p in parts[:1]]
print(f"\nReference: part-AUC (part {parts[0]}) of 40 random unflagged columns: median {np.median(ref):.3f}, max {np.max(ref):.3f}")
