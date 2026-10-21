"""Choose the Figure 1 series by a stated rule and compute what the figure shows.
Rule: from part all72a (level 0-1 classes), base AR, GARCH innovations for the volatility
series. Volatility series = the one whose stored pwin_H0__W1__std (the feature that carries
the V detector) is closest to the 90th percentile of the GARCH-AR volatility series, among
those with max |z| < 4.5 (no single spike dominates the picture). Stationary series = AR
series closest to the median of the same feature. Writes fig1_data.npz and fig1_select.txt."""
import sys, numpy as np, pandas as pd
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO / "src"))
from topology import _zscore, windows, paper_features
from ripser import ripser
D = REPO / "data/paper"
R = pd.read_parquet(D / "raw/all72a.parquet", columns=["series_id", "time", "data", "primary_category", "base_components", "volatility_type"])
F = pd.read_parquet(D / "features/all72a/tda90.parquet")
meta = R[R.time == R.time.min()].set_index("series_id")
log = []
def say(*a):
    s = " ".join(map(str, a)); print(s); log.append(s)
say("volatility_type values:", meta.volatility_type.astype(str).value_counts().to_dict())
say("base_components examples:", meta.base_components.astype(str).value_counts().head(6).to_dict())
X = {sid: g.sort_values("time")["data"].to_numpy(float) for sid, g in R.groupby("series_id")}
zmax = pd.Series({sid: np.abs(_zscore(x)).max() for sid, x in X.items()})
feat = F["pwin_H0__W1__std"]
base_ar = meta.base_components.astype(str).str.contains("'ar'|\"ar\"|^ar$|\\bar\\b", regex=True) & ~meta.base_components.astype(str).str.contains("arma")
garch = meta.volatility_type.astype(str).str.lower().eq("garch")
V = meta.index[(meta.primary_category == "volatility") & garch & base_ar]
S = meta.index[(meta.primary_category == "stationary") & base_ar]
say("candidates: volatility GARCH-AR", len(V), " stationary AR", len(S))
if len(V) == 0:
    V = meta.index[meta.primary_category == "volatility"]; say("fallback: all volatility series", len(V))
if len(S) == 0:
    S = meta.index[meta.primary_category == "stationary"]; say("fallback: all stationary series", len(S))
q90 = feat.loc[V].quantile(0.9); okV = [s for s in V if zmax[s] < 4.5]
sv = min(okV, key=lambda s: abs(feat[s] - q90))
med = feat.loc[S].median(); ss = min(S, key=lambda s: abs(feat[s] - med))
say(f"chosen volatility series {sv}: feature {feat[sv]:.3f} (q90 {q90:.3f}), max|z| {zmax[sv]:.2f}, base {meta.base_components[sv]}")
say(f"chosen stationary series {ss}: feature {feat[ss]:.3f} (median {med:.3f}), max|z| {zmax[ss]:.2f}, base {meta.base_components[ss]}")
say("percentile of the stationary series' feature among volatility series:", round((feat.loc[V] < feat[ss]).mean(), 2),
    "; of the volatility series among stationary series:", round((feat.loc[S] < feat[sv]).mean(), 2))
out = {"volatility_id": sv, "stationary_id": ss}
for key, sid in (("volatility", sv), ("stationary", ss)):
    z = _zscore(X[sid]); w = len(z) // 5; W = windows(len(z), w, w // 2)
    out[f"{key}_x"] = X[sid]; out[f"{key}_z"] = z; feats = []
    for i, (a, b) in enumerate(W):
        v = z[a:b]; P = np.column_stack([v[k:k + len(v) - 2] for k in range(3)])
        dg = ripser(P, maxdim=1)["dgms"]
        out[f"{key}_w{i}_H0"] = dg[0]; out[f"{key}_w{i}_H1"] = dg[1]; out[f"{key}_w{i}_cloud"] = P
        feats.append(paper_features(dg[0]))
    out[f"{key}_H0feats"] = np.array(feats)
out["windows"] = np.array(W)
w1 = out["volatility_H0feats"][:, 0]
say("volatility series, window H0 W1:", np.round(w1, 2).tolist(), "calm window", int(w1.argmin()) + 1, "turbulent window", int(w1.argmax()) + 1)
say("stationary series, window H0 W1:", np.round(out["stationary_H0feats"][:, 0], 2).tolist())
np.savez_compressed("fig1_data.npz", **out); Path("fig1_select.txt").write_text("\n".join(log) + "\n")
