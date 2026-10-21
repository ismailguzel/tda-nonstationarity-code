"""Realised generator settings of data/paper (betise 0.6.0), for the appendix table: per
component, quantiles of the parameters recorded in the series metadata, and betise's own
configured ranges (config/params.json). Read-only. Writes generator_settings.json/.txt."""
import json, glob
import numpy as np, pandas as pd
from pathlib import Path
import betise
REPO = Path(__file__).resolve().parents[2]; D = REPO / "data/paper/raw"
P = json.loads((Path(betise.__file__).parent / "config" / "params.json").read_text())
cols = ['series_id','time','primary_category','base_components','ar_order','ma_order','ar_coefs','ma_coefs','trend_slope',
        'trend_intercept','stochastic_type','difference','volatility_type','volatility_alpha','volatility_beta','volatility_omega',
        'volatility_theta','volatility_lambda','volatility_gamma','volatility_delta','anomaly_shapes','anomaly_indices',
        'anomaly_magnitudes','break_indices','break_magnitudes','trend_shift_change_types']
M = pd.concat([pd.read_parquet(f, columns=cols).query("time == time.min()") for f in sorted(glob.glob(str(D / "all72?.parquet")))])
assert len(M) == 36000, len(M)
for c_ in ['ar_order','ma_order','trend_slope','trend_intercept','difference','volatility_alpha','volatility_beta','volatility_omega',
           'volatility_theta','volatility_lambda','volatility_gamma','volatility_delta']:
    M[c_] = pd.to_numeric(M[c_].astype(str).replace({"nan": None, "None": None}), errors="coerce")
J = lambda s: json.loads(s) if isinstance(s, str) and s[:1] in "[{" else None
def q(x):
    x = np.asarray([v for v in x if v == v], float)
    return None if len(x) == 0 else dict(n=int(len(x)), min=float(x.min()), q05=float(np.quantile(x, .05)),
                                         median=float(np.median(x)), q95=float(np.quantile(x, .95)), max=float(x.max()))
out = {"betise_version": betise.__version__, "n_series": len(M), "configured": P}
base = M.base_components.map(lambda s: tuple(J(s) or []))
out["base_counts"] = {"+".join(k): int(v) for k, v in base.value_counts().items()}
ar = M[M.ar_order.notna()]
out["ar_order"] = {str(int(k)): int(v) for k, v in ar.ar_order.value_counts().sort_index().items()}
out["ma_order"] = {str(int(k)): int(v) for k, v in M.ma_order.dropna().value_counts().sort_index().items()}
out["ar_coef"] = q([c for s in ar.ar_coefs for c in (J(s) or [])])
out["ma_coef"] = q([c for s in M.ma_coefs.dropna() for c in (J(s) or [])])
out["ari_difference"] = {str(k): int(v) for k, v in M.difference.dropna().value_counts().items()}
out["stochastic_type"] = {str(k): int(v) for k, v in M.stochastic_type.dropna().value_counts().items()}
for vt in ("garch", "egarch"):
    m = M[M.volatility_type == vt]
    out[vt] = {c.replace("volatility_", ""): q(m[c]) for c in ("volatility_alpha", "volatility_beta", "volatility_omega",
                                                              "volatility_theta", "volatility_lambda", "volatility_gamma", "volatility_delta") if m[c].notna().any()}
    out[vt]["n"] = int(len(m))
has = lambda ch: M.primary_category.str.contains(ch)
lt = M[M.trend_slope.notna()]
out["linear_trend"] = {"abs_slope": q(lt.trend_slope.abs()), "share_upward": float((lt.trend_slope > 0).mean()), "intercept": q(lt.trend_intercept)}
for br in ("mean_shift", "variance_shift", "trend_shift"):
    bm = M.break_magnitudes.map(lambda s: (J(s) or {}).get(br, [np.nan])[0])
    bi = M.break_indices.map(lambda s: (J(s) or {}).get(br, [np.nan])[0])
    out[br] = {"magnitude": q(bm), "abs_magnitude": q(bm.abs()), "share_positive": float((bm.dropna() > 0).mean()),
               "position": q(bi / 500)}
out["trend_shift"]["change_types"] = {str(k): int(v) for k, v in M.trend_shift_change_types.dropna().value_counts().items()}
pa = M.anomaly_indices.map(lambda s: (J(s) or {}).get("point_anomaly", [np.nan]))
out["point_anomaly"] = {"position": q([v[0] / 500 for v in pa if isinstance(v, list)]), "magnitude_recorded": False}
ca = M.anomaly_indices.map(lambda s: (J(s) or {}).get("collective_anomaly"))
cm = M.anomaly_magnitudes.map(lambda s: (J(s) or {}).get("collective_anomaly", [np.nan])[0])
sh = M.anomaly_shapes.map(lambda s: tuple((J(s) or {}).get("collective_anomaly", [])))
cav = [d for d in ca if isinstance(d, dict)]
out["collective_anomaly"] = {"length": q([e - s for d in cav for s, e in zip(d["starts"], d["ends"])]),
                             "position": q([s / 500 for d in cav for s in d["starts"]]),
                             "magnitude": q(cm), "abs_magnitude": q(cm.abs()),
                             "shapes": {"+".join(k): int(v) for k, v in sh[sh.map(len) > 0].value_counts().items()}}
Path("generator_settings.json").write_text(json.dumps(out, indent=1))
Path("generator_settings.txt").write_text(json.dumps({k: v for k, v in out.items() if k != "configured"}, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "configured"}, indent=1)[:9000])
