"""Read-only checks on the paper run (commit f9b8417): (1) TSFresh non-finite cells and the
per-part impute(); (2) logistic-regression convergence in fold 0. Writes ./checks.txt, ./checks.json."""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; DATA = REPO / "data/paper"
sys.path.insert(0, str(REPO / "src"))
out, res = [], {}
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

# ---------------- 1. TSFresh non-finite cells ----------------
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from features import LEAKAGE_PATTERNS
K = 30; rng = np.random.default_rng(0)
parts = sorted(p.name for p in (DATA / "features").iterdir())
NFrows, fills, repro = [], {}, []
t0 = time.time()
for p in parts:
    L = pd.read_parquet(DATA / f"features/{p}/labels.parquet")
    S_all = pd.read_parquet(DATA / f"features/{p}/tsfresh.parquet")
    fills[p] = S_all.median()                                    # what impute() puts in a NaN cell (median of the part)
    ids = np.concatenate([rng.choice(g.index.to_numpy(), K, replace=False) for _, g in L.groupby("primary_category")])
    raw = pd.read_parquet(DATA / f"raw/{p}.parquet", columns=["series_id", "time", "data"])
    raw = raw[raw.series_id.isin(ids)].sort_values(["series_id", "time"])
    g = raw.groupby("series_id")["data"]
    raw = raw.assign(data=(raw["data"] - g.transform("mean")) / g.transform("std").replace(0, 1.0))
    F = extract_features(raw, column_id="series_id", column_sort="time", column_value="data",
                         default_fc_parameters=EfficientFCParameters(), disable_progressbar=True, n_jobs=12).sort_index()
    F = F[S_all.columns]                                         # the 774 stored columns (leakage columns already absent)
    S = S_all.loc[F.index]
    fin = np.isfinite(F.to_numpy())
    repro.append(float(np.nanmax(np.abs(F.to_numpy()[fin] - S.to_numpy()[fin]))))
    A = F.to_numpy()
    rec = pd.DataFrame({"part": p, "series_id": F.index, "cls": L.loc[F.index, "primary_category"].to_numpy()})
    NF = pd.DataFrame(~fin, index=F.index, columns=F.columns)
    NFrows.append((rec, NF, pd.DataFrame(np.isnan(A), index=F.index, columns=F.columns)))
    say(f"{p}: {len(F)} series extracted; NaN {np.isnan(A).mean():.5f}, +inf {np.isposinf(A).mean():.5f}, "
        f"-inf {np.isneginf(A).mean():.5f} of cells; series with >=1 non-finite: {(~fin).any(1).mean():.3f}; "
        f"max |fresh - stored| on finite cells {repro[-1]:.2e} ({time.time()-t0:.0f}s)")
rec = pd.concat([r for r, _, _ in NFrows]); NF = pd.concat([n for _, n, _ in NFrows]); NaNm = pd.concat([n for _, _, n in NFrows])
rec["level"] = [0 if c == "stationary" else c.count("+") + 1 for c in rec.cls]
cols = NF.columns[NF.any()]
say(f"\nOverall: {len(NF)} series, non-finite cell fraction {NF.to_numpy().mean():.5f}; columns with any non-finite: {len(cols)} of {NF.shape[1]}")
FILL = pd.DataFrame(fills)                                       # column x part
spread = ((FILL.max(1) - FILL.min(1)) / (pd.concat([pd.read_parquet(DATA / f"features/{p}/tsfresh.parquet", columns=list(cols)) for p in parts]).std() + 1e-12))
tab = pd.DataFrame({"nonfinite_rate": NF[cols].mean(), "nan_rate": NaNm[cols].mean(),
                    "rate_range_over_classes": NF[cols].groupby(rec.cls.values).mean().pipe(lambda d: d.max() - d.min()),
                    "fill_spread_over_parts_in_sd": spread.reindex(cols)}).sort_values("nonfinite_rate", ascending=False)
by_part = NF[cols].groupby(rec.part.values).mean().T.reindex(tab.index)
say("\nColumns with non-finite values (rate over the sample; class-range; spread of the per-part NaN fill in column SDs):")
say(pd.concat([tab, by_part.add_prefix("rate_")], axis=1).round(4).to_string())
say("\nNon-finite cell fraction by level:", NF.groupby(rec.level.values).mean().mean(1).round(5).to_dict())
res["tsfresh"] = dict(nonfinite_cell_fraction=float(NF.to_numpy().mean()), n_cols=int(len(cols)),
                      series_any=float(NF.any(axis=1).mean()), repro_max=max(repro), table=tab.reset_index().to_dict("records"))

# ---------------- 2. LR convergence, fold 0 ----------------
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
import detectors as D
SETS = ["TDA-90", "TSFresh-774", "catch22", "TSFresh-774+TDA-90", "TSFresh-774+catch22"]
parts_, key, cls, X, names = D.load(DATA / "features", SETS)
Y = D.labels(cls); lev = np.array([0 if c == "stationary" else c.count("+") + 1 for c in cls])
tr, te = next(StratifiedKFold(5, shuffle=True, random_state=42).split(cls, cls))
def fit(s, j, max_iter):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.linear_model import LogisticRegression
    m = make_pipeline(FunctionTransformer(D.signed_log), StandardScaler(),
                      LogisticRegression(C=1.0, class_weight="balanced", max_iter=max_iter))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        m.fit(X[s][tr], Y[tr, j])
    conv_warn = any(issubclass(x.category, ConvergenceWarning) for x in w)
    pr, pb = m.predict(X[s][te]), m.predict_proba(X[s][te])[:, 1]
    sc = {k: (balanced_accuracy_score(Y[te][lev[te] == k, j], pr[lev[te] == k]), roc_auc_score(Y[te][lev[te] == k, j], pb[lev[te] == k]))
          for k in range(1, 5) if len(set(Y[te][lev[te] == k, j])) == 2}
    return dict(set=s, char=D.LETTERS[j], max_iter=max_iter, n_iter=int(m[-1].n_iter_[0]), conv_warning=conv_warn, scores=sc)
rows = []
for s in SETS:
    t = time.time()
    rows += Parallel(n_jobs=8)(delayed(fit)(s, j, 3000) for j in range(8))
    say(f"LR fold 0 {s}: {time.time()-t:.0f}s")
R = pd.DataFrame(rows)
say("\nLR n_iter_ (max_iter 3000), fold 0:"); say(R.pivot(index="set", columns="char", values="n_iter")[D.LETTERS].to_string())
say("ConvergenceWarning:", R.groupby("set").conv_warning.sum().to_dict())
# reproduction against the committed folds
ref = pd.concat([pd.read_csv(REPO / "results/paper/main/folds.csv"), pd.read_csv(REPO / "results/paper/complementarity/folds.csv")])
ref = ref[(ref.model == "lr") & (ref.fold == 0)].drop_duplicates(["set", "char", "level"]).set_index(["set", "char", "level"])
diffs = [abs(r["scores"][k][0] - ref.loc[(r["set"], r["char"], k), "bal_acc"]) for r in rows for k in r["scores"] if (r["set"], r["char"], k) in ref.index]
say(f"Fold-0 LR balanced accuracy vs committed folds.csv: {len(diffs)} cells compared, max |diff| {max(diffs):.2e}")
bad = R[R.conv_warning | (R.n_iter >= 3000)]
if len(bad):
    say(f"\nRefitting {len(bad)} non-converged models with max_iter 20000:")
    re = Parallel(n_jobs=8)(delayed(fit)(r.set, D.LETTERS.index(r.char), 20000) for r in bad.itertuples())
    for a, b in zip(bad.itertuples(), re):
        d = {k: (round(b["scores"][k][0] - a.scores[k][0], 4), round(b["scores"][k][1] - a.scores[k][1], 4)) for k in a.scores}
        say(f"  {a.set} {a.char}: n_iter {b['n_iter']} (warning {b['conv_warning']}); change in (bal_acc, AUC) by level {d}")
    res["lr_refit"] = [dict(set=a.set, char=a.char, new=b) for a, b in zip(bad.itertuples(), re)]
res["lr"] = R.drop(columns="scores").to_dict("records")
Path("checks.txt").write_text("\n".join(out) + "\n"); Path("checks.json").write_text(json.dumps(res, indent=1, default=str))
say("done")
