"""Exploratory (Section 5.4 of the paper): classical statistical tests
used as detectors of the eight causes, scored exactly like the learned detectors.

One standard test per cause, default specification, decision at the 5% level, applied to the
standardised series (Section 4.3 of the paper). Every test assumes that the causes it does not
target are absent; that is the situation the paper's introduction describes, and it is what
this check measures.

  S  augmented Dickey-Fuller, constant, lag by AIC (statsmodels); S if unit root not rejected.
     Variant S_ct: constant and trend.
  V  Engle ARCH-LM, 5 lags, on the residuals of an AR(5) fit.
  T  slope of a regression on time, Newey-West (HAC) Wald test.
  R  sup-LM test for a change of slope at an unknown date (Andrews 1993), trimming 15%,
     HAC variance; 5% critical value and p value from the null distribution of the same
     statistic simulated with i.i.d. Gaussian errors (its asymptotic null distribution).
  M  OLS-CUSUM (Ploberger-Kraemer) on the demeaned series, long-run variance by Newey-West,
     p value from the Kolmogorov distribution.
  Q  CUSUM of squares of the AR(5) residuals (Inclan-Tiao), robust to serial dependence of the
     squares (long-run variance, as in Sanso, Arago and Carrion 2004), Kolmogorov p value.
  P  Grubbs test for one outlier on the AR(5) residuals.
  C  no standard single test: not assessed.

Folds: the same 5-fold assignment as src/detectors.py (StratifiedKFold, shuffle, seed 42, on
the class labels of data/paper/features/*/labels.parquet). Tests are not trained; each series
is scored in the fold that tests it, so every comparison with the learned detectors is paired
by fold and series.

Outputs: folds_tests.csv (same columns as results/paper/main/folds.csv), decisions.parquet,
critical_values.json, stat_tests.log.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from detectors import LETTERS, labels  # noqa: E402  same label construction as the paper

warnings.filterwarnings("ignore")
N, ALPHA, P_AR, SEED = 500, 0.05, 5, 42
NW_LAGS = int(np.floor(4 * (N / 100) ** (2 / 9)))           # 5 for n = 500
TT = np.arange(N, dtype=float)
X0 = np.column_stack([np.ones(N), TT])
H0 = X0 @ np.linalg.solve(X0.T @ X0, X0.T)                  # projection on [1, t]
TAUS = np.arange(int(0.15 * N), int(0.85 * N) + 1)
W = np.maximum(TT[:, None] - TAUS[None, :], 0.0)            # kink regressors (t - tau)_+
U = W - H0 @ W                                              # residualised on [1, t]
UT = TT - TT.mean()


def lrv(S: np.ndarray) -> np.ndarray:
    """Newey-West long-run variance of each column (Bartlett kernel, NW_LAGS lags)."""
    S = S if S.ndim == 2 else S[:, None]
    n = len(S)
    v = (S * S).sum(0) / n
    for k in range(1, NW_LAGS + 1):
        v = v + 2 * (1 - k / (NW_LAGS + 1)) * (S[k:] * S[:-k]).sum(0) / n
    return np.maximum(v, 1e-12)


def ar_resid(z: np.ndarray, p: int = P_AR) -> np.ndarray:
    Y = z[p:]
    X = np.column_stack([np.ones(len(Y))] + [z[p - k:-k] for k in range(1, p + 1)])
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return Y - X @ b


def sup_lm_slope(z: np.ndarray) -> float:
    e0 = z - H0 @ z
    s = U * e0[:, None]
    return float(((s.sum(0) ** 2) / (N * lrv(s))).max())


def one(z: np.ndarray) -> dict:
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import het_arch
    out = {}
    out["S_p"] = adfuller(z, regression="c", autolag="AIC")[1]
    out["S_ct_p"] = adfuller(z, regression="ct", autolag="AIC")[1]
    e = ar_resid(z)
    lm, lm_p = het_arch(e, nlags=5)[:2]
    out["V_stat"], out["V_p"] = lm, lm_p
    b = UT @ z / (UT @ UT)
    eT = z - H0 @ z
    wT = b ** 2 * (UT @ UT) ** 2 / (N * lrv(UT * eT)[0])
    out["T_stat"], out["T_p"] = wT, stats.chi2.sf(wT, 1)
    out["R_stat"] = sup_lm_slope(z)
    ed = z - z.mean()
    cm = np.abs(np.cumsum(ed)).max() / np.sqrt(N * lrv(ed)[0])
    out["M_stat"], out["M_p"] = cm, stats.kstwobign.sf(cm)
    x = e ** 2 - (e ** 2).mean()
    cq = np.abs(np.cumsum(x)).max() / np.sqrt(len(x) * lrv(x)[0])
    out["Q_stat"], out["Q_p"] = cq, stats.kstwobign.sf(cq)
    m = len(e)
    G = np.abs(e - e.mean()).max() / e.std(ddof=1)
    t2 = m * (m - 2) * G ** 2 / max((m - 1) ** 2 - m * G ** 2, 1e-12)
    out["P_stat"], out["P_p"] = G, min(1.0, 2 * m * stats.t.sf(np.sqrt(t2), m - 2))
    return out


def main() -> None:
    t0 = time.time()
    feats, raw = REPO / "data/paper/features", REPO / "data/paper/raw"
    parts = sorted(p.name for p in feats.iterdir() if (p / "labels.parquet").exists())
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    Ls, Zs = [], []
    for p in parts:
        L = pd.read_parquet(feats / p / "labels.parquet")
        R = pd.read_parquet(raw / f"{p}.parquet", columns=["series_id", "time", "data"]).sort_values(["series_id", "time"])
        A = np.stack([g.to_numpy(float) for _, g in R.groupby("series_id")["data"]])
        ids = np.array(sorted(R.series_id.unique()))
        A = A[pd.Index(ids).get_indexer(L.index)]                 # align with the labels' order
        assert A.shape == (len(L), N) and (pd.Index(ids).get_indexer(L.index) >= 0).all()
        Ls.append(L.assign(part=p)); Zs.append(A)
    L = pd.concat(Ls); Z = np.vstack(Zs)
    sd = Z.std(1, keepdims=True)
    Z = (Z - Z.mean(1, keepdims=True)) / np.where(sd > 0, sd, 1.0)
    cls = L["primary_category"].to_numpy()
    Y = labels(cls)
    lev = np.array([0 if c == "stationary" else c.count("+") + 1 for c in cls])
    print(f"{len(cls)} series, {len(set(cls))} classes; loaded in {time.time() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(SEED)
    null = np.array([sup_lm_slope(rng.standard_normal(N)) for _ in range(5000)])
    crit_R = float(np.quantile(null, 1 - ALPHA))
    print(f"sup-LM slope-break null: 5% critical value {crit_R:.3f} ({time.time() - t0:.0f}s)", flush=True)

    idx = np.arange(len(Z)) if limit is None else np.arange(limit)
    res = Parallel(n_jobs=12, batch_size=64)(delayed(one)(Z[i]) for i in idx)
    D = pd.DataFrame(res)
    D["R_p"] = [(null >= s).mean() for s in D.R_stat]
    print(f"tests done in {time.time() - t0:.0f}s", flush=True)

    # decision (1 = cause present) and score (larger = more evidence that it is present)
    dec = {"S": D.S_p >= ALPHA, "S_ct": D.S_ct_p >= ALPHA, "V": D.V_p < ALPHA, "T": D.T_p < ALPHA,
           "R": D.R_stat > crit_R, "M": D.M_p < ALPHA, "Q": D.Q_p < ALPHA, "P": D.P_p < ALPHA}
    score = {"S": D.S_p, "S_ct": D.S_ct_p, "V": D.V_stat, "T": D.T_stat, "R": D.R_stat,
             "M": D.M_stat, "Q": D.Q_stat, "P": D.P_stat}
    out = L.iloc[idx][["part", "primary_category"]].reset_index().assign(level=lev[idx])
    for k in dec:
        out[f"{k}_dec"] = dec[k].to_numpy().astype(int)
    out = pd.concat([out, D.reset_index(drop=True)], axis=1)

    rows = []
    if limit is None:
        out["fold"] = -1
        for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(cls, cls)):
            out.loc[te, "fold"] = f
            for k in dec:
                j = LETTERS.index(k[0])
                for lv in sorted(set(lev[te])):
                    m = te[lev[te] == lv]
                    y = Y[m, j]
                    if len(set(y)) == 2:
                        setname = "Statistical tests" if k != "S_ct" else "Statistical tests (ADF with trend)"
                        rows.append(dict(set=setname, model="5% level", level=lv, char=k[0], fold=f,
                                         bal_acc=balanced_accuracy_score(y, dec[k].to_numpy()[m]),
                                         auc=roc_auc_score(y, score[k].to_numpy()[m])))
    pd.DataFrame(rows).to_csv("folds_tests.csv", index=False)
    out.to_parquet("decisions.parquet")
    Path("critical_values.json").write_text(json.dumps({"R_sup_lm_5pct": crit_R, "null_reps": len(null),
                                                         "nw_lags": NW_LAGS, "ar_order": P_AR}, indent=1))
    if rows:
        F = pd.DataFrame(rows)
        print(F[F.set == "Statistical tests"].groupby(["char", "level"]).bal_acc.mean().unstack().round(3).to_string())
    else:
        print(out.filter(like="_dec").mean().round(3).to_string())
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
