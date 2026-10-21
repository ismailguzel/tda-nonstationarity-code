"""
The eight detectors as a diagnosis: "this series contains S + T + M, with these confidences".
Exploratory set-level diagnosis (Section 5.6 of the paper). Uses only the main run's
out-of-fold scores (oof.parquet); nothing is refitted.

For every feature set under logistic regression and gradient boosting (MiniRocket's ridge
gives no probabilities and is left out):

1. Two decisions per series.
     independent   each detector says yes when its probability exceeds 0.5
     constrained   the most probable of the 72 combinations that exist in the design:
                   argmax over valid sets of sum_j log p_j (present) or log(1 - p_j) (absent)
   The constrained answer can never be impossible (e.g. a trend and a trend shift together).
2. How right is the answer, per level: exact set; Hamming (share of the eight right);
   missed causes (share of present characteristics not reported); invented causes (share
   of absent ones reported); number of causes right.
3. Stationary or not: a series is called non-stationary when the answer has any
   characteristic. Recall on the 500 stationary series and on the 35 500 others.
4. Calibration of the stated confidence: the eight probabilities pooled, in ten bins —
   mean stated probability against the share actually present; expected calibration error
   (ECE) and Brier score.
5. Example profiles of a few series.

    python src/diagnose.py --run runs/<tag>/main --out runs/<tag>/diagnose
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LETTERS = ["S", "V", "T", "R", "M", "Q", "P", "C"]
NAME = {"stochastic_trend": "S", "volatility": "V", "trend": "T", "trend_shift": "R",
        "mean_shift": "M", "variance_shift": "Q", "point_anomaly": "P", "collective_anomaly": "C"}


def truth(cls: pd.Series) -> np.ndarray:
    return np.array([[int(l in ({NAME[p] for p in c.split("+")} if c != "stationary" else set())) for l in LETTERS]
                     for c in cls])


def code(row) -> str:
    s = "".join(l for l, v in zip(LETTERS, row) if v)
    return s or "stationary"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--focus", default="TDA-90", help="set used for the reliability bins and the example profiles")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    O = pd.read_parquet(Path(args.run) / "oof.parquet")
    Y = truth(O["class"])
    lev = O["level"].to_numpy()
    valid = np.unique(Y, axis=0)                                   # the 72 combinations of the design
    keys = sorted({tuple(c.split("|")[:2]) for c in O.columns if c.count("|") == 2 and not c.startswith("MiniRocket")})
    lines = [f"=== Diagnosis from the out-of-fold scores — exploratory, not in the analysis plan; "
             f"{len(O)} series, {len(valid)} valid combinations"]
    acc_rows, stat_rows, cal_rows, bins_rows = [], [], [], []

    for s, m in keys:
        P = np.clip(O[[f"{s}|{m}|{l}" for l in LETTERS]].to_numpy(float), 1e-6, 1 - 1e-6)
        ind = (P > 0.5).astype(int)
        score = np.log(P) @ valid.T + np.log(1 - P) @ (1 - valid).T
        con = valid[score.argmax(1)]
        for rule, D in (("independent", ind), ("constrained", con)):
            for k in sorted(set(lev)):
                sel = lev == k
                t, d = Y[sel], D[sel]
                pos, neg = t.sum(), (1 - t).sum()
                acc_rows.append(dict(set=s, model=m, rule=rule, level=k,
                                     exact=(d == t).all(1).mean(), hamming=(d == t).mean(),
                                     missed=((t == 1) & (d == 0)).sum() / pos if pos else np.nan,
                                     invented=((t == 0) & (d == 1)).sum() / neg if neg else np.nan,
                                     count_right=(d.sum(1) == t.sum(1)).mean()))
            ns_true, ns_pred = Y.any(1), D.any(1)
            stat_rows.append(dict(set=s, model=m, rule=rule,
                                  stationary_recall=(~ns_pred[~ns_true]).mean(), nonstationary_recall=ns_pred[ns_true].mean(),
                                  balanced=0.5 * ((~ns_pred[~ns_true]).mean() + ns_pred[ns_true].mean())))
        p, y = P.ravel(), Y.ravel()
        b = np.minimum((p * 10).astype(int), 9)
        ece = sum(abs(p[b == i].mean() - y[b == i].mean()) * (b == i).mean() for i in range(10) if (b == i).any())
        cal_rows.append(dict(set=s, model=m, ECE=ece, Brier=((p - y) ** 2).mean()))
        for i in range(10):
            if (b == i).any():
                bins_rows.append(dict(set=s, model=m, bin=f"{i / 10:.1f}-{(i + 1) / 10:.1f}", n=int((b == i).sum()),
                                      stated=p[b == i].mean(), observed=y[b == i].mean()))

    A = pd.DataFrame(acc_rows); S = pd.DataFrame(stat_rows); C = pd.DataFrame(cal_rows); B = pd.DataFrame(bins_rows)
    pd.set_option("display.width", 220)
    for metric in ("exact", "hamming", "missed", "invented", "count_right"):
        T = A.pivot_table(index=["set", "model", "rule"], columns="level", values=metric)
        lines += [f"\n  {metric} by level", "  " + T.round(3).to_string().replace("\n", "\n  ")]
    lines += ["\n  stationary or not (non-stationary = any characteristic reported)",
              "  " + S.round(3).to_string(index=False).replace("\n", "\n  "),
              "\n  calibration of the eight probabilities, pooled", "  " + C.round(4).to_string(index=False).replace("\n", "\n  ")]
    for s, m in [kk for kk in keys if kk[0] == args.focus]:
        R = B[(B.set == s) & (B.model == m)]
        lines += [f"\n  reliability, {s} {m}: stated -> observed (n)",
                  "    " + "   ".join(f"{r.stated:.2f}->{r.observed:.2f} ({r.n})" for r in R.itertuples())]

    s, m = args.focus, ("hgb" if (args.focus, "hgb") in keys else "lr")    # examples: the stronger classifier when run
    P = O[[f"{s}|{m}|{l}" for l in LETTERS]].to_numpy(float)
    score = np.log(np.clip(P, 1e-6, 1)) @ valid.T + np.log(np.clip(1 - P, 1e-6, 1)) @ (1 - valid).T
    con = valid[score.argmax(1)]
    rng = np.random.default_rng(0)
    lines.append(f"\n  example profiles, {s} {m}, constrained decision (probability per characteristic)")
    for k in (1, 3, 4):
        for i in rng.choice(np.where(lev == k)[0], max(1, args.examples // 3), replace=False):
            prof = "  ".join(f"{l} {P[i, j]:.2f}" for j, l in enumerate(LETTERS))
            verdict = "right" if (con[i] == Y[i]).all() else "wrong"
            lines.append(f"    true {code(Y[i]):9s} said {code(con[i]):9s} ({verdict})   {prof}")

    A.to_csv(out / "accuracy_by_level.csv", index=False); S.to_csv(out / "stationary_or_not.csv", index=False)
    C.to_csv(out / "calibration.csv", index=False); B.to_csv(out / "reliability_bins.csv", index=False)
    text = "\n".join(lines)
    (out / "diagnose.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
