"""Cross-generator check: detectors fitted once on all 36 000 BeTiSe series are applied,
unchanged, to series generated with GutenTAG by xgen_generate_2026-11-15.py (arm A: bases with
labels known by construction; arm C: periodic and shape-based bases, descriptive).
Inputs (git-ignored): data/external/xgen/{series.parquet,labels.csv}. Outputs: results/paper/xgen/.

    python experiments/checks/xgen_2026-11-15.py [--smoke | --rescore]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "exploratory"))
from detectors import LETTERS, labels, load, make  # noqa: E402
from features import tsfresh_features  # noqa: E402
from topology import tda90, tda90_names  # noqa: E402
import stat_tests  # noqa: E402  classical tests exactly as in the paper

warnings.filterwarnings("ignore")
N, SEED, ALPHA = 500, 42, 0.05
SETS = ["TSFresh-774", "TSFresh-774+TDA-90"]
MODELS = ["hgb", "lr"]


def extract(Z: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    n = len(Z)
    ts = pd.DataFrame({"series_id": np.repeat(np.arange(n), Z.shape[1]), "time": np.tile(np.arange(Z.shape[1]), n),
                       "data": Z.ravel()})
    F = tsfresh_features(ts)
    assert (F.index.to_numpy() == np.arange(n)).all()
    T = np.array(Parallel(n_jobs=-1, batch_size=16)(delayed(tda90)(x) for x in Z))
    return F, T


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (c - h, c + h)


def rate(dec: np.ndarray) -> str:
    k, n = int(dec.sum()), len(dec)
    lo, hi = wilson(k, n)
    return f"{k:3d}/{n:3d} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]" if n else "  0/  0"


IN = REPO / "data/external/xgen"
OUT = REPO / "results/paper/xgen"
TESTS = {"S": ("S_p", "ge"), "V": ("V_p", "lt"), "T": ("T_p", "lt"), "M": ("M_p", "lt"), "Q": ("Q_p", "lt"), "P": ("P_p", "lt")}


def ba(d, y):
    se, sp = d[y == 1], ~d[y == 0]
    return (se.mean() + sp.mean()) / 2 if len(se) and len(sp) else np.nan, se, sp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="40 series, 3 000 training series, gradient boosting only")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute stats.txt from the saved probabilities.csv and tests.csv with the current labels.csv")
    a = ap.parse_args()
    if a.rescore:
        return report(pd.read_csv(IN / "labels.csv"), pd.read_csv(OUT / "probabilities.csv"), pd.read_csv(OUT / "tests.csv"),
                      None, OUT, time.time())
    out = OUT if not a.smoke else OUT / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta = pd.read_csv(IN / "labels.csv")
    Zdf = pd.read_parquet(IN / "series.parquet")
    assert list(Zdf.index) == list(meta["id"])
    if a.smoke:
        keep = meta.groupby(["arm", "base"]).head(3).index
        meta, Zdf = meta.loc[keep].reset_index(drop=True), Zdf.iloc[keep]
    Z = Zdf.to_numpy(float)
    meta.to_csv(out / "windows.csv", index=False)
    print(f"{len(meta)} series", flush=True)

    parts, key, cls, X, names = load(REPO / "data/paper/features", SETS)
    Y = labels(cls)
    if a.smoke:
        sub = np.random.default_rng(SEED).choice(len(cls), 3000, replace=False)
        X, Y = {s: v[sub] for s, v in X.items()}, Y[sub]
    F, T = extract(Z)
    Ft = np.nan_to_num(F[names["TSFresh-774"]].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    Tt = np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)
    Xr = {"TSFresh-774": Ft, "TSFresh-774+TDA-90": np.hstack([Ft, Tt])}
    pd.DataFrame(Ft, index=meta["id"], columns=names["TSFresh-774"]).to_parquet(out / "features_tsfresh.parquet")
    pd.DataFrame(Tt, index=meta["id"], columns=tda90_names()).to_parquet(out / "features_tda90.parquet")
    print(f"features ({time.time() - t0:.0f}s)", flush=True)

    def fit(model, s, j):
        m = make(model, SEED).fit(X[s], Y[:, j])
        return model, s, j, m.predict_proba(Xr[s])[:, 1]

    P = meta[["id"]].copy()
    for model in (["hgb"] if a.smoke else MODELS):
        t = time.time()
        res = Parallel(n_jobs=8 if model == "lr" else 1)(delayed(fit)(model, s, j) for s in SETS for j in range(len(LETTERS)))
        for model_, s, j, p in res:
            P[f"{s}|{model_}|{LETTERS[j]}"] = p
        print(f"fitted {model} ({time.time() - t:.0f}s)", flush=True)
    P.to_csv(out / "probabilities.csv", index=False)

    Zs = (Z - Z.mean(1, keepdims=True)) / Z.std(1, keepdims=True)
    tests = pd.DataFrame(Parallel(n_jobs=-1)(delayed(stat_tests.one)(z) for z in Zs))
    tests.insert(0, "id", meta["id"].to_numpy())
    tests.to_csv(out / "tests.csv", index=False)

    report(meta, P, tests, len(Y), out, t0)


def report(meta, P, tests, n_train, out, t0) -> None:
    assert list(P["id"]) == list(meta["id"]) == list(tests["id"])
    D = {c: (P[c].to_numpy() > 0.5) for c in P.columns if c != "id"}
    for ch, (col, how) in TESTS.items():
        D[f"test|{ch}"] = tests[col].to_numpy() >= ALPHA if how == "ge" else tests[col].to_numpy() < ALPHA
    combos = [c for c in sorted({c.rsplit("|", 1)[0] for c in D}) if c != "test"] + ["test"]
    arm, base, inj, nev = (meta[k].to_numpy() for k in ("arm", "base", "injection", "n_events"))
    L = [f"=== cross-generator check (GutenTAG); {len(meta)} series of {N}; detectors fitted on {n_train or 36000} BeTiSe series",
         "    decision: probability > 0.5 (tests: 5 %); rates with Wilson 95% intervals", ""]

    A = arm == "A"
    L.append(f"X1  Arm A (GutenTAG, defined labels): {int(A.sum())} series; R on LT base only")
    for ch in LETTERS:
        m = A & (base == "LT") if ch == "R" else A
        y = meta[ch].to_numpy(float)[m]
        L.append(f"  {ch}  positives {int((y == 1).sum())}, negatives {int((y == 0).sum())}")
        for cm in combos:
            k = f"{cm}|{ch}"
            if k not in D:
                continue
            b, se, sp = ba(D[k][m], y)
            L.append(f"    {cm:<26} sensitivity {rate(se)}   specificity {rate(sp)}   BA {b:.3f}")
    L.append("")
    L.append("X2  Arm A: share flagged by each detector, per base and injection (TSFresh-774+TDA-90 | hgb; test row below each)")
    L.append(f"    {'cell':<16} n   " + "  ".join(f"{x:>5}" for x in LETTERS))
    for bb in ["WN", "RW", "LT"]:
        for ii in ["none", "M", "Q", "P", "C", "Cplat", "R"]:
            m = A & (base == bb) & (inj == ii)
            if not m.any():
                continue
            L.append(f"    {bb + ' ' + ii:<16} {int(m.sum()):3d} " + "  ".join(f"{D[f'TSFresh-774+TDA-90|hgb|{x}'][m].mean():5.2f}" for x in LETTERS))
            L.append(f"    {'  test':<16}     " + "  ".join(f"{D[f'test|{x}'][m].mean():5.2f}" if f"test|{x}" in D else "    -" for x in LETTERS))
    L.append("")
    L.append("X3  Arm A: balanced accuracy, TSFresh-774+TDA-90 minus TSFresh-774 (descriptive, no test)")
    for model in sorted({c.split("|")[1] for c in D if c.startswith("TSFresh-774|")}):
        row = []
        for ch in LETTERS:
            m = A & (base == "LT") if ch == "R" else A
            y = meta[ch].to_numpy(float)[m]
            b1 = ba(D[f"TSFresh-774+TDA-90|{model}|{ch}"][m], y)[0]
            b0 = ba(D[f"TSFresh-774|{model}|{ch}"][m], y)[0]
            row.append(f"{ch} {b1 - b0:+.3f}" if np.isfinite(b1 - b0) else f"{ch}   n/a")
        L.append(f"  {model}: " + "  ".join(row))
    L.append("")
    C = arm == "C"
    L.append("X5  Arm C (GutenTAG bases outside the generator's scope): share flagged (TSFresh-774+TDA-90 | hgb)")
    L.append(f"    {'cell':<30} n   " + "  ".join(f"{x:>5}" for x in LETTERS))
    for bb in ["sine", "ecg", "cylinder-bell-funnel"]:
        for ii in ["none", "M", "P", "C"]:
            m = C & (base == bb) & (inj == ii)
            if m.any():
                L.append(f"    {bb + ' ' + ii:<30} {int(m.sum()):3d} " + "  ".join(f"{D[f'TSFresh-774+TDA-90|hgb|{x}'][m].mean():5.2f}" for x in LETTERS))
    L += ["", f"total time {time.time() - t0:.0f}s"]
    (out / "stats.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
