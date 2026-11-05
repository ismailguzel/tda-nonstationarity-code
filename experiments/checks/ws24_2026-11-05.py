"""Windowed statistical control WS-24 (hypotheses H12-H15 of the paper).

    python experiments/checks/ws24_2026-11-05.py features    # WS-24 of every part -> data/paper/features_ws24/<part>/ws24.parquet
    python experiments/checks/ws24_2026-11-05.py detectors   # WS-24, TSFresh-774+WS-24, TSFresh-774+WS-24+TDA-90 -> results/paper/ws24/
    python experiments/checks/ws24_2026-11-05.py stats       # T14-T18 with Nadeau-Bengio tests -> results/paper/ws24/stats.txt

WS-24: six statistics (mean, std, MAD, IQR, lag-1 autocorrelation, range) of the z-scored
series in the nine windows of TDA-90, each aggregated over the windows by mean, std, range and
largest absolute change between consecutive windows. data/paper/features_ws24/<part>/ also
links labels, tsfresh and tda90 of data/paper/features (not modified). Detectors:
src/detectors.py unchanged (same classifiers, folds, seed 42); only the WS-24 file name and the
three set names are registered. The fold of every series is checked against
results/paper/main/oof.parquet.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as st

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import detectors as D  # noqa: E402

FEAT0, RAW = REPO / "data/paper/features", REPO / "data/paper/raw"
FEAT = REPO / "data/paper/features_ws24"
OUT = REPO / "results/paper/ws24"
STATS = ["mean", "std", "mad", "iqr", "acf1", "range"]
AGGS = ["mean", "std", "range", "maxjump"]
NEW = ["WS-24", "TSFresh-774+WS-24", "TSFresh-774+WS-24+TDA-90"]


def ws24(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    sd = x.std()
    z = (x - x.mean()) / (sd if sd > 0 else 1.0)
    n = len(z); w = n // 5
    rows = []
    for a in range(0, n - w + 1, w // 2):
        v = z[a:a + w]; m = v.mean(); d = v - m
        den = (d * d).sum()
        acf1 = float((d[:-1] * d[1:]).sum() / den) if den > 0 else 0.0
        q75, q25 = np.percentile(v, [75, 25])
        rows.append([m, v.std(), np.median(np.abs(v - np.median(v))), q75 - q25, acf1, v.max() - v.min()])
    W = np.array(rows)
    assert W.shape == (9, 6), W.shape
    return np.concatenate([W.mean(0), W.std(0), W.max(0) - W.min(0), np.abs(np.diff(W, axis=0)).max(0)])


def names():
    return [f"ws__{s}__{a}" for a in AGGS for s in STATS]


def step_features():
    for p in sorted(q for q in FEAT0.iterdir() if (q / "labels.parquet").exists()):
        out = FEAT / p.name; out.mkdir(parents=True, exist_ok=True)
        for f in ["labels", "tsfresh", "tda90"]:
            link = out / f"{f}.parquet"
            if not link.exists():
                link.symlink_to(p / f"{f}.parquet")
        raw = pd.read_parquet(RAW / f"{p.name}.parquet", columns=["series_id", "time", "data"]).sort_values(["series_id", "time"])
        ids, X = [], []
        for sid, g in raw.groupby("series_id")["data"]:
            ids.append(sid); X.append(ws24(g.to_numpy(float)))
        L = pd.read_parquet(p / "labels.parquet")
        assert list(L.index) == ids, f"{p.name}: series order differs from labels"
        F = pd.DataFrame(np.array(X), index=pd.Index(ids, name="series_id"), columns=names())
        assert np.isfinite(F.to_numpy()).all()
        F.to_parquet(out / "ws24.parquet")
        print(f"{p.name}: {F.shape}", flush=True)


def step_detectors(models):
    D.FILES["WS-24"] = "ws24"
    D.SETS.append("WS-24")
    D.COMBINED.extend(NEW[1:])
    sys.argv = ["detectors.py", "--features", str(FEAT), "--raw", str(RAW), "--out", str(OUT), "--sets", *NEW, "--models", *models]
    D.main()
    o = pd.read_parquet(OUT / "oof.parquet")[["part", "series_id", "fold"]]
    m = pd.read_parquet(REPO / "results/paper/main/oof.parquet")[["part", "series_id", "fold"]]
    j = o.merge(m, on=["part", "series_id"], suffixes=("", "_main"))
    assert len(j) == len(o) == len(m) and (j["fold"] == j["fold_main"]).all(), "fold assignment differs from the main run"
    print("fold assignment identical to results/paper/main/oof.parquet", flush=True)


def mean8(folds, metric):
    d = folds[folds.level.between(1, 4)]
    n = d.groupby(["set", "model", "level", "fold"]).char.nunique()
    assert (n == 8).all()
    return d.groupby(["set", "model", "level", "fold"])[metric].mean()


def nb(delta):
    delta = np.asarray(delta, float); k = len(delta)
    se = np.sqrt((1 / k + 1 / 4) * delta.var(ddof=1))
    t = delta.mean() / se if se > 0 else np.inf * np.sign(delta.mean())
    return delta.mean(), se, 2 * st.t.sf(abs(t), k - 1)


def fmt(m, se, p):
    star = "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"{m:+.3f} ({se:.3f}) p={p:.3f}{star}"


def step_stats():
    out = OUT; lines = []
    F = pd.concat([pd.read_csv(out / "folds.csv"),
                   pd.read_csv(REPO / "results/paper/main/folds.csv").query("set in ['TDA-90','TSFresh-774']"),
                   pd.read_csv(REPO / "results/paper/complementarity/folds.csv").query("set == 'TSFresh-774+TDA-90'")])
    order = ["WS-24", "TDA-90", "TSFresh-774+WS-24", "TSFresh-774+TDA-90", "TSFresh-774+WS-24+TDA-90"]
    for metric in ["bal_acc", "auc"]:
        M = mean8(F, metric)
        def v(s, m, l): return M.xs((s, m, l), level=["set", "model", "level"]).sort_index().to_numpy()
        rows = []
        for m in ["lr", "hgb"]:
            for s in order:
                r = dict(metric=metric, model=m, set=s)
                for l in range(1, 5):
                    x = v(s, m, l); r[f"L{l}"] = f"{x.mean():.3f} ({np.sqrt((1/5+1/4)*x.var(ddof=1)):.3f})"
                rows.append(r)
        T14 = pd.DataFrame(rows); T14.to_csv(out / f"T14_mean_of_eight_{metric}.csv", index=False)
        lines += [f"\n=== {metric} ===", "T14 mean of the eight detectors (corrected SE)", T14.to_string(index=False)]
        for m in ["lr", "hgb"]:
            a, b = "TSFresh-774+WS-24+TDA-90", "TSFresh-774+WS-24"
            h12 = {l: nb(v(a, m, l) - v(b, m, l)) for l in range(1, 5)}
            g = nb((v(a, m, 4) - v(b, m, 4)) - (v(a, m, 1) - v(b, m, 1)))
            lines.append(f"T15 H12 {m}: " + "  ".join(f"L{l} {fmt(*h12[l])}" for l in h12) + f"  L4-L1 {fmt(*g)}")
            lines.append(f"T16 H13 {m}: L4 (TSFresh-774+TDA-90) - (TSFresh-774+WS-24) {fmt(*nb(v('TSFresh-774+TDA-90', m, 4) - v(b, m, 4)))}")
            d1 = v("TDA-90", m, 1) - v("WS-24", m, 1); d4 = v("TDA-90", m, 4) - v("WS-24", m, 4)
            lines.append(f"T17 H14 {m}: L1 {fmt(*nb(d1))}  L4 {fmt(*nb(d4))}  drop diff {fmt(*nb(d4 - d1))}")
        if metric == "bal_acc" or True:
            P = F[F.level == 4].set_index(["set", "model", "char", "fold"])[metric]
            for m in ["lr", "hgb"]:
                cells = []
                for c in D.LETTERS:
                    d = P.xs(("TSFresh-774+WS-24+TDA-90", m, c), level=["set", "model", "char"]).sort_index().to_numpy() - \
                        P.xs(("TSFresh-774+WS-24", m, c), level=["set", "model", "char"]).sort_index().to_numpy()
                    cells.append(f"{c} {fmt(*nb(d))}")
                lines.append(f"T18 H15 {m}: " + "  ".join(cells))
    txt = "\n".join(lines); (out / "stats.txt").write_text(txt); print(txt)


if __name__ == "__main__":
    t0 = time.time()
    s = sys.argv[1]
    if s == "features": step_features()
    elif s == "detectors": step_detectors(sys.argv[2:] or ["lr", "hgb"])
    elif s == "stats": step_stats()
    print(f"{s} done in {time.time() - t0:.0f}s", flush=True)
