"""Generate the series of the cross-generator check with GutenTAG.

Three steps; GutenTAG 1.5.0 needs numpy >= 2:

    python xgen_generate_2026-11-15.py params              # any env with numpy: draws all parameters
    python xgen_generate_2026-11-15.py gutentag            # env with timeeval-gutentag 1.5.0 (arms A, C)
    python xgen_generate_2026-11-15.py combine             # series.parquet + labels.csv

Output in data/external/xgen/ (git-ignored).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/external/xgen"
N, SEED, PER_A, PER_B, PER_C = 500, 42, 40, 40, 20
LETTERS = ["S", "V", "T", "R", "M", "Q", "P", "C"]
ARM_A = {"WN": ["none", "M", "Q", "P", "C", "Cplat"], "RW": ["none", "M", "Q", "P", "C", "Cplat"],
         "LT": ["none", "M", "Q", "P", "C", "Cplat", "R"]}
# Parameter sets of a further arm (B) are drawn between arms A and C and not used here; they are
# kept so that the random stream, and hence every series of arm C, is unchanged.
ARM_B = {"WN": ["Shock", "LevelShift", "TrendChange"], "RW": ["Shock", "LevelShift", "TrendChange"]}
ARM_C = {"sine": ["none", "M", "P", "C"], "ecg": ["none", "M", "P", "C"], "cylinder-bell-funnel": ["none", "M", "P", "C"]}


# ----------------------------------------------------------------------------- parameters
def params() -> None:
    rng = np.random.default_rng(SEED)
    P = []
    for arm, cells, per in (("A", ARM_A, PER_A), ("B", ARM_B, PER_B), ("C", ARM_C, PER_C)):
        for base, injs in cells.items():
            for inj in injs:
                for i in range(per):
                    p = dict(id=f"{arm}|{base}|{inj}|{i + 1}", arm=arm, base=base, injection=inj,
                             seed=int(rng.integers(2 ** 31 - 1)),
                             pos=int(round(rng.uniform(0.41, 0.59) * N)),
                             sign=int(rng.choice([-1, 1])),
                             uM=float(rng.uniform(1.5, 3)), sM=float(rng.uniform(1, 3)),
                             uQ=float(rng.uniform(1.5, 3)), sQ=float(rng.uniform(2, 5)), up=bool(rng.random() < 0.5),
                             uC=float(rng.uniform(1, 1.75)), sC=float(rng.uniform(1, 2)),
                             lenC=int(rng.integers(25, 45)),
                             slope=float(rng.uniform(0.01, 0.1)) * int(rng.choice([-1, 1])))
                    P.append(p)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "params.json").write_text(json.dumps(P, indent=0))
    print(f"{len(P)} parameter sets")


def load_params(arm: str | None = None) -> list[dict]:
    P = json.loads((OUT / "params.json").read_text())
    return [p for p in P if arm is None or p["arm"] in arm]


# ----------------------------------------------------------------------------- GutenTAG
def gt_base(p: dict) -> dict:
    b = p["base"]
    if b == "WN":
        return {"kind": "polynomial", "polynomial": [0.0], "variance": 1.0}
    if b == "LT":
        return {"kind": "polynomial", "polynomial": [0.0, p["slope"]], "variance": 1.0}
    if b == "RW":
        return {"kind": "random-walk", "amplitude": 1.0, "smoothing": 0.01, "variance": 0.05}
    return {"kind": b, "variance": 0.1}                              # arm C: GutenTAG defaults otherwise


def gutentag() -> None:
    os.environ["HOME"] = str(OUT / ".home")                          # GitPython must not read a global git config
    (OUT / ".home").mkdir(parents=True, exist_ok=True)
    from gutenTAG import GutenTAG
    import importlib.metadata as md

    def gen(p, anomalies):
        g = GutenTAG(seed=p["seed"])
        g.load_config_dict({"timeseries": [{"name": "x", "length": N, "base-oscillations": [gt_base(p)],
                                            "anomalies": anomalies}]})
        ts = g.generate(return_timeseries=True)[0].timeseries
        return ts.iloc[:, 0].to_numpy(float), ts["is_anomaly"].to_numpy(int)

    rows, X = [], []
    for p in load_params("AC"):
        base, _ = gen(p, [])
        k, s, inj = p["pos"], p["sign"], p["injection"]
        sd_before = float(base[:k].std())
        b = gt_base(p)
        noise_sd = b["variance"] * (b.get("amplitude", 1.0))
        a, c0, c1 = [], k - p["lenC"] // 2, k - p["lenC"] // 2 + p["lenC"]
        loc_sd = float(base[max(0, c0 - 50):min(N, c1 + 50)].std())
        offC = s * p["uC"] * p["sC"] * loc_sd
        if inj == "M":
            a = [{"exact-position": k, "length": N - k, "kinds": [{"kind": "mean", "offset": s * p["uM"] * p["sM"] * sd_before}]}]
        elif inj == "Q":
            f = p["uQ"] * p["sQ"]
            a = [{"exact-position": k, "length": N - k, "kinds": [{"kind": "variance", "variance": noise_sd * (f if p["up"] else 1 / f)}]}]
        elif inj == "P":
            a = [{"exact-position": k, "length": 1, "kinds": [{"kind": "extremum", "min": s < 0, "local": False}]}]
        elif inj == "C":
            a = [{"exact-position": c0, "length": p["lenC"], "kinds": [{"kind": "mean", "offset": offC}]}]
        elif inj == "Cplat":
            a = [{"exact-position": c0, "length": p["lenC"], "kinds": [{"kind": "platform", "value": float(base[c0:c1].mean() + offC)}]}]
        elif inj == "R":
            # GutenTAG 1.5.0 trend anomaly fails unless its length is a multiple of 5
            # (int(0.2 L) + int(0.8 L) must equal L): the break is moved back by at most 4 points
            k = N - ((N - k) // 5) * 5
            a = [{"exact-position": k, "length": N - k,
                  "kinds": [{"kind": "trend", "oscillation": {"kind": "polynomial", "polynomial": [0.0, -2 * p["slope"]]}}]}]
        x, lab = gen(p, [dict(d, channel=0) for d in a]) if a else (base, np.zeros(N, int))
        first = c0 if inj in ("C", "Cplat") else k
        assert np.allclose(x[:first - 1], base[:first - 1]), (p["id"], "base differs before the injection")
        if inj in ("M", "Q", "R", "P", "C", "Cplat"):
            assert lab.sum() > 0, (p["id"], "no anomaly label")
        y = {l: 0 for l in LETTERS}
        if p["arm"] == "A":
            y["S"] = int(p["base"] == "RW")
            y["T"] = int(p["base"] == "LT" and inj != "R")   # paper convention (Section 2.1): a trend shift is labelled R, not T
            if inj in ("M", "Q", "P", "R"):
                y[inj] = 1
            if inj in ("C", "Cplat"):
                y["C"] = 1
        else:
            y = {l: np.nan for l in LETTERS}
        rows.append(dict(id=p["id"], arm=p["arm"], base=p["base"], injection=inj, n_events=int(inj != "none"),
                         generator=f"GutenTAG {md.version('timeeval-gutentag')}", **y))
        X.append(x)
    pd.DataFrame(rows).to_csv(OUT / "labels_gutentag.csv", index=False)
    pd.DataFrame(np.array(X), index=[r["id"] for r in rows]).rename(columns=str).to_parquet(OUT / "series_gutentag.parquet")
    print(f"GutenTAG: {len(rows)} series")


def combine() -> None:
    L = pd.read_csv(OUT / "labels_gutentag.csv")
    Z = pd.read_parquet(OUT / "series_gutentag.parquet")
    assert list(Z.index) == list(L["id"]) and Z.shape[1] == N and np.isfinite(Z.to_numpy()).all()
    assert (Z.std(axis=1) > 0).all()
    L.to_csv(OUT / "labels.csv", index=False)
    Z.to_parquet(OUT / "series.parquet")
    print(L.groupby(["arm", "base", "injection"]).size().to_string())


if __name__ == "__main__":
    {"params": params, "gutentag": gutentag, "combine": combine}[sys.argv[1]]()
