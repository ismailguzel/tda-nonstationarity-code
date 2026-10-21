"""
Build the generation configs of the paper's dataset from betise's own list of combinations.

The design: every combination of eight non-stationarity characteristics that betise's
canonical full_dataset.json lists, with at most one structural break and at most one
anomaly per series, plus the stationary class — 72 classes.

    S stochastic trend   V volatility        T deterministic (linear) trend
    R trend shift        M mean shift        Q variance shift
    P point anomaly      C collective anomaly

betise 0.6.0 also lists combinations with two breaks (M+Q, M+R, Q+R) or two anomalies
(P+C) in one series — 126 in all; 0.5.0 did not allow them. They are left out, so the
dataset keeps the 72-class design (at most one break and one anomaly per series). Seasonality, contextual anomalies and long memory are out of
scope.

Scenario choices, uniform across classes so a letter means the same thing everywhere:
    base   S present -> random_walk or ari; otherwise ar or arma
    V      garch or egarch innovations on that base
    T      linear trend, direction drawn per series;  R  linear trend + direction change
    M, Q   a single break, direction drawn per series;  P, C  one anomaly
Series per class are split evenly over the base x volatility variants.

Six parts (levels 0-1 | level 2 in two halves | level 3 in two halves | levels 4-5), so no
single generation call holds all series in memory.

    python src/make_configs.py --per-class 500 --length 500 --out configs
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import betise
from betise.core import rules

ORDER = ["S", "V", "T", "R", "M", "Q", "P", "C"]
NAME = {"S": "stochastic_trend", "V": "volatility", "T": "trend", "R": "trend_shift",
        "M": "mean_shift", "Q": "variance_shift", "P": "point_anomaly", "C": "collective_anomaly"}
FAMILY_CHAR = {"stationary": None, "stochastic": "S", "volatility": "V", "seasonality": "Se", "fractional": "F"}
FEATURE_CHAR = {"trend_shift": "R", "mean_shift": "M", "variance_shift": "Q", "point_anomaly": "P",
                "collective_anomaly": "C", "contextual_anomaly": "X"}
OUT_OF_SCOPE = {"Se", "X", "F"}
BREAKS, ANOMALIES = {"M", "Q", "R"}, {"P", "C"}
EXPECTED_PER_LEVEL = {0: 1, 1: 8, 2: 22, 3: 26, 4: 13, 5: 2}

FEATURES = {
    "T": {"linear_trend": {"enabled": True, "direction": "both"}},
    "R": {"linear_trend": {"enabled": True, "direction": "both"},
          "trend_shift": {"enabled": True, "mode": "single", "change_type": "direction_change"}},
    "M": {"mean_shift": {"enabled": True, "mode": "single", "direction": "both"}},
    "Q": {"variance_shift": {"enabled": True, "mode": "single", "direction": "both"}},
    "P": {"point_anomaly": {"enabled": True}},
    "C": {"collective_anomaly": {"enabled": True}},
}


def characteristics(comp: dict) -> frozenset:
    s = {FAMILY_CHAR[rules.base_family(b)] for b in comp["base_components"]} - {None}
    for f in comp["features"]:
        s.add("T" if f in rules.TREND_FEATURES and f != "trend_shift" else FEATURE_CHAR[f])
    if "R" in s:
        s.discard("T")            # a trend shift is applied to the linear trend it requires
    return frozenset(s)


def scenarios(chars: frozenset, per_class: int) -> list[dict]:
    bases = ("random_walk", "ari") if "S" in chars else ("ar", "arma")
    vols = ("garch", "egarch") if "V" in chars else (None,)
    variants = list(itertools.product(bases, vols))
    out = []
    for base, vol in variants:
        feats = {}
        for c in ORDER:
            feats.update(FEATURES.get(c, {}) if c in chars else {})
        if vol:
            feats[vol] = {"enabled": True}
        out.append({"base_series": base, "n": per_class // len(variants), "features": feats})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--length", type=int, default=500)
    ap.add_argument("--prefix", default="all72")
    ap.add_argument("--out", default="configs")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    canon = json.loads((Path(betise.__file__).parent / "config" / "full_dataset.json").read_text())
    sets = {characteristics(c) for c in canon["compositions"]}
    sets = sorted((s for s in sets if not s & OUT_OF_SCOPE and len(s & BREAKS) <= 1 and len(s & ANOMALIES) <= 1),
                  key=lambda s: (len(s), [ORDER.index(c) for c in sorted(s, key=ORDER.index)]))
    per_level = {}
    for s in sets:
        per_level[len(s)] = per_level.get(len(s), 0) + 1
    assert per_level == EXPECTED_PER_LEVEL, f"betise {betise.__version__} gives {per_level}, expected {EXPECTED_PER_LEVEL}"

    classes = {"+".join(NAME[c] for c in ORDER if c in s) or "stationary": scenarios(s, args.per_class) for s in sets}
    names = list(classes)
    lvl = {n: 0 if n == "stationary" else n.count("+") + 1 for n in names}
    l2 = [n for n in names if lvl[n] == 2]; l3 = [n for n in names if lvl[n] == 3]
    P = args.prefix
    parts = {f"{P}a": [n for n in names if lvl[n] <= 1],
             f"{P}b": l2[: len(l2) // 2], f"{P}c": l2[len(l2) // 2:],
             f"{P}d": l3[: len(l3) // 2], f"{P}e": l3[len(l3) // 2:],
             f"{P}f": [n for n in names if lvl[n] >= 4]}
    assert sorted(sum(parts.values(), [])) == sorted(names)
    for part, members in parts.items():
        cfg = {"_comment": (f"Part {part}: generated by src/make_configs.py from betise {betise.__version__}'s "
                            f"full_dataset.json; do not edit by hand."),
               "betise_version": betise.__version__, "fixed_length": args.length, "random_seed": 42,
               "classes": {n: classes[n] for n in members}}
        (out / f"{part}-config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"  {part}: {len(members):2d} classes, levels {sorted({lvl[n] for n in members})}, "
              f"{sum(s['n'] for n in members for s in classes[n])} series")
    print(f"betise {betise.__version__}: {len(classes)} classes, per level {per_level}, "
          f"{sum(len(v) for v in classes.values())} scenarios, {sum(s['n'] for v in classes.values() for s in v)} series")


if __name__ == "__main__":
    main()
