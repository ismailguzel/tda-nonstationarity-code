"""
Check generation configs against the installed betise before a long run. Three checks:

1. ALL_FEATURES_OFF names exactly betise's FEATURE_ORDER, so no overlay is left at
   betise's default.
2. Every scenario generates (betise's combination rules accept it).
3. The output carries exactly what was asked for: on a few series per scenario, the
   overlays in feature_components are exactly the enabled trends, breaks and anomalies,
   and base_components is the requested base plus any volatility model (betise records
   volatility as a base component).

    python src/validate.py configs/all72?-config.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import betise  # noqa: E402
from betise import generate_dataframe  # noqa: E402
from betise.config import load_config  # noqa: E402

from feature_defaults import ALL_FEATURES_OFF  # noqa: E402

N_PER_SCENARIO = 3
VOLATILITY = {"arch", "garch", "egarch", "aparch"}


def main() -> int:
    from betise.dataset_generation import FEATURE_ORDER
    ok = sorted(FEATURE_ORDER) == sorted(ALL_FEATURES_OFF)
    print(f"betise {betise.__version__}: ALL_FEATURES_OFF {'matches' if ok else 'DOES NOT MATCH'} FEATURE_ORDER")
    for path in map(Path, sys.argv[1:]):
        cfg_json = json.loads(path.read_text())
        length, bad, n = cfg_json["fixed_length"], 0, 0
        for cls, scens in cfg_json["classes"].items():
            for i, s in enumerate(scens):
                enabled = {k for k, v in s["features"].items() if v.get("enabled")}
                want, want_base = enabled - VOLATILITY, [s["base_series"]] + sorted(enabled & VOLATILITY)
                try:
                    df, _ = generate_dataframe(load_config(dataset={
                        "base_series": s["base_series"], "num_series": N_PER_SCENARIO, "length_range": [length, length],
                        "random_seed": 3, "features": {**ALL_FEATURES_OFF, **s["features"]}}))
                except Exception as e:  # noqa: BLE001 — any failure here is a config problem
                    bad += 1
                    print(f"  FAIL {path.name} {cls} #{i}: {type(e).__name__}: {' '.join(str(e).split())[:160]}")
                    continue
                for _, r in df.groupby("series_id").first().iterrows():
                    n += 1
                    got = set(json.loads(r.feature_components)) if isinstance(r.feature_components, str) else set()
                    base = json.loads(r.base_components) if isinstance(r.base_components, str) else []
                    if got != want or base != want_base:
                        bad += 1
                        print(f"  BAD  {path.name} {cls}: base {base} wanted {want_base}, overlays {sorted(got)} wanted {sorted(want)}")
        print(f"  {'ok  ' if not bad else 'FAIL'} {path.name}: {len(cfg_json['classes'])} classes, {n} series checked")
        ok &= not bad
    print("all valid" if ok else "INVALID — fix before generating")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
