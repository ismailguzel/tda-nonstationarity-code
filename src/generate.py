"""
Generate one part of the dataset from its config.

Each scenario is generated with seed = random_seed + its index within its class (as in the
study), series ids are shuffled once with random_seed, and the class name is written to
primary_category. betise's own metadata columns are kept, so the realised breaks and
anomalies can be inspected later.

    python src/generate.py configs/all72a-config.json --out data/run/raw
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from betise import generate_dataframe  # noqa: E402
from betise.config import load_config  # noqa: E402

from feature_defaults import ALL_FEATURES_OFF  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", required=True, help="directory; writes <part>.parquet")
    args = ap.parse_args()
    path = Path(args.config)
    part = path.name.removesuffix("-config.json")
    cfg = json.loads(path.read_text())
    seed0, length = cfg["random_seed"], cfg["fixed_length"]

    frames, offset = [], 0
    for cls, scens in cfg["classes"].items():
        for i, s in enumerate(scens):
            df, ctx = generate_dataframe(load_config(dataset={
                "base_series": s["base_series"], "num_series": s["n"], "length_range": [length, length],
                "random_seed": seed0 + i, "features": {**ALL_FEATURES_OFF, **s["features"]}}))
            df["primary_category"] = cls
            df["sub_category"] = ctx["label"]
            df["series_id"] = df["series_id"] + offset
            offset = int(df["series_id"].max())
            frames.append(df)
        print(f"  {cls}: {sum(s['n'] for s in scens)} series", flush=True)

    D = pd.concat(frames, ignore_index=True)
    ids = D["series_id"].unique()
    remap = {old: new + 1 for new, old in enumerate(np.random.default_rng(seed0).permutation(ids))}
    D["series_id"] = D["series_id"].map(remap)
    D = D.sort_values(["series_id", "time"]).reset_index(drop=True)
    for c in D.select_dtypes(include=["object", "string"]).columns:
        D[c] = D[c].astype(str)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    D.to_parquet(out / f"{part}.parquet", index=False)
    print(f"{part}: {D['series_id'].nunique()} series, {D['primary_category'].nunique()} classes -> {out / f'{part}.parquet'}")


if __name__ == "__main__":
    main()
