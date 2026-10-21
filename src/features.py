"""
Extract the three feature families of one part, on the same series.

    tsfresh   EfficientFCParameters on each series z-scored with pandas std (ddof 1), as in
              the study; tsfresh's impute(); then the unit-root and stationarity tests
              dropped (augmented Dickey-Fuller and the other patterns below) — they test
              stationarity directly. 774 features at length 500.
    catch22   aeon's catch22 on each series z-scored with numpy std (ddof 0).
    tda90     src/topology.py, on the raw series (it z-scores internally, ddof 0);
              stored on a signed-log scale.
Writes <out>/<part>/{tsfresh,catch22,tda90,labels}.parquet, rows in series_id order.

    python src/features.py data/run/raw/all72a.parquet --out data/run/features
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

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from topology import tda90, tda90_names  # noqa: E402

# remove_leakage_features.py of the study: a TSFresh column is dropped if its lower-cased
# name contains any of these.
LEAKAGE_PATTERNS = ["augmented_dickey_fuller", "dickey_fuller", "adf_test", "adf", "kpss", "kpss_test", "kwiatkowski",
                    "phillips_perron", "phillips_peron", "pp_test", "perron", "unit_root", "unitroot", "unit_root_test",
                    "stationarity_test", "stationary_test", "is_stationary", "test_stationarity", "variance_ratio_test",
                    "lo_mackinlay", "zivot_andrews", "breakpoint_test", "cointegration_test"]


def tsfresh_features(ts: pd.DataFrame) -> pd.DataFrame:
    from tsfresh import extract_features
    from tsfresh.feature_extraction import EfficientFCParameters
    from tsfresh.utilities.dataframe_functions import impute
    g = ts.groupby("series_id")["data"]
    ts = ts.assign(data=(ts["data"] - g.transform("mean")) / g.transform("std").replace(0, 1.0))
    F = extract_features(ts, column_id="series_id", column_sort="time", column_value="data",
                         default_fc_parameters=EfficientFCParameters(), disable_progressbar=True, pivot=True)
    impute(F)
    F = F.sort_index()
    F.index.name = "series_id"
    leak = [c for c in F.columns if any(p in c.lower() for p in LEAKAGE_PATTERNS)]
    return F.drop(columns=leak)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    raw_path = Path(args.raw)
    part = raw_path.stem
    out = Path(args.out) / part; out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(raw_path, columns=["series_id", "time", "data", "primary_category"])
    raw = raw.sort_values(["series_id", "time"])
    labels = raw.groupby("series_id")["primary_category"].first()
    labels.to_frame().to_parquet(out / "labels.parquet")
    ids = labels.index.to_numpy()
    X = np.stack([g.to_numpy(float) for _, g in raw.groupby("series_id")["data"]])
    print(f"{part}: {len(ids)} series of length {X.shape[1]}", flush=True)

    t = time.time()
    T = np.array(Parallel(n_jobs=-1, batch_size=64)(delayed(tda90)(x) for x in X))
    pd.DataFrame(T, index=pd.Index(ids, name="series_id"), columns=tda90_names()).to_parquet(out / "tda90.parquet")
    print(f"  tda90    {T.shape[1]:4d} features ({time.time() - t:.0f}s)", flush=True)

    t = time.time()
    from aeon.transformations.collection.feature_based import Catch22
    from aeon.transformations.collection.feature_based._catch22 import feature_names
    sd = X.std(1, keepdims=True)
    Z = (X - X.mean(1, keepdims=True)) / np.where(sd > 0, sd, 1.0)
    C = np.nan_to_num(Catch22(n_jobs=-1).fit_transform(Z[:, None, :]))
    pd.DataFrame(C, index=pd.Index(ids, name="series_id"), columns=list(feature_names)).to_parquet(out / "catch22.parquet")
    print(f"  catch22  {C.shape[1]:4d} features ({time.time() - t:.0f}s)", flush=True)

    t = time.time()
    F = tsfresh_features(raw[["series_id", "time", "data"]])
    assert (F.index.to_numpy() == ids).all()
    F.to_parquet(out / "tsfresh.parquet")
    print(f"  tsfresh  {F.shape[1]:4d} features ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
