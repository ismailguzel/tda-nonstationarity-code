"""Sensitivity re-run: every TSFresh-containing comparison without the
three TSFresh columns that are not invariant to the raw level/scale or are non-finite
(leak_scan.py): data__variation_coefficient (std/mean of a z-scored series, NaN for ~23%,
imputed with part statistics), data__fft_coefficient__attr_"angle"__coeff_0 (phase of the
mean residue, 0 or 180), data__query_similarity_count__query_None__threshold_0.0 (constant).
Same code (src/detectors.py, src/detectors_budget.py), same folds, seed 42; only load() is
wrapped to drop the columns. Outputs: results/paper/sensitivity_tsfresh771/{main,complementarity,budget}/.

    python experiments/checks/tsfresh771_sensitivity_2026-10-22.py main | complementarity | budget
"""
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/paper/sensitivity_tsfresh771"
sys.path.insert(0, str(REPO / "src"))
import detectors as D                    # noqa: E402
DROP = {"data__variation_coefficient", 'data__fft_coefficient__attr_"angle"__coeff_0',
        "data__query_similarity_count__query_None__threshold_0.0"}
_orig = D.load
def load(features, sets):
    parts, key, cls, out, names = _orig(features, sets)
    for s in out:
        keep = [i for i, n in enumerate(names[s]) if n not in DROP]
        dropped = len(names[s]) - len(keep)
        out[s], names[s] = out[s][:, keep], [names[s][i] for i in keep]
        print(f"  {s}: dropped {dropped}, {len(keep)} columns", flush=True)
    return parts, key, cls, out, names
D.load = load
import detectors_budget as B             # noqa: E402
B.load = load
F, RAW = str(REPO / "data/paper/features"), str(REPO / "data/paper/raw")
t0 = time.time()
step = sys.argv[1]
if step == "main":
    sys.argv = ["detectors.py", "--features", F, "--raw", RAW, "--out", str(OUT / "main"), "--sets", "TSFresh-774"]; D.main()
elif step == "complementarity":
    sys.argv = ["detectors.py", "--features", F, "--raw", RAW, "--out", str(OUT / "complementarity"),
                "--sets", "TSFresh-774+TDA-90", "TSFresh-774+catch22"]; D.main()
elif step == "budget":
    sys.argv = ["detectors_budget.py", "--features", F, "--folds-from", str(REPO / "results/paper/main/oof.parquet"),
                "--out", str(OUT / "budget")]; B.main()
print(f"{step} done in {time.time() - t0:.0f}s", flush=True)
