#!/bin/bash
# =============================================================================
# Every result of the paper, end to end, in one command:
#
#   configs -> validation -> generation -> features
#   -> main detectors          (TDA-90, TSFresh-774, catch22, MiniRocket)     H1-H4
#   -> complementarity         (TSFresh-774 + TDA-90, TSFresh-774 + catch22)  H5-H8
#   -> equal feature budget    (TSFresh top-100 vs union top-100)            H9-H11
#   -> tables and tests of each; interpretation; diagnosis
#
#   bash experiments/run_all.sh <tag> [series per class, default 500]
#
#   bash experiments/run_all.sh smoke 10        # every step, minutes (with MODELS=lr)
#   bash experiments/run_all.sh paper           # the paper's numbers (overnight)
#
# Environment variables:
#   PAPER_PYTHON   interpreter (default: python on PATH); must import betise 0.6.0,
#                  tsfresh, aeon, ripser — checked before anything runs
#   MODELS         classifiers, default "lr hgb"
#   RUNS_DIR       where outputs go, default <repo>/runs (git-ignored)
#
# Data    -> data/<tag>/            (git-ignored; regenerated from the configs)
# Outputs -> $RUNS_DIR/<tag>/
#   manifest.txt, environment.txt, configs/, logs/<step>.log
#   main/  complementarity/  budget/   detectors' folds.csv, oof.parquet, ... and stats.txt, tables/
#   interpret/  diagnose/
# =============================================================================
set -euo pipefail

TAG=${1:?usage: run_all.sh <tag> [series per class]}
PER=${2:-500}
REPO=$(cd "$(dirname "$0")/.." && pwd)
# Its own variable, not PYTHON: a PYTHON exported for another project once picked the
# wrong interpreter silently.
PY=${PAPER_PYTHON:-python}
MODELS=${MODELS:-lr hgb}
BETISE=0.6.0
DATA=$REPO/data/$TAG
OUT=${RUNS_DIR:-$REPO/runs}/$TAG
mkdir -p "$DATA/raw" "$DATA/features" "$OUT/logs"
cd "$REPO"

$PY -c "
import sys, importlib
for m in ('betise', 'tsfresh', 'aeon', 'ripser', 'sklearn', 'scipy', 'pandas', 'pyarrow'):
    importlib.import_module(m)
import betise
assert betise.__version__ == '$BETISE', f'betise {betise.__version__}, expected $BETISE'
print(f'environment ok: {sys.executable}, betise {betise.__version__}')
" || { echo "environment check failed for $PY — stop"; exit 1; }

step() {
    local name=$1; shift
    echo "[$(date '+%F %T')] >>> $name"
    "$@" 2>&1 | { grep --line-buffered -v -i "warning" || true; } | tee "$OUT/logs/$name.log"
    echo "[$(date '+%F %T')] <<< $name"
}

{
    echo "tag            $TAG"
    echo "per class      $PER"
    echo "models         $MODELS"
    echo "commit         $(git rev-parse HEAD)"
    echo "dirty files    $(git status --porcelain | grep -v '^??' | wc -l | tr -d ' ')"
    echo "betise         $($PY -c 'import betise; print(betise.__version__)')"
    echo "python         $($PY -c 'import sys; print(sys.version.split()[0])') ($($PY -c 'import sys; print(sys.executable)'))"
    echo "host           $(hostname), $(nproc 2>/dev/null || sysctl -n hw.ncpu) cores"
    echo "start          $(date '+%F %T')"
} | tee "$OUT/manifest.txt"
$PY -m pip freeze > "$OUT/environment.txt"

step configs  $PY src/make_configs.py --per-class "$PER" --out "$OUT/configs"
if [ "$PER" = 500 ]; then
    diff -r "$OUT/configs" configs > /dev/null || { echo "configs differ from the committed configs/ — stop"; exit 1; }
fi
step validate $PY src/validate.py "$OUT"/configs/all72?-config.json
for c in "$OUT"/configs/all72?-config.json; do
    p=$(basename "$c" -config.json)
    step "generate_$p" $PY src/generate.py "$c" --out "$DATA/raw"
done
for c in "$OUT"/configs/all72?-config.json; do
    p=$(basename "$c" -config.json)
    step "features_$p" $PY src/features.py "$DATA/raw/$p.parquet" --out "$DATA/features"
done

F="--features $DATA/features"
step detectors_main            $PY -u src/detectors.py $F --raw "$DATA/raw" --out "$OUT/main" --models $MODELS
step detectors_complementarity $PY -u src/detectors.py $F --raw "$DATA/raw" --out "$OUT/complementarity" --models $MODELS \
                                   --sets TSFresh-774+TDA-90 TSFresh-774+catch22
step detectors_budget          $PY -u src/detectors_budget.py $F --folds-from "$OUT/main/oof.parquet" --out "$OUT/budget" --models $MODELS
step stats_main                $PY src/stats_main.py "$OUT/main"
step stats_complementarity     $PY src/stats_complementarity.py "$OUT/main" "$OUT/complementarity"
step stats_budget              $PY src/stats_budget.py "$OUT/main" "$OUT/complementarity" "$OUT/budget"
step interpret                 $PY src/interpret.py --run "$OUT/main" $F --out "$OUT/interpret"
step diagnose                  $PY src/diagnose.py --run "$OUT/main" --out "$OUT/diagnose"

echo "end            $(date '+%F %T')" | tee -a "$OUT/manifest.txt"
