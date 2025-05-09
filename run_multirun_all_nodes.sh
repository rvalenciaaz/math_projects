#!/usr/bin/env bash
# run_multirun_all_nodes.sh
# ------------------------------------------------------------
# Batch‑run the multi‑run NSGA‑II script for node counts 12‑48.
#
# Usage (examples):
#   ./run_multirun_all_nodes.sh                    # default params
#   ./run_multirun_all_nodes.sh --runs 5 --cores 4 # override some params
#
# Any extra flags you pass are forwarded verbatim to
#   conn_cubic_evol_return_scalabel_multirun.py
# ------------------------------------------------------------
set -euo pipefail

SCRIPT="conn_cubic_evol_return_scalabel_multirun.py"

# Extra args let you override gens, runs, seed, cores, etc.
EXTRA_ARGS="$@"

# We iterate over *even* node counts; 3‑regular graphs require n even.
for N in $(seq 12 2 48); do
    OUTDIR="deap_return_conn_3reg_n${N}"
    echo "=== Running nodes=${N} → ${OUTDIR} ==="

    python "$SCRIPT" \
        --nodes "$N" \
        --outdir "$OUTDIR" \
        $EXTRA_ARGS

done

echo "[✓] All runs completed."
