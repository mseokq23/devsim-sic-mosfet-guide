#!/usr/bin/env bash
# Run every step in order and keep the output in logs/NN_name.log (about 45 min on 2 cores).
# Steps 22 and 23 skip runs that already have a JSON in results/; delete results/oat, results/doe to redo them.
set -euo pipefail
cd "$(dirname "$0")"
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p logs
for s in steps/[0-9][0-9]_*.py; do
  name=$(basename "$s" .py)
  echo "== $name"
  if [ "$name" = "24_inverse_gp" ]; then
    python "$s" --src "${SRC:-../sic-mosfet-devsim}" 2>&1 | tee "logs/$name.log"
  else
    python "$s" 2>&1 | tee "logs/$name.log"
  fi
done
