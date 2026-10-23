#!/usr/bin/env bash
# Full pipeline: 13 dynamic simulations (~35 min, single core), then statistics,
# energy budget, figures and LaTeX macros.
set -euo pipefail
cd "$(dirname "$0")"

echo "== dynamic simulations (control 200 Hz, RK4 1 ms) =="
for m in 0 0.5 1 2; do
  for law in ct pdg pd; do
    python3 simulate_control.py "$law" 5 5 "$m"
  done
done
echo "== integration convergence check (RK4 0.5 ms) =="
python3 simulate_control.py ct 5 10

echo "== inverse kinematics statistics =="
python3 ik_statistics.py
echo "== energy budget =="
python3 energy_budget.py
echo "== figures =="
python3 figure_1_kinematics.py
python3 figure_2_workspace.py
python3 figures_3_to_7.py
echo "== LaTeX macros =="
python3 make_macros.py
echo "done: results/, figures/, latex/"
