#!/usr/bin/env bash
# Pillar 1: open-loop dynamics-equivalence probe.
# Records the same action tape under both backends (twice each, to verify
# same-backend determinism), then compares trajectories.
set -u
: "${ISAACLAB_PATH:?set ISAACLAB_PATH to your Isaac Lab checkout}"
: "${ISAACLAB_PYTHON:?set ISAACLAB_PYTHON to the Isaac Lab venv python}"
BENCH="$(cd "$(dirname "$0")/.." && pwd)"
export OMNI_KIT_ACCEPT_EULA=YES
export VIRTUAL_ENV="$(dirname "$(dirname "$ISAACLAB_PYTHON")")"

TASK=Isaac-Velocity-Flat-Unitree-Go2-v0
OUT=$BENCH/results/pillar1
mkdir -p "$OUT" "$BENCH/logs"

for preset in physx newton; do
  for rep in "" 2; do
    TAG=${preset}${rep}
    echo "[PILLAR1] record backend=$preset rep=${rep:-1} $(date -Is)"
    "$ISAACLAB_PYTHON" "$BENCH/scripts/record_dynamics.py" \
      --task "$TASK" --headless \
      --tag "$TAG" --outdir "$OUT" --tape "$OUT/tape.npz" --horizon 300 \
      presets=$preset > "$BENCH/logs/pillar1_${TAG}.log" 2>&1 \
      || { echo "[PILLAR1] FAILED $TAG — tail:"; tail -15 "$BENCH/logs/pillar1_${TAG}.log"; exit 1; }
  done
done

"$ISAACLAB_PYTHON" "$BENCH/scripts/compare_dynamics.py"
echo "[PILLAR1] done — see $OUT/dynamics_summary.json and dynamics_divergence.png"
