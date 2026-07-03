#!/usr/bin/env bash
# Controlled PhysX-vs-Newton benchmark: 2 backends x 3 seeds, identical config.
# Only difference between arms: the `presets=` Hydra token.
set -u
: "${ISAACLAB_PATH:?set ISAACLAB_PATH to your Isaac Lab checkout}"
: "${ISAACLAB_PYTHON:?set ISAACLAB_PYTHON to the Isaac Lab venv python}"
IL=$ISAACLAB_PATH
BENCH="$(cd "$(dirname "$0")/.." && pwd)"
PY=$ISAACLAB_PYTHON
export OMNI_KIT_ACCEPT_EULA=YES
export VIRTUAL_ENV="$(dirname "$(dirname "$ISAACLAB_PYTHON")")"

TASK=Isaac-Velocity-Flat-Unitree-Go2-v0
NUM_ENVS=2048
MAX_ITERS=300
SEEDS=(0 1 2)
EXP=go2flat_bench
MANIFEST=$BENCH/results/manifest.csv
echo "backend,seed,run_name,exit_code,total_wall_s,gpu_peak_mib,log_dir" > "$MANIFEST"

cd "$IL" || exit 1
for preset in physx newton; do
  for seed in "${SEEDS[@]}"; do
    RUN=${preset}_s${seed}
    LOG=$BENCH/logs/train_${RUN}.log
    GPUSAMP=$BENCH/logs/gpu_${RUN}.log
    echo "=========================================================="
    echo "[RUN] backend=$preset seed=$seed  $(date -Is)"
    # GPU memory sampler in background
    ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > "$GPUSAMP" 2>/dev/null &
    SAMP_PID=$!

    START=$(date +%s.%N)
    "$PY" scripts/reinforcement_learning/rsl_rl/train.py \
      --task "$TASK" --headless \
      --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERS" --seed "$seed" \
      --experiment_name "$EXP" --run_name "$RUN" --logger tensorboard \
      presets=$preset > "$LOG" 2>&1
    EXITC=$?
    END=$(date +%s.%N)
    kill "$SAMP_PID" 2>/dev/null; wait "$SAMP_PID" 2>/dev/null

    WALL=$(echo "$END - $START" | bc)
    GPUPEAK=$(sort -n "$GPUSAMP" 2>/dev/null | tail -1)
    LOGDIR=$(ls -dt "$IL"/logs/rsl_rl/$EXP/*_${RUN} 2>/dev/null | head -1)
    echo "[RUN] done backend=$preset seed=$seed exit=$EXITC wall=${WALL}s gpu_peak=${GPUPEAK}MiB logdir=$LOGDIR"
    echo "${preset},${seed},${RUN},${EXITC},${WALL},${GPUPEAK},${LOGDIR}" >> "$MANIFEST"
    if [ "$EXITC" -ne 0 ]; then
      echo "[RUN] NONZERO EXIT — tail of log:"; tail -25 "$LOG"
    fi
  done
done
echo "[BENCH] ALL RUNS COMPLETE $(date -Is)"
cat "$MANIFEST"
