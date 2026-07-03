#!/usr/bin/env bash
# Strengthening suite: env capture -> throughput/VRAM sweep -> 10-seed training.
# Sequential on the GPU; nothing else should be running on it.
set -u
: "${ISAACLAB_PATH:?set ISAACLAB_PATH to your Isaac Lab checkout}"
: "${ISAACLAB_PYTHON:?set ISAACLAB_PYTHON to the Isaac Lab venv python}"
IL=$ISAACLAB_PATH
BENCH="$(cd "$(dirname "$0")/.." && pwd)"
PY=$ISAACLAB_PYTHON
export OMNI_KIT_ACCEPT_EULA=YES
export VIRTUAL_ENV="$(dirname "$(dirname "$ISAACLAB_PYTHON")")"
export PYTHONUNBUFFERED=1

TASK=Isaac-Velocity-Flat-Unitree-Go2-v0
OUT=$BENCH/results/strengthen
mkdir -p "$OUT" "$BENCH/logs/strengthen"

# ---------- 0. Environment capture ----------
{
  echo "== date ==";            date -Is
  echo "== isaaclab sha ==";    git -C "$IL" rev-parse HEAD
  echo "== isaaclab status =="; git -C "$IL" status --porcelain | head -20
  echo "== nvidia-smi ==";      nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
  echo "== cuda ==";            nvidia-smi | grep -o "CUDA Version: [0-9.]*"
  echo "== python ==";          "$PY" --version
  echo "== key packages ==";    "$PY" -m pip freeze | grep -Ei "^(newton|warp|torch|rsl|isaaclab|mujoco|gymnasium|numpy)"
  echo "== full pip freeze below =="
  "$PY" -m pip freeze
} > "$OUT/env_capture.txt" 2>&1
echo "[STRENGTHEN] env captured"

# ---------- 1. Throughput + per-process VRAM sweep ----------
for preset in physx newton; do
  for ne in 256 1024 2048 4096; do
    TAGN=probe_${preset}_${ne}
    echo "[STRENGTHEN] probe backend=$preset num_envs=$ne $(date -Is)"
    "$PY" "$BENCH/scripts/probe_sweep.py" \
      --task "$TASK" --headless --num_envs "$ne" \
      --n_steps 100 --repeats 10 --warmup 10 \
      --tag "$TAGN" --out "$OUT/${TAGN}.json" \
      presets=$preset > "$BENCH/logs/strengthen/${TAGN}.log" 2>&1
    EC=$?
    if [ $EC -ne 0 ] || [ ! -f "$OUT/${TAGN}.json" ]; then
      echo "[STRENGTHEN] PROBE FAILED backend=$preset num_envs=$ne exit=$EC — tail:"
      tail -15 "$BENCH/logs/strengthen/${TAGN}.log"
    fi
  done
done

# ---------- 2. Learning: 10 seeds x 2 backends x 300 iters ----------
NUM_ENVS=2048
MAX_ITERS=300
SEEDS=(0 1 2 3 4 5 6 7 8 9)
EXP=go2flat_bench10
MANIFEST=$OUT/manifest10.csv
echo "backend,seed,run_name,exit_code,total_wall_s,gpu_peak_mib,proc_peak_mib,log_dir" > "$MANIFEST"

cd "$IL" || exit 1
for preset in physx newton; do
  for seed in "${SEEDS[@]}"; do
    RUN=${preset}_s${seed}
    LOG=$BENCH/logs/strengthen/train_${RUN}.log
    GPUSAMP=$BENCH/logs/strengthen/gpu_${RUN}.log
    PROCSAMP=$BENCH/logs/strengthen/proc_${RUN}.log
    echo "[STRENGTHEN] train backend=$preset seed=$seed $(date -Is)"
    ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > "$GPUSAMP" 2>/dev/null &
    SAMP_PID=$!

    START=$(date +%s.%N)
    "$PY" scripts/reinforcement_learning/rsl_rl/train.py \
      --task "$TASK" --headless \
      --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERS" --seed "$seed" \
      --experiment_name "$EXP" --run_name "$RUN" --logger tensorboard \
      presets=$preset > "$LOG" 2>&1 &
    TRAIN_PID=$!
    # per-process VRAM sampler keyed to the train PID
    ( while kill -0 "$TRAIN_PID" 2>/dev/null; do
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
          | awk -F', *' -v p="$TRAIN_PID" '$1==p {print $2}'
        sleep 0.5
      done ) > "$PROCSAMP" 2>/dev/null &
    PSAMP_PID=$!
    wait "$TRAIN_PID"; EXITC=$?
    END=$(date +%s.%N)
    kill "$SAMP_PID" "$PSAMP_PID" 2>/dev/null; wait "$SAMP_PID" "$PSAMP_PID" 2>/dev/null

    WALL=$(echo "$END - $START" | bc)
    GPUPEAK=$(sort -n "$GPUSAMP" 2>/dev/null | tail -1)
    PROCPEAK=$(sort -n "$PROCSAMP" 2>/dev/null | tail -1)
    LOGDIR=$(ls -dt "$IL"/logs/rsl_rl/$EXP/*_${RUN} 2>/dev/null | head -1)
    echo "[STRENGTHEN] done backend=$preset seed=$seed exit=$EXITC wall=${WALL}s gpu_peak=${GPUPEAK}MiB proc_peak=${PROCPEAK}MiB"
    echo "${preset},${seed},${RUN},${EXITC},${WALL},${GPUPEAK},${PROCPEAK},${LOGDIR}" >> "$MANIFEST"
    if [ "$EXITC" -ne 0 ]; then
      echo "[STRENGTHEN] NONZERO EXIT — tail of log:"; tail -25 "$LOG"
    fi
  done
done
echo "[STRENGTHEN] ALL COMPLETE $(date -Is)"
cat "$MANIFEST"
