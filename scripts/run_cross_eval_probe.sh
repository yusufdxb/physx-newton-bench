#!/usr/bin/env bash
# Cheap falsification probe: fixed PhysX-trained checkpoints, two backends.
set -u
: "${ISAACLAB_PATH:?set ISAACLAB_PATH to the Isaac Lab checkout used for the checkpoints}"
: "${ISAACLAB_PYTHON:?set ISAACLAB_PYTHON to its Python interpreter}"

BENCH="$(cd "$(dirname "$0")/.." && pwd)"
TASK=Isaac-Velocity-Flat-Unitree-Go2-v0
OUT_NAME="${CROSS_EVAL_OUT:-cross_eval_probe}"
OUT="$BENCH/results/$OUT_NAME"
LOGS="$BENCH/logs/$OUT_NAME"
EXP="$ISAACLAB_PATH/logs/rsl_rl/go2flat_bench10"
read -r -a SEEDS <<< "${POLICY_SEEDS:-0 1 2}"
EVAL_SEED="${CROSS_EVAL_SEED:-20260726}"
TRAIN_BACKEND="${POLICY_TRAIN_BACKEND:-physx}"
if [ "$TRAIN_BACKEND" != "physx" ] && [ "$TRAIN_BACKEND" != "newton" ]; then
  echo "[CROSS-EVAL] POLICY_TRAIN_BACKEND must be physx or newton"
  exit 2
fi

export OMNI_KIT_ACCEPT_EULA=YES
export VIRTUAL_ENV="$(dirname "$(dirname "$ISAACLAB_PYTHON")")"
mkdir -p "$OUT/cells" "$LOGS"

for seed in "${SEEDS[@]}"; do
  policy_id="${TRAIN_BACKEND}_s${seed}"
  run_dir="$(find "$EXP" -maxdepth 1 -type d -name "*_${policy_id}" | sort | tail -1)"
  checkpoint="$run_dir/model_299.pt"
  if [ ! -f "$checkpoint" ]; then
    echo "[CROSS-EVAL] missing checkpoint for $policy_id: $checkpoint"
    exit 2
  fi
  for backend in physx newton; do
    backend_overrides=()
    if [ "$backend" = "newton" ]; then
      if [ -n "${CROSS_EVAL_NEWTON_NJMAX:-}" ]; then
        backend_overrides+=("sim.physics.solver_cfg.njmax=$CROSS_EVAL_NEWTON_NJMAX")
      fi
      if [ -n "${CROSS_EVAL_NEWTON_NCONMAX:-}" ]; then
        backend_overrides+=("sim.physics.solver_cfg.nconmax=$CROSS_EVAL_NEWTON_NCONMAX")
      fi
    fi
    tag="${policy_id}_${backend}"
    cell="$OUT/cells/${tag}.json"
    if [ "${CROSS_EVAL_RESUME:-0}" = "1" ] && \
       "$ISAACLAB_PYTHON" -c \
         'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(not (d.get("seed")==int(sys.argv[2]) and d.get("policy_id")==sys.argv[3] and d.get("backend")==sys.argv[4]))' \
         "$cell" "$EVAL_SEED" "$policy_id" "$backend" 2>/dev/null; then
      echo "[CROSS-EVAL] resume skip policy=$policy_id backend=$backend"
      continue
    fi
    echo "[CROSS-EVAL] policy=$policy_id backend=$backend $(date -Is)"
    "$ISAACLAB_PYTHON" "$BENCH/scripts/evaluate_checkpoint.py" \
      --task "$TASK" --headless --device cuda:0 \
      --checkpoint "$checkpoint" --policy-id "$policy_id" --backend-id "$backend" \
      --policy-joint-order "$TRAIN_BACKEND" \
      --num-envs 64 --num-episodes 64 --seed "$EVAL_SEED" \
      --output "$cell" \
      "presets=$backend" "${backend_overrides[@]}" "hydra.run.dir=$OUT/hydra/$tag" \
      > "$LOGS/${tag}.log" 2>&1
    exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
      echo "[CROSS-EVAL] failed policy=$policy_id backend=$backend exit=$exit_code"
      tail -30 "$LOGS/${tag}.log"
      exit "$exit_code"
    fi
  done
done

"$ISAACLAB_PYTHON" "$BENCH/scripts/analyze_cross_eval.py" \
  --results-dir "$OUT/cells" \
  --output-json "$OUT/summary.json" \
  --output-md "$OUT/summary.md"
