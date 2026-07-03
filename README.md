# physx-newton-bench

Controlled end-to-end comparison of the **PhysX** and **Newton (MuJoCo-Warp)** physics
backends in **Isaac Lab 3.0**, on the stock `Isaac-Velocity-Flat-Unitree-Go2-v0`
RL locomotion task, on a single consumer Blackwell-architecture NVIDIA GPU.

Three things are measured, in increasing order of novelty:

1. **Compute**: pure-step throughput scaling over `num_envs` (10 timed repeats per
   point), end-to-end RL training throughput, and per-process VRAM (n=10 training
   seeds per backend).
2. **Learning**: reward curves under identical stock config (deliberately untuned
   for Newton), reported as IQM over 10 seeds with 95% bootstrap CIs.
3. **Dynamics equivalence**: an open-loop probe showing both backends replay
   deterministically (repeat divergence exactly 0.0), quantifying their
   cross-backend divergence on the same asset,
   and documenting two migration footguns (joint-ordering mismatch, joint-drive
   gain representation).

## Headline results

| | PhysX | Newton | ratio |
|---|---|---|---|
| Pure-step throughput @2048 envs (env-steps/s) | 247,140 ± 14,538 | 477,471 ± 49,831 | **1.93x** |
| Pure-step throughput @4096 envs (env-steps/s) | 371,229 ± 18,567 | 732,796 ± 62,073 | **1.97x** |
| End-to-end training fps, IQM over 10 seeds | 147,155 | 241,070 | **1.64x** |
| Training wall-clock, mean of 10 runs (s) | 115.7 | 79.3 | **1.46x** |
| Per-process VRAM, training peak, stock configs (MiB) | 3247 | 617 | **5.3x** |
| Final reward @300 iters, IQM [95% CI] (mean±std: 29.8±2.7 / 19.6±3.5) | 30.4 [28.7, 31.1] | 19.5 [16.8, 22.4] | - |

![scaling](results/strengthen/fig_scaling.png)

**Honest framing that matters:**

- The pure-stepping speedup (~2x) is larger than the end-to-end training speedup
  (1.64x), because learning overhead is backend-independent. Quote the number that
  matches your workload.
- **At equal wall-clock, Newton is ahead; at equal iterations, PhysX converges
  higher** on this PhysX-tuned asset (see both learning-curve figures in
  [report.md](report.md)). The reward gap is an expected out-of-box-swap effect -
  the dynamics-equivalence probe shows the two backends genuinely integrate
  different dynamics, so the policies optimize different landscapes. It is not
  evidence that either backend is "worse."
- Newton's near-flat memory curve (304→566 MiB while quadrupling envs, vs
  2451→3553 MiB for PhysX) directly buys `num_envs` headroom on
  memory-constrained GPUs, but it is a **stock-config footprint, not an
  intrinsic engine property**: the stock PhysX arm pre-allocates a large patch
  pool (`gpu_max_rigid_patch_count = 10·2¹⁵`) while stock Newton sizes tight
  per-env buffers (`njmax=65, nconmax=35`). See the VRAM caveat in
  [report.md](report.md).

## Reproduce

```bash
# prerequisites: Isaac Lab 3.0 checkout + its venv; set these two paths
export ISAACLAB_PATH=/path/to/IsaacLab
export ISAACLAB_PYTHON=/path/to/venv/bin/python

./scripts/run_strengthen.sh   # env capture + probe sweep + 10-seed training (~35 min)
./scripts/run_pillar1.sh      # open-loop dynamics-equivalence probe (~5 min)
python scripts/analyze_strengthen.py   # tables + figures from raw artifacts
```

Every number traces to an on-disk artifact under `results/` (tensorboard event
extracts, JSON, CSV): never stdout, which Kit captures. Version capture
(`pip freeze`, Isaac Lab SHA, driver/CUDA) is in `results/strengthen/env_capture.txt`.

## Repo layout

- [`report.md`](report.md): full writeup: methodology, tables, figures, the
  dynamics-equivalence probe, and the gain-representation trace
- `scripts/`: probes, training orchestration, dynamics recorder/comparator, analysis
- `results/`: raw artifacts (probe JSONs, manifests, per-iteration CSVs, pillar-1 npz)

## Caveats (read before quoting numbers)

Single task, single robot, single GPU. Stock solver settings on both sides
(`MJWarpSolverCfg(njmax=65, nconmax=35, cone="pyramidal", integrator="implicitfast")`,
1 substep: deliberately not retuned). Zero-action stepping in the throughput probe.
The Newton backend in Isaac Lab 3.0 is experimental. No sim-to-real claims: which
backend's policies transfer better to a real robot is the question this repo does
NOT answer.

## License

Apache-2.0
