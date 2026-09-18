# Strengthening-suite summary

## Scientific validity preflight

**INVALID PENDING RERUN for backend-only learning claims.** The committed resolved configs contain uncontrolled non-backend differences. Run `python3 scripts/compare_semantic_configs.py --fail-on-unsafe` for the machine-readable manifest and rerun after the preflight passes.

## Pure-step throughput + per-process VRAM

| num_envs | PhysX steps/s | Newton steps/s | ratio | PhysX VRAM MiB | Newton VRAM MiB |
|---|---|---|---|---|---|
| 256 | 36,877 ± 2,829 | 82,281 ± 10,609 | 2.23x | 2451 | 304 |
| 1024 | 136,708 ± 9,811 | 282,463 ± 33,819 | 2.07x | 2723 | 368 |
| 2048 | 247,140 ± 14,538 | 477,471 ± 49,831 | 1.93x | 3019 | 434 |
| 4096 | 371,229 ± 18,567 | 732,796 ± 62,073 | 1.97x | 3553 | 566 |

## Training (n=10 seeds, 2048 envs, 300 iters)

| metric | PhysX | Newton |
|---|---|---|
| final reward IQM [95% CI] | 30.36 [28.71, 31.09] | 19.48 [16.77, 22.36] |
| training fps IQM [95% CI] | 147,155 [145,943, 148,544] | 241,070 [240,527, 242,441] |
| total wall mean (s) | 115.7 | 79.3 |
| proc VRAM peak mean (MiB) | 3247 | 617 |
