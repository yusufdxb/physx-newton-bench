# Controlled cross-backend checkpoint evaluation

- Policies: 10
- Pairwise preference reversals: 18 / 45
- Spearman rank correlation: 0.2364
- Maximum rank shift: 8
- Probe survives: **True**
- Uncertainty: Per-policy return intervals bootstrap vectorized episodes within one simulator process. They do not replace process-level replication.

| policy | PhysX return | Newton return | Newton - PhysX [95% CI] | PhysX success | Newton success | ranks P/N |
|---|---:|---:|---:|---:|---:|---:|
| physx_s0 | 30.952 | -3.741 | -34.693 [-36.610, -32.713] | 0.984 | 0.000 | 3 / 5 |
| physx_s1 | 21.309 | -4.703 | -26.012 [-28.639, -23.417] | 0.750 | 0.000 | 10 / 8 |
| physx_s2 | 32.683 | -4.721 | -37.403 [-39.029, -35.501] | 0.969 | 0.000 | 1 / 9 |
| physx_s3 | 26.854 | -3.936 | -30.790 [-33.555, -27.858] | 0.938 | 0.000 | 9 / 7 |
| physx_s4 | 30.658 | -3.800 | -34.458 [-36.346, -32.469] | 0.953 | 0.000 | 4 / 6 |
| physx_s5 | 31.246 | -3.413 | -34.659 [-36.494, -32.630] | 0.969 | 0.000 | 2 / 1 |
| physx_s6 | 29.816 | -3.420 | -33.236 [-35.274, -31.088] | 0.953 | 0.000 | 6 / 2 |
| physx_s7 | 30.388 | -3.518 | -33.906 [-36.178, -31.542] | 0.875 | 0.000 | 5 / 3 |
| physx_s8 | 28.151 | -4.817 | -32.967 [-34.947, -30.769] | 0.984 | 0.000 | 7 / 10 |
| physx_s9 | 26.956 | -3.545 | -30.501 [-32.407, -28.510] | 1.000 | 0.000 | 8 / 4 |

At least one fixed-checkpoint preference reverses across backends.
