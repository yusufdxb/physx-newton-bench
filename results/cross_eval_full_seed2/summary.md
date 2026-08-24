# Controlled cross-backend checkpoint evaluation

- Policies: 10
- Pairwise preference reversals: 14 / 45
- Spearman rank correlation: 0.5515
- Maximum rank shift: 5
- Probe survives: **True**
- Uncertainty: Per-policy return intervals bootstrap vectorized episodes within one simulator process. They do not replace process-level replication.

| policy | PhysX return | Newton return | Newton - PhysX [95% CI] | PhysX success | Newton success | ranks P/N |
|---|---:|---:|---:|---:|---:|---:|
| physx_s0 | 31.247 | -3.648 | -34.895 [-36.757, -33.006] | 1.000 | 0.000 | 4 / 4 |
| physx_s1 | 24.133 | -4.332 | -28.465 [-30.750, -26.198] | 0.906 | 0.000 | 10 / 8 |
| physx_s2 | 29.082 | -4.684 | -33.766 [-36.656, -30.598] | 0.891 | 0.000 | 7 / 9 |
| physx_s3 | 25.731 | -3.844 | -29.575 [-32.507, -26.349] | 0.828 | 0.000 | 9 / 7 |
| physx_s4 | 31.405 | -3.694 | -35.099 [-36.799, -33.204] | 0.984 | 0.000 | 3 / 5 |
| physx_s5 | 32.283 | -3.451 | -35.734 [-37.204, -34.175] | 0.984 | 0.000 | 2 / 2 |
| physx_s6 | 29.542 | -3.363 | -32.905 [-35.343, -30.228] | 0.891 | 0.000 | 6 / 1 |
| physx_s7 | 34.532 | -3.631 | -38.164 [-39.485, -36.742] | 0.984 | 0.000 | 1 / 3 |
| physx_s8 | 30.612 | -5.030 | -35.642 [-37.471, -33.596] | 1.000 | 0.000 | 5 / 10 |
| physx_s9 | 28.535 | -3.835 | -32.370 [-34.751, -29.873] | 1.000 | 0.000 | 8 / 6 |

At least one fixed-checkpoint preference reverses across backends.
