# Controlled cross-backend checkpoint evaluation

- Policies: 3
- Pairwise preference reversals: 1 / 3
- Spearman rank correlation: 0.5000
- Maximum rank shift: 1
- Probe survives: **True**

| policy | PhysX return | Newton return | Newton - PhysX [95% CI] | PhysX success | Newton success | ranks P/N |
|---|---:|---:|---:|---:|---:|---:|
| physx_s0 | 30.952 | -3.925 | -34.877 [-36.756, -32.831] | 0.984 | 0.000 | 2 / 1 |
| physx_s1 | 21.309 | -4.838 | -26.146 [-28.754, -23.611] | 0.750 | 0.000 | 3 / 3 |
| physx_s2 | 32.683 | -4.721 | -37.404 [-39.003, -35.510] | 0.969 | 0.000 | 1 / 2 |

At least one fixed-checkpoint preference reverses across backends.
