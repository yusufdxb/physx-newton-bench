# Controlled cross-backend checkpoint evaluation

- Policies: 3
- Pairwise preference reversals: 1 / 3
- Spearman rank correlation: 0.5000
- Maximum rank shift: 1
- Probe survives: **True**
- Uncertainty: Per-policy return intervals bootstrap vectorized episodes within one simulator process. They do not replace process-level replication.

| policy | PhysX return | Newton return | Newton - PhysX [95% CI] | PhysX success | Newton success | ranks P/N |
|---|---:|---:|---:|---:|---:|---:|
| physx_s0 | 30.952 | 29.237 | -1.715 [-4.010, 0.663] | 0.984 | 1.000 | 2 / 1 |
| physx_s1 | 21.309 | 23.232 | 1.923 [-1.332, 5.073] | 0.750 | 0.938 | 3 / 3 |
| physx_s2 | 32.683 | 28.785 | -3.898 [-6.095, -1.553] | 0.969 | 0.984 | 1 / 2 |

At least one fixed-checkpoint preference reverses across backends.
