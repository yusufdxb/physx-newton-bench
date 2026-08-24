# Controlled cross-backend checkpoint evaluation

- Policies: 3
- Pairwise preference reversals: 2 / 3
- Spearman rank correlation: -0.5000
- Maximum rank shift: 2
- Probe survives: **True**
- Uncertainty: Per-policy return intervals bootstrap vectorized episodes within one simulator process. They do not replace process-level replication.

| policy | PhysX return | Newton return | Newton - PhysX [95% CI] | PhysX success | Newton success | ranks P/N |
|---|---:|---:|---:|---:|---:|---:|
| newton_s0 | 23.817 | 30.063 | 6.246 [3.292, 9.198] | 0.938 | 1.000 | 3 / 1 |
| newton_s1 | 25.277 | 29.514 | 4.237 [0.638, 7.853] | 0.922 | 0.953 | 2 / 3 |
| newton_s2 | 26.162 | 30.031 | 3.868 [1.725, 5.977] | 1.000 | 0.984 | 1 / 2 |

At least one fixed-checkpoint preference reverses across backends.
