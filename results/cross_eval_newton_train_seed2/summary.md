# Controlled cross-backend checkpoint evaluation

- Policies: 10
- Pairwise preference reversals: 29 / 45
- Spearman rank correlation: -0.4545
- Maximum rank shift: 7
- Probe survives: **True**
- Uncertainty: Per-policy return intervals bootstrap vectorized episodes within one simulator process. They do not replace process-level replication.

| policy | PhysX return | Newton return | Newton - PhysX [95% CI] | PhysX success | Newton success | ranks P/N |
|---|---:|---:|---:|---:|---:|---:|
| newton_s0 | -4.770 | 30.610 | 35.380 [33.325, 37.214] | 0.000 | 0.984 | 5 / 3 |
| newton_s1 | -3.834 | 29.805 | 33.639 [31.102, 35.949] | 0.000 | 0.953 | 1 / 5 |
| newton_s2 | -5.900 | 30.585 | 36.485 [34.888, 37.961] | 0.000 | 0.984 | 8 / 4 |
| newton_s3 | -4.196 | 20.973 | 25.170 [22.547, 27.766] | 0.000 | 0.953 | 2 / 8 |
| newton_s4 | -4.927 | 32.902 | 37.829 [35.771, 39.687] | 0.000 | 1.000 | 7 / 1 |
| newton_s5 | -7.672 | 31.461 | 39.133 [36.289, 41.914] | 0.438 | 0.984 | 9 / 2 |
| newton_s6 | -4.833 | 22.577 | 27.409 [25.376, 29.477] | 0.000 | 1.000 | 6 / 6 |
| newton_s7 | -4.642 | 18.738 | 23.380 [20.437, 26.227] | 0.000 | 0.953 | 3 / 10 |
| newton_s8 | -4.733 | 19.734 | 24.467 [21.874, 26.991] | 0.000 | 0.984 | 4 / 9 |
| newton_s9 | -11.748 | 21.527 | 33.275 [29.463, 37.250] | 0.141 | 0.969 | 10 / 7 |

At least one fixed-checkpoint preference reverses across backends.
