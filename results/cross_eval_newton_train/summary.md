# Controlled cross-backend checkpoint evaluation

- Policies: 10
- Pairwise preference reversals: 25 / 45
- Spearman rank correlation: -0.2000
- Maximum rank shift: 7
- Probe survives: **True**
- Uncertainty: Per-policy return intervals bootstrap vectorized episodes within one simulator process. They do not replace process-level replication.

| policy | PhysX return | Newton return | Newton - PhysX [95% CI] | PhysX success | Newton success | ranks P/N |
|---|---:|---:|---:|---:|---:|---:|
| newton_s0 | -4.634 | 30.167 | 34.801 [32.994, 36.532] | 0.000 | 1.000 | 4 / 3 |
| newton_s1 | -4.125 | 28.760 | 32.885 [30.315, 35.233] | 0.000 | 0.938 | 2 / 5 |
| newton_s2 | -7.326 | 29.781 | 37.107 [35.190, 38.874] | 0.000 | 0.984 | 10 / 4 |
| newton_s3 | -3.963 | 22.032 | 25.996 [23.611, 28.401] | 0.000 | 0.953 | 1 / 7 |
| newton_s4 | -4.773 | 31.855 | 36.628 [34.254, 38.806] | 0.000 | 0.969 | 5 / 1 |
| newton_s5 | -7.053 | 30.640 | 37.693 [34.528, 40.636] | 0.234 | 0.969 | 9 / 2 |
| newton_s6 | -4.818 | 23.692 | 28.510 [26.524, 30.470] | 0.000 | 0.984 | 6 / 6 |
| newton_s7 | -4.371 | 20.149 | 24.520 [21.620, 27.357] | 0.000 | 0.969 | 3 / 10 |
| newton_s8 | -5.439 | 20.593 | 26.033 [23.769, 28.288] | 0.000 | 0.984 | 7 / 8 |
| newton_s9 | -5.970 | 20.477 | 26.447 [23.391, 29.497] | 0.000 | 0.938 | 8 / 9 |

At least one fixed-checkpoint preference reverses across backends.
