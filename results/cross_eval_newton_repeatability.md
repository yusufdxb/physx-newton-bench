# Cross-backend seed-repeatability control

- Policies: 10
- Policy pairs: 45

| comparison | Spearman | preference reversals | mean success first / second |
|---|---:|---:|---:|
| PhysX seed repeat | 0.8545 | 8 / 45 | 0.0234 / 0.0578 |
| Newton seed repeat | 0.9636 | 2 / 45 | 0.9688 / 0.9766 |
| Cross-backend first | -0.2000 | 25 / 45 | n/a |
| Cross-backend second | -0.4545 | 29 / 45 | n/a |

The deployability collapse repeats, but return ranking is also evaluation-seed sensitive. Do not attribute every pairwise reversal to the backend.
