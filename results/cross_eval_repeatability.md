# Cross-backend seed-repeatability control

- Policies: 10
- Policy pairs: 45

| comparison | Spearman | preference reversals | mean success first / second |
|---|---:|---:|---:|
| PhysX seed repeat | 0.6485 | 11 / 45 | 0.9375 / 0.9469 |
| Newton seed repeat | 0.9515 | 3 / 45 | 0.0000 / 0.0000 |
| Cross-backend first | 0.2364 | 18 / 45 | n/a |
| Cross-backend second | 0.5515 | 14 / 45 | n/a |

The deployability collapse repeats, but return ranking is also evaluation-seed sensitive. Do not attribute every pairwise reversal to the backend.
