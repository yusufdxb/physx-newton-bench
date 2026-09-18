# Controlled zero-action baseline

The zero-action controller commands the task's default joint-position offsets.
Both backends completed all 64 episodes by timeout at 1,000 control steps.

| backend | mean return | timeout-success rate | mean episode length |
|---|---:|---:|---:|
| PhysX | 9.8503 | 1.000 | 1,000 |
| Newton | 9.2666 | 1.000 | 1,000 |

This control rules out a globally unstable Newton task instance under the
matched evaluation protocol. It does not establish matched locomotion quality.
