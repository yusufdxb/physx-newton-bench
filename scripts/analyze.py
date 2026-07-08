#!/usr/bin/env python3
"""Aggregate results.csv -> per-backend table + reward-vs-wallclock plot."""
import csv, collections, json, statistics, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BENCH = os.environ.get("BENCH_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = 18.0  # ~80% of pooled plateau (~22.5); fixed threshold, both backends

rows = collections.defaultdict(list)   # (backend,seed) -> [(it, reward, wall)]
with open(f"{BENCH}/results/results.csv") as f:
    for r in csv.DictReader(f):
        rows[(r["backend"], r["seed"])].append(
            (int(r["iteration"]), float(r["mean_reward"]), float(r["train_wall_s"])))

# throughput from the 2048-env pure-step probes
thru = {}
for b in ("physx", "newton"):
    with open(f"{BENCH}/results/probe_{b}_2048.json") as f:
        thru[b] = json.load(f)["env_steps_per_sec"]

def per_run(d):
    plateau = statistics.mean(v for _, v, _ in d[-50:])
    final = d[-1][1]
    cross = next(((it, w) for it, v, w in d if v >= R), None)
    iters_to_R = cross[0] if cross else None
    wall_to_R = cross[1] if cross else None
    return plateau, final, iters_to_R, wall_to_R

agg = collections.defaultdict(lambda: collections.defaultdict(list))
per = {}
for k in sorted(rows):
    b, s = k
    plat, fin, itr, wtr = per_run(rows[k])
    per[k] = (plat, fin, itr, wtr)
    agg[b]["plateau"].append(plat); agg[b]["final"].append(fin)
    agg[b]["iters_to_R"].append(itr); agg[b]["wall_to_R"].append(wtr)
    agg[b]["throughput"].append(thru[b])

def mr(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return None
    return statistics.mean(vals), min(vals), max(vals)

print(f"R = {R}")
print(f"{'run':12}{'plateau':>9}{'final':>8}{'iters_to_R':>11}{'wall_to_R':>10}")
for k in sorted(per):
    p, fin, itr, wtr = per[k]
    print(f"{k[0]+'_s'+k[1]:12}{p:9.2f}{fin:8.2f}{str(itr):>11}{(f'{wtr:.1f}' if wtr else 'NONE'):>10}")

print("\n=== per-backend mean [min, max] over 3 seeds ===")
hdr = ["throughput(steps/s)", "iters_to_R", "wall_to_R(s)", "final_reward", "plateau"]
for b in ("physx", "newton"):
    print(f"\n{b}:")
    for m in ["throughput", "iters_to_R", "wall_to_R", "final", "plateau"]:
        res = mr(agg[b][m])
        if res: print(f"  {m:12}: {res[0]:9.2f}  [{res[1]:.2f}, {res[2]:.2f}]")

# save aggregate json
out = {"R": R, "throughput_steps_per_s": thru,
       "per_backend": {b: {m: mr(agg[b][m]) for m in
                           ["throughput","iters_to_R","wall_to_R","final","plateau"]}
                       for b in ("physx","newton")}}
with open(f"{BENCH}/results/aggregate.json", "w") as f:
    json.dump(out, f, indent=2)

# ---- plot: mean reward vs wall-clock, shaded min/max across seeds ----
N = 300
fig, ax = plt.subplots(figsize=(8, 5))
colors = {"physx": "#2a78d6", "newton": "#1baf7a"}
for b in ("physx", "newton"):
    seeds = [k for k in rows if k[0] == b]
    rew = [[rows[k][i][1] for k in seeds] for i in range(N)]
    wall = [[rows[k][i][2] for k in seeds] for i in range(N)]
    mean_r = [statistics.mean(x) for x in rew]
    min_r = [min(x) for x in rew]; max_r = [max(x) for x in rew]
    mean_w = [statistics.mean(x) for x in wall]
    ax.plot(mean_w, mean_r, color=colors[b], label=f"{b} (mean of 3 seeds)", lw=2)
    ax.fill_between(mean_w, min_r, max_r, color=colors[b], alpha=0.2)
ax.axhline(R, color="gray", ls="--", lw=1, label=f"R = {R}")
ax.set_xlabel("Training wall-clock (s, excludes sim startup)")
ax.set_ylabel("Mean reward")
ax.set_title("Go2 flat velocity-tracking: PhysX vs Newton (MuJoCo-Warp)\n"
             "Isaac Lab 3.0, 2048 envs, 300 iters, identical config")
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{BENCH}/results/reward_vs_wallclock.png", dpi=140)
print(f"\n[plot] {BENCH}/results/reward_vs_wallclock.png")
