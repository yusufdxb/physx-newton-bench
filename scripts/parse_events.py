#!/usr/bin/env python3
"""Parse rsl_rl tensorboard event files into a per-iteration CSV.

Extracts, per backend/seed run, the iteration index, Train/mean_reward,
event wall_time (zeroed at the run's first scalar = training-loop seconds,
excludes sim startup), and Perf/total_fps. Reads the manifest produced by
run_bench.sh to associate each log_dir with (backend, seed).
"""
import csv
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

_BENCH = os.environ.get("BENCH_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_BENCH, "results", "manifest.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_BENCH, "results", "results.csv")


def load_run(log_dir):
    ev = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not ev:
        return None
    ev.sort(key=os.path.getmtime)
    acc = EventAccumulator(ev[-1], size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if "Train/mean_reward" not in tags:
        return {"_error": f"no Train/mean_reward in {ev[-1]} (tags={tags})"}
    reward = {e.step: (e.value, e.wall_time) for e in acc.Scalars("Train/mean_reward")}
    fps = {e.step: e.value for e in acc.Scalars("Perf/total_fps")} if "Perf/total_fps" in tags else {}
    t0 = min(w for _, w in reward.values())
    rows = []
    for it in sorted(reward):
        val, wt = reward[it]
        rows.append((it, val, wt - t0, fps.get(it, "")))
    return rows


def main():
    runs = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            runs.append(r)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "seed", "run_name", "iteration",
                    "mean_reward", "train_wall_s", "total_fps"])
        for r in runs:
            ld = os.path.expandvars(r["log_dir"].strip())
            if not ld or not os.path.isdir(ld):
                print(f"[WARN] missing log_dir for {r['run_name']}: {ld!r}")
                continue
            data = load_run(ld)
            if data is None:
                print(f"[WARN] no event file in {ld}")
                continue
            if isinstance(data, dict) and "_error" in data:
                print(f"[WARN] {r['run_name']}: {data['_error']}")
                continue
            for it, val, wt, fps in data:
                w.writerow([r["backend"], r["seed"], r["run_name"], it,
                            f"{val:.6f}", f"{wt:.4f}", fps])
            print(f"[OK] {r['run_name']}: {len(data)} iters, "
                  f"final_reward={data[-1][1]:.3f}, train_wall={data[-1][2]:.1f}s")
    print(f"[DONE] wrote {OUT}")


if __name__ == "__main__":
    main()
