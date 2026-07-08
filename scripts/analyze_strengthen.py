#!/usr/bin/env python3
"""Aggregate the strengthening suite into tables + figures.

Inputs (results/strengthen/): probe_{backend}_{ne}.json (10 timed repeats +
per-process VRAM each), manifest10.csv (20 training runs), and the tensorboard
event files referenced by the manifest. Outputs (same dir): summary.json,
summary.md, fig_scaling.png, fig_learning.png.

Statistics: probe points are mean +- std over the 10 in-process repeats.
Learning curves are IQM over seeds with 95% percentile-bootstrap CIs
(resampling seeds, 2000 reps), reported vs training wall-clock on a common
interpolation grid. IQM = mean of the middle 50% (Agarwal et al. 2021).
"""
import csv
import glob
import json
import os

import numpy as np

BENCH = os.environ.get("BENCH_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SDIR = os.path.join(BENCH, "results", "strengthen")
ENV_COUNTS = [256, 1024, 2048, 4096]
BACKENDS = ["physx", "newton"]
COLORS = {"physx": "#2a78d6", "newton": "#1baf7a"}
RNG = np.random.default_rng(0)
N_BOOT = 2000


def iqm(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    return float(np.mean(x[lo:hi]))


def boot_ci(x, stat=iqm, n=N_BOOT):
    x = np.asarray(x, dtype=float)
    reps = [stat(RNG.choice(x, size=len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def load_probes():
    out = {}
    for b in BACKENDS:
        for ne in ENV_COUNTS:
            p = os.path.join(SDIR, f"probe_{b}_{ne}.json")
            if os.path.exists(p):
                with open(p) as f:
                    out[(b, ne)] = json.load(f)
    return out


def load_manifest():
    rows = []
    with open(os.path.join(SDIR, "manifest10.csv")) as f:
        rows = list(csv.DictReader(f))
    return rows


def load_curves(manifest):
    """Per run: arrays (wall_s, reward, fps) from the tensorboard event file."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    curves = {}
    for r in manifest:
        ld = os.path.expandvars(r["log_dir"].strip())
        ev = sorted(glob.glob(os.path.join(ld, "events.out.tfevents.*")), key=os.path.getmtime)
        if not ev:
            print(f"[WARN] no events for {r['run_name']}")
            continue
        acc = EventAccumulator(ev[-1], size_guidance={"scalars": 0})
        acc.Reload()
        rew = acc.Scalars("Train/mean_reward")
        fps = {e.step: e.value for e in acc.Scalars("Perf/total_fps")}
        t0 = min(e.wall_time for e in rew)
        wall = np.array([e.wall_time - t0 for e in rew])
        reward = np.array([e.value for e in rew])
        fpsv = np.array([fps.get(e.step, np.nan) for e in rew])
        curves[(r["backend"], int(r["seed"]))] = (wall, reward, fpsv)
    return curves


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    probes = load_probes()
    manifest = load_manifest()
    curves = load_curves(manifest)
    summary = {"probes": {}, "training": {}}

    # ---------- probe table ----------
    for (b, ne), d in sorted(probes.items()):
        summary["probes"][f"{b}_{ne}"] = {
            "sps_mean": d["env_steps_per_sec_mean"], "sps_std": d["env_steps_per_sec_std"],
            "proc_vram_mib": d["proc_vram_mib"],
        }

    # ---------- training aggregates ----------
    for b in BACKENDS:
        finals = [curves[(b, s)][1][-1] for s in range(10) if (b, s) in curves]
        fps_all = [np.nanmean(curves[(b, s)][2][10:]) for s in range(10) if (b, s) in curves]
        walls = [float(r["total_wall_s"]) for r in manifest if r["backend"] == b]
        procs = [float(r["proc_peak_mib"]) for r in manifest if r["backend"] == b and r["proc_peak_mib"]]
        summary["training"][b] = {
            "n_seeds": len(finals),
            "final_reward_iqm": iqm(finals), "final_reward_ci": boot_ci(finals),
            "final_reward_per_seed": [float(x) for x in finals],
            "train_fps_iqm": iqm(fps_all), "train_fps_ci": boot_ci(fps_all),
            "total_wall_mean_s": float(np.mean(walls)),
            "proc_vram_peak_mean_mib": float(np.mean(procs)) if procs else None,
            "proc_vram_peak_max_mib": float(np.max(procs)) if procs else None,
        }

    # ---------- figure 1: scaling (two panels, one axis each) ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ENV_COUNTS)
        ax.set_xticklabels([str(x) for x in ENV_COUNTS])
        ax.grid(True, color="#e1e0d9", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c3c2b7")
        ax.tick_params(colors="#898781")
        ax.set_xlabel("num_envs", color="#52514e")
    for b in BACKENDS:
        xs = [ne for ne in ENV_COUNTS if (b, ne) in probes]
        mu = [probes[(b, ne)]["env_steps_per_sec_mean"] / 1e3 for ne in xs]
        sd = [probes[(b, ne)]["env_steps_per_sec_std"] / 1e3 for ne in xs]
        vr = [probes[(b, ne)]["proc_vram_mib"] for ne in xs]
        ax1.errorbar(xs, mu, yerr=sd, color=COLORS[b], linewidth=2, marker="o",
                     markersize=5, capsize=3)
        ax1.annotate(b.capitalize() if b == "newton" else "PhysX",
                     (xs[-1], mu[-1]), xytext=(6, 0), textcoords="offset points",
                     color="#0b0b0b", fontsize=10, va="center")
        ax2.plot(xs, vr, color=COLORS[b], linewidth=2, marker="o", markersize=5)
        ax2.annotate(b.capitalize() if b == "newton" else "PhysX",
                     (xs[-1], vr[-1]), xytext=(6, 0), textcoords="offset points",
                     color="#0b0b0b", fontsize=10, va="center")
    ax1.set_ylabel("throughput (k env-steps/s), mean ± std of 10 repeats", color="#52514e")
    ax2.set_ylabel("per-process VRAM (MiB)", color="#52514e")
    ax1.set_title("Pure-step throughput", color="#0b0b0b", fontsize=11)
    ax2.set_title("Per-process GPU memory", color="#0b0b0b", fontsize=11)
    ax1.set_xlim(right=ENV_COUNTS[-1] * 2.2)
    ax2.set_xlim(right=ENV_COUNTS[-1] * 2.2)
    fig.tight_layout()
    fig.savefig(os.path.join(SDIR, "fig_scaling.png"), facecolor="#fcfcfb")
    plt.close(fig)

    # ---------- figure 2: learning curves, IQM + bootstrap band ----------
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    tmax = min(min(curves[(b, s)][0][-1] for s in range(10) if (b, s) in curves)
               for b in BACKENDS)
    grid = np.linspace(0, tmax, 200)
    for b in BACKENDS:
        mat = np.stack([np.interp(grid, curves[(b, s)][0], curves[(b, s)][1])
                        for s in range(10) if (b, s) in curves])
        center = np.array([iqm(mat[:, i]) for i in range(mat.shape[1])])
        boots = np.empty((N_BOOT, mat.shape[1]))
        for k in range(N_BOOT):
            idx = RNG.integers(0, mat.shape[0], size=mat.shape[0])
            sub = mat[idx]
            boots[k] = [iqm(sub[:, i]) for i in range(sub.shape[1])]
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        ax.fill_between(grid, lo, hi, color=COLORS[b], alpha=0.18, linewidth=0)
        ax.plot(grid, center, color=COLORS[b], linewidth=2)
        ax.annotate("Newton" if b == "newton" else "PhysX",
                    (grid[-1], center[-1]), xytext=(6, 0), textcoords="offset points",
                    color="#0b0b0b", fontsize=10, va="center")
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")
    ax.set_xlabel("training wall-clock (s, sim startup excluded)", color="#52514e")
    ax.set_ylabel("Train/mean_reward: IQM over 10 seeds, 95% bootstrap CI", color="#52514e")
    ax.set_title("Learning curves, identical stock config (n=10 seeds/backend)",
                 color="#0b0b0b", fontsize=11)
    ax.set_xlim(right=tmax * 1.12)
    fig.tight_layout()
    fig.savefig(os.path.join(SDIR, "fig_learning.png"), facecolor="#fcfcfb")
    plt.close(fig)

    # ---------- figure 3: learning curves vs iteration ----------
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    for b in BACKENDS:
        mat = np.stack([curves[(b, s)][1] for s in range(10) if (b, s) in curves])
        it = np.arange(mat.shape[1])
        center = np.array([iqm(mat[:, i]) for i in range(mat.shape[1])])
        boots = np.empty((N_BOOT, mat.shape[1]))
        for k in range(N_BOOT):
            idx = RNG.integers(0, mat.shape[0], size=mat.shape[0])
            sub = mat[idx]
            boots[k] = [iqm(sub[:, i]) for i in range(sub.shape[1])]
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        ax.fill_between(it, lo, hi, color=COLORS[b], alpha=0.18, linewidth=0)
        ax.plot(it, center, color=COLORS[b], linewidth=2)
        ax.annotate("Newton" if b == "newton" else "PhysX",
                    (it[-1], center[-1]), xytext=(6, 0), textcoords="offset points",
                    color="#0b0b0b", fontsize=10, va="center")
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")
    ax.set_xlabel("training iteration", color="#52514e")
    ax.set_ylabel("Train/mean_reward: IQM over 10 seeds, 95% bootstrap CI", color="#52514e")
    ax.set_title("Learning curves vs iteration, identical stock config (n=10 seeds/backend)",
                 color="#0b0b0b", fontsize=11)
    ax.set_xlim(right=len(it) * 1.12)
    fig.tight_layout()
    fig.savefig(os.path.join(SDIR, "fig_learning_iter.png"), facecolor="#fcfcfb")
    plt.close(fig)

    # ---------- markdown ----------
    lines = ["# Strengthening-suite summary\n", "## Pure-step throughput + per-process VRAM\n",
             "| num_envs | PhysX steps/s | Newton steps/s | ratio | PhysX VRAM MiB | Newton VRAM MiB |",
             "|---|---|---|---|---|---|"]
    for ne in ENV_COUNTS:
        if ("physx", ne) in probes and ("newton", ne) in probes:
            p, n = probes[("physx", ne)], probes[("newton", ne)]
            lines.append(
                f"| {ne} | {p['env_steps_per_sec_mean']:,.0f} ± {p['env_steps_per_sec_std']:,.0f} "
                f"| {n['env_steps_per_sec_mean']:,.0f} ± {n['env_steps_per_sec_std']:,.0f} "
                f"| {n['env_steps_per_sec_mean']/p['env_steps_per_sec_mean']:.2f}x "
                f"| {p['proc_vram_mib']} | {n['proc_vram_mib']} |")
    lines.append("\n## Training (n=10 seeds, 2048 envs, 300 iters)\n")
    lines.append("| metric | PhysX | Newton |")
    lines.append("|---|---|---|")
    tp, tn = summary["training"]["physx"], summary["training"]["newton"]
    lines.append(f"| final reward IQM [95% CI] | {tp['final_reward_iqm']:.2f} "
                 f"[{tp['final_reward_ci'][0]:.2f}, {tp['final_reward_ci'][1]:.2f}] "
                 f"| {tn['final_reward_iqm']:.2f} "
                 f"[{tn['final_reward_ci'][0]:.2f}, {tn['final_reward_ci'][1]:.2f}] |")
    lines.append(f"| training fps IQM [95% CI] | {tp['train_fps_iqm']:,.0f} "
                 f"[{tp['train_fps_ci'][0]:,.0f}, {tp['train_fps_ci'][1]:,.0f}] "
                 f"| {tn['train_fps_iqm']:,.0f} "
                 f"[{tn['train_fps_ci'][0]:,.0f}, {tn['train_fps_ci'][1]:,.0f}] |")
    lines.append(f"| total wall mean (s) | {tp['total_wall_mean_s']:.1f} | {tn['total_wall_mean_s']:.1f} |")
    lines.append(f"| proc VRAM peak mean (MiB) | {tp['proc_vram_peak_mean_mib']:.0f} | {tn['proc_vram_peak_mean_mib']:.0f} |")
    with open(os.path.join(SDIR, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(SDIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n".join(lines))
    print(f"\n[DONE] wrote summary.json, summary.md, fig_scaling.png, fig_learning.png in {SDIR}")


if __name__ == "__main__":
    main()
