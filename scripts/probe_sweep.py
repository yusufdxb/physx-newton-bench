#!/usr/bin/env python3
"""Repeated pure-step throughput probe with per-process VRAM, for one backend
and one num_envs value.

Extends smoke.py: instead of a single timed block (n=1), runs `--repeats`
independently timed blocks in-process (warmup excluded once, kernels already
compiled), so the published throughput carries a mean and std instead of a
point estimate. Also records per-process GPU memory (nvidia-smi
--query-compute-apps for this PID) and torch.cuda.max_memory_allocated, so
VRAM is attributed to the benchmark process rather than the whole GPU.

Backend selection stays purely the Hydra `presets=` token, as in smoke.py.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

parser = argparse.ArgumentParser(description="Repeated throughput probe.")
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--n_steps", type=int, default=100, help="Steps per timed repeat.")
parser.add_argument("--repeats", type=int, default=10, help="Independently timed blocks.")
parser.add_argument("--warmup", type=int, default=10, help="Steps excluded from timing.")
parser.add_argument("--tag", type=str, default="probe_sweep")
parser.add_argument("--out", type=str, required=True, help="JSON result path (written directly, bypassing Kit stdout capture).")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


def proc_vram_mib(pid: int) -> int | None:
    """Per-process GPU memory in MiB from nvidia-smi compute-apps, else None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0] == str(pid):
                return int(parts[1])
    except Exception:
        pass
    return None


def main():
    torch.manual_seed(42)
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = 42
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device

        phys = type(env_cfg.sim.physics).__name__
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

        # warmup (kernel compile / allocation) excluded from timing
        for _ in range(args_cli.warmup):
            with torch.inference_mode():
                env.step(actions)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        sps = []
        for _ in range(args_cli.repeats):
            t0 = time.perf_counter()
            for _ in range(args_cli.n_steps):
                with torch.inference_mode():
                    env.step(actions)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            sps.append(args_cli.n_steps * args_cli.num_envs / dt)

        mean = sum(sps) / len(sps)
        var = sum((x - mean) ** 2 for x in sps) / (len(sps) - 1) if len(sps) > 1 else 0.0
        result = {
            "tag": args_cli.tag, "task": args_cli.task, "backend_cfg": phys,
            "num_envs": args_cli.num_envs, "n_steps": args_cli.n_steps,
            "repeats": args_cli.repeats, "warmup": args_cli.warmup,
            "env_steps_per_sec_all": sps,
            "env_steps_per_sec_mean": mean,
            "env_steps_per_sec_std": var ** 0.5,
            "proc_vram_mib": proc_vram_mib(os.getpid()),
            "torch_max_mem_alloc_mib": torch.cuda.max_memory_allocated() / 2**20,
            "device": str(env_cfg.sim.device), "dt": env_cfg.sim.dt,
            "decimation": env_cfg.decimation, "status": "ok",
        }
        # Write directly to disk: Kit captures the stdout fd, so print() is unreliable here.
        with open(args_cli.out, "w") as f:
            json.dump(result, f, indent=2)
        os.write(2, f"[PROBE-RESULT] {json.dumps(result)}\n".encode())
        env.close()


if __name__ == "__main__":
    main()
