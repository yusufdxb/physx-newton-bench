#!/usr/bin/env python3
"""Controlled N-step smoke + throughput probe for one physics backend.

Mirrors Isaac Lab's stock zero_agent launch flow (add_launcher_args,
resolve_task_config, launch_simulation) but steps a FIXED number of times
with zero actions and reports wall-clock + env-steps/sec. The physics
backend is selected purely via the Hydra `presets=` override passed on the
command line (e.g. `presets=physx` or `presets=newton`); nothing in this
script touches contact or solver params.
"""

import argparse
import json
import os
import sys
import time

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

parser = argparse.ArgumentParser(description="N-step smoke probe.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--n_steps", type=int, default=50)
parser.add_argument("--warmup", type=int, default=5, help="Steps excluded from timing.")
parser.add_argument("--tag", type=str, default="smoke")
parser.add_argument("--out", type=str, required=True, help="JSON result path (written directly, bypassing Kit stdout capture).")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


def main():
    torch.manual_seed(42)
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device

        phys = type(env_cfg.sim.physics).__name__
        print(f"[SMOKE] task={args_cli.task} backend_cfg={phys} num_envs={args_cli.num_envs} "
              f"device={env_cfg.sim.device} dt={env_cfg.sim.dt} decimation={env_cfg.decimation}")

        env = gym.make(args_cli.task, cfg=env_cfg)
        print(f"[SMOKE] obs_space={env.observation_space} act_space={env.action_space}")
        env.reset()
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

        # warmup (kernel compile / allocation) excluded from timing
        for _ in range(args_cli.warmup):
            with torch.inference_mode():
                env.step(actions)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(args_cli.n_steps):
            with torch.inference_mode():
                env.step(actions)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        env_steps = args_cli.n_steps * args_cli.num_envs
        sps = env_steps / dt
        result = {
            "tag": args_cli.tag, "task": args_cli.task, "backend_cfg": phys,
            "num_envs": args_cli.num_envs, "n_steps": args_cli.n_steps,
            "warmup": args_cli.warmup, "wall_s": dt, "env_steps": env_steps,
            "env_steps_per_sec": sps, "device": str(env_cfg.sim.device),
            "dt": env_cfg.sim.dt, "decimation": env_cfg.decimation, "status": "ok",
        }
        # Write directly to disk: Kit captures the stdout fd, so print() is unreliable here.
        with open(args_cli.out, "w") as f:
            json.dump(result, f, indent=2)
        os.write(2, f"[SMOKE-RESULT] {json.dumps(result)}\n".encode())
        env.close()


if __name__ == "__main__":
    main()
