#!/usr/bin/env python3
"""Empirical confirmation for the gain-representation finding (Newton backend).

Static tracing says: all 12 Go2 joints import as EFFORT-mode (USD drive gains
are 0/0), so MJWarp builds ZERO PD actuators (mjw_model.nu == 0) and the cfg
gains that Isaac Lab writes into Model.joint_target_ke/kd are never consumed
by the solver. This script checks that on a LIVE model: launches the task with
presets=newton, 1 env, and dumps mjw_model.nu, joint_target_mode, and
joint_target_ke/kd to JSON (disk, not stdout — Kit captures the fd).
"""

import argparse
import json
import os
import sys

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


def arr(x):
    try:
        return x.numpy().tolist()
    except Exception:
        try:
            return list(x)
        except Exception:
            return repr(x)


def main():
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = 1
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()

        from isaaclab_newton.physics.newton_manager import NewtonManager
        out = {"backend_cfg": type(env_cfg.sim.physics).__name__}
        solver, model = NewtonManager._solver, NewtonManager._model
        out["solver_class"] = type(solver).__name__
        try:
            out["mjw_model_nu"] = int(solver.mjw_model.nu)
        except Exception as e:
            out["mjw_model_nu"] = f"ERROR:{e!r}"
        for field in ("joint_target_mode", "joint_target_ke", "joint_target_kd"):
            out[field] = arr(getattr(model, field, None))
        try:
            import newton
            out["JointTargetMode_enum"] = {m.name: int(m.value) for m in newton.JointTargetMode}
        except Exception:
            try:
                from newton._src.sim.enums import JointTargetMode
                out["JointTargetMode_enum"] = {m.name: int(m.value) for m in JointTargetMode}
            except Exception as e:
                out["JointTargetMode_enum"] = f"ERROR:{e!r}"

        with open(args_cli.out, "w") as f:
            json.dump(out, f, indent=2)
        os.write(2, f"[CHECK-RESULT] {json.dumps(out)[:2000]}\n".encode())
        env.close()


if __name__ == "__main__":
    main()
