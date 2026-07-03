#!/usr/bin/env python3
"""Introspect the Isaac Lab 3.0 articulation/contact API at runtime."""
import argparse, sys
import gymnasium as gym
import torch
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
add_launcher_args(p)
a, h = p.parse_known_args()
sys.argv = [sys.argv[0]] + h

env_cfg, _ = resolve_task_config(a.task, "")
with launch_simulation(env_cfg, a):
    env_cfg.scene.num_envs = 1
    env = gym.make(a.task, cfg=env_cfg)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    d = robot.data
    def show(obj, names):
        for n in names:
            try:
                v = getattr(obj, n)
                shape = tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
                import sys as _s; _s.stderr.write(f"  {n}: {shape}\n")
            except Exception as e:
                import sys as _s; _s.stderr.write(f"  {n}: MISSING ({type(e).__name__})\n")
    import sys as _s
    _s.stderr.write("=== data fields ===\n")
    show(d, ["root_pos_w","root_quat_w","root_state_w","root_link_state_w","root_link_pose_w",
             "joint_pos","joint_vel","default_joint_pos","default_root_state","body_names",
             "joint_names","body_pos_w","root_lin_vel_w","root_ang_vel_w"])
    _s.stderr.write("=== write methods ===\n")
    wm = [m for m in dir(robot) if m.startswith("write_")]
    _s.stderr.write("  " + ", ".join(wm) + "\n")
    _s.stderr.write(f"=== joint_names ===\n  {getattr(d,'joint_names',None)}\n")
    _s.stderr.write(f"=== body_names ===\n  {getattr(d,'body_names',None)}\n")
    # contact sensor
    try:
        cs = env.unwrapped.scene["contact_forces"]
        _s.stderr.write("=== contact sensor data fields ===\n")
        show(cs.data, ["net_forces_w","net_forces_w_history","force_matrix_w"])
        _s.stderr.write(f"  body_names: {getattr(cs,'body_names',None)}\n")
    except Exception as e:
        _s.stderr.write(f"contact sensor: {e}\n")
    _s.stderr.write(f"=== action space: {env.action_space.shape} obs: {env.observation_space} ===\n")
    env.close()
