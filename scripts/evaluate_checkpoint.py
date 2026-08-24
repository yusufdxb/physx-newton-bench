#!/usr/bin/env python3
"""Evaluate one fixed RSL-RL checkpoint under one Isaac Lab physics backend.

The probe removes observation corruption and non-shared startup or fault
randomization while retaining the shared base and joint reset distributions.
A PhysX/Newton pair therefore differs only in the selected backend
implementation. It writes episode-level returns, lengths, and termination
outcomes for offline rank-stability analysis.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--policy-id", required=True)
parser.add_argument("--backend-id", required=True, choices=("physx", "newton"))
parser.add_argument("--controller", choices=("policy", "zero"), default="policy")
parser.add_argument(
    "--policy-joint-order",
    choices=("physx", "newton"),
    default=None,
    help="Expose joint observations and actions in the checkpoint's training-backend order.",
)
parser.add_argument("--output", required=True)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--num-episodes", type=int, default=64)
parser.add_argument("--seed", type=int, default=20260726)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="RSL-RL agent configuration entry point.",
)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

_DISABLED_EVENTS = (
    "base_external_force_torque",
    "push_robot",
    "physics_material",
    "add_base_mass",
    "base_com",
)

_POLICY_JOINT_ORDERS = {
    "physx": [
        "FL_hip_joint",
        "FR_hip_joint",
        "RL_hip_joint",
        "RR_hip_joint",
        "FL_thigh_joint",
        "FR_thigh_joint",
        "RL_thigh_joint",
        "RR_thigh_joint",
        "FL_calf_joint",
        "FR_calf_joint",
        "RL_calf_joint",
        "RR_calf_joint",
    ],
    "newton": [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ],
}


def _set_policy_joint_order(env_cfg, order_name: str | None) -> list[str] | None:
    if order_name is None:
        return None
    joint_names = _POLICY_JOINT_ORDERS[order_name]
    action_cfg = env_cfg.actions.joint_pos
    action_cfg.joint_names = joint_names
    action_cfg.preserve_order = True
    for term_name in ("joint_pos", "joint_vel"):
        term_cfg = getattr(env_cfg.observations.policy, term_name)
        asset_cfg = term_cfg.params.setdefault("asset_cfg", SceneEntityCfg("robot"))
        asset_cfg.joint_names = joint_names
        asset_cfg.preserve_order = True
    return joint_names


def _index_spec(value):
    if isinstance(value, slice):
        return {"start": value.start, "stop": value.stop, "step": value.step}
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _runtime_contract(env) -> dict:
    """Capture policy-facing runtime ordering, not only unresolved config."""
    robot = env.scene["robot"]
    actions = {}
    for term_name in env.action_manager.active_terms:
        term = env.action_manager.get_term(term_name)
        actions[term_name] = {
            "dimension": int(term.action_dim),
            "joint_names": list(getattr(term, "_joint_names", [])),
            "joint_ids": _index_spec(getattr(term, "_joint_ids", [])),
        }

    observations = {}
    for group_name, term_names in env.observation_manager.active_terms.items():
        term_dims = env.observation_manager.group_obs_term_dim[group_name]
        term_cfgs = env.observation_manager._group_obs_term_cfgs[group_name]
        terms = []
        for term_name, term_dim, term_cfg in zip(term_names, term_dims, term_cfgs, strict=True):
            asset_cfg = term_cfg.params.get("asset_cfg")
            terms.append(
                {
                    "name": term_name,
                    "dimension": list(term_dim),
                    "function": f"{term_cfg.func.__module__}:{term_cfg.func.__qualname__}",
                    "joint_names": (
                        list(asset_cfg.joint_names)
                        if asset_cfg is not None and asset_cfg.joint_names is not None
                        else None
                    ),
                    "joint_ids": (
                        _index_spec(asset_cfg.joint_ids) if asset_cfg is not None else None
                    ),
                    "body_names": (
                        list(asset_cfg.body_names)
                        if asset_cfg is not None and asset_cfg.body_names is not None
                        else None
                    ),
                    "body_ids": (
                        _index_spec(asset_cfg.body_ids) if asset_cfg is not None else None
                    ),
                }
            )
        observations[group_name] = {
            "dimension": list(env.observation_manager.group_obs_dim[group_name]),
            "concatenate": bool(env.observation_manager.group_obs_concatenate[group_name]),
            "terms": terms,
        }

    return {
        "robot_joint_names": list(robot.joint_names),
        "robot_body_names": list(robot.body_names),
        "actions": actions,
        "observations": observations,
    }


def _provenance() -> dict:
    packages = {}
    for name in (
        "isaaclab",
        "isaaclab_tasks",
        "isaacsim",
        "newton",
        "mujoco-warp",
        "rsl-rl-lib",
        "torch",
    ):
        with contextlib.suppress(metadata.PackageNotFoundError):
            packages[name] = metadata.version(name)
    isaaclab_path = os.environ.get("ISAACLAB_PATH")
    git_commit = None
    if isaaclab_path and os.path.isdir(os.path.join(isaaclab_path, ".git")):
        completed = subprocess.run(
            ["git", "-C", isaaclab_path, "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            git_commit = completed.stdout.strip()
    return {
        "isaaclab_path": isaaclab_path,
        "isaaclab_git_commit": git_commit,
        "packages": packages,
    }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_observation(observation):
    if isinstance(observation, tuple):
        observation = observation[0]
    if isinstance(observation, dict):
        return observation.get("policy", next(iter(observation.values())))
    return observation


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    """Run the controlled evaluation and write a JSON artifact."""
    with launch_simulation(env_cfg, args_cli):
        checkpoint = os.path.abspath(args_cli.checkpoint) if args_cli.checkpoint else None
        output = os.path.abspath(args_cli.output)
        if args_cli.controller == "policy" and (checkpoint is None or not os.path.isfile(checkpoint)):
            raise FileNotFoundError(checkpoint or "<missing --checkpoint>")
        if args_cli.num_episodes < 1 or args_cli.num_envs < 1:
            raise ValueError("num_envs and num_episodes must both be positive")

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        agent_cfg.seed = args_cli.seed
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device

        disabled_events: list[str] = []
        if getattr(env_cfg, "events", None) is not None:
            for event_name in _DISABLED_EVENTS:
                if hasattr(env_cfg.events, event_name):
                    setattr(env_cfg.events, event_name, None)
                    disabled_events.append(event_name)
        policy_obs_cfg = getattr(getattr(env_cfg, "observations", None), "policy", None)
        if policy_obs_cfg is not None and hasattr(policy_obs_cfg, "enable_corruption"):
            policy_obs_cfg.enable_corruption = False
        policy_joint_names = _set_policy_joint_order(env_cfg, args_cli.policy_joint_order)

        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)
        runtime_contract = _runtime_contract(env.unwrapped)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        policy = None
        if args_cli.controller == "policy":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(checkpoint)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

        observation = _policy_observation(env.get_observations())
        episode_return = torch.zeros(args_cli.num_envs, device=env.unwrapped.device)
        episode_length = torch.zeros(args_cli.num_envs, device=env.unwrapped.device)
        returns: list[float] = []
        lengths: list[int] = []
        successes: list[bool] = []
        start = time.perf_counter()

        with torch.inference_mode():
            while len(returns) < args_cli.num_episodes:
                if policy is None:
                    actions = torch.zeros(
                        (args_cli.num_envs, env.num_actions),
                        device=env.unwrapped.device,
                    )
                else:
                    actions = policy(observation)
                observation, reward, dones, extras = env.step(actions)
                observation = _policy_observation(observation)
                episode_return += reward
                episode_length += 1

                done_indices = dones.nonzero(as_tuple=False).flatten()
                time_outs = extras.get("time_outs")
                for env_index in done_indices.tolist():
                    returns.append(float(episode_return[env_index].item()))
                    lengths.append(int(episode_length[env_index].item()))
                    successes.append(bool(time_outs[env_index].item()) if time_outs is not None else False)
                    episode_return[env_index] = 0.0
                    episode_length[env_index] = 0.0

        elapsed = time.perf_counter() - start
        env.close()
        limit = args_cli.num_episodes
        returns = returns[:limit]
        lengths = lengths[:limit]
        successes = successes[:limit]

        payload = {
            "schema_version": 1,
            "task": args_cli.task,
            "policy_id": args_cli.policy_id,
            "backend": args_cli.backend_id,
            "controller": args_cli.controller,
            "checkpoint": checkpoint,
            "checkpoint_sha256": _sha256(checkpoint) if checkpoint else None,
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "num_episodes": len(returns),
            "controlled_eval": {
                "disabled_events": disabled_events,
                "observation_corruption": False,
                "backend_override": f"presets={args_cli.backend_id}",
                "hydra_overrides": hydra_args,
                "policy_joint_order": args_cli.policy_joint_order,
                "policy_joint_names": policy_joint_names,
            },
            "runtime_contract": runtime_contract,
            "provenance": _provenance(),
            "episode_return": returns,
            "episode_length_steps": lengths,
            "success": successes,
            "mean_return": float(np.mean(returns)),
            "mean_episode_length_steps": float(np.mean(lengths)),
            "success_rate": float(np.mean(successes)),
            "rollout_wall_s": elapsed,
        }
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.write(
            2,
            (
                f"[CROSS-EVAL] policy={args_cli.policy_id} backend={args_cli.backend_id} "
                f"episodes={len(returns)} mean_return={payload['mean_return']:.6f} "
                f"success_rate={payload['success_rate']:.4f} output={output}\n"
            ).encode(),
        )


if __name__ == "__main__":
    main()
