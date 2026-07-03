#!/usr/bin/env python3
"""Pillar 1: open-loop dynamics recorder for one backend.

Sets a deterministic initial state, disables ALL randomization events and the
base_contact termination (so no mid-tape auto-reset), replays a fixed action
tape (loaded from disk so both backends get bit-identical inputs), and records
base pose, joint state, and foot contact forces to npz. Also dumps the parsed
model params (mass, joint stiffness/damping/armature/friction, limits) to json.
Everything written to disk -- Kit captures stdout.
"""
import argparse, json, os, sys
import numpy as np
import gymnasium as gym
import torch
import warp as wp
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config


def to_t(x, dev):
    """Coerce a data field (warp array / numpy / torch) to a torch tensor."""
    if torch.is_tensor(x):
        return x
    if isinstance(x, wp.array):
        return wp.to_torch(x)
    return torch.as_tensor(np.asarray(x), device=dev)


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, wp.array):
        return x.numpy()
    return np.asarray(x)

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--outdir", required=True)
p.add_argument("--tape", required=True)
p.add_argument("--horizon", type=int, default=300)
add_launcher_args(p)
a, h = p.parse_known_args()
sys.argv = [sys.argv[0]] + h

# Canonical orders -- backends order joints/bodies differently, so we remap
# everything (actions in, state out) to these fixed orders before comparing.
CANON_JOINTS = ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]
FOOT_CANON = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


def make_tape(H, n_act=12):
    """Deterministic tape: 50 steps zero (passive settle), then per-joint sinusoids."""
    t = np.arange(H)
    tape = np.zeros((H, n_act), dtype=np.float32)
    dt = 0.02
    settle = 50
    for j in range(n_act):
        f = 0.6 + 0.15 * j      # distinct freq per joint
        ph = 0.4 * j
        amp = 0.5
        s = amp * np.sin(2 * np.pi * f * (t - settle) * dt + ph)
        s[:settle] = 0.0
        tape[:, j] = s
    return tape


def main():
    env_cfg, _ = resolve_task_config(a.task, "")
    with launch_simulation(env_cfg, a):
        env_cfg.scene.num_envs = 1
        if a.device is not None:
            env_cfg.sim.device = a.device
        # kill all stochastic events
        for ev in ("base_external_force_torque", "reset_base", "reset_robot_joints",
                   "push_robot", "physics_material", "add_base_mass", "base_com"):
            if hasattr(env_cfg.events, ev):
                setattr(env_cfg.events, ev, None)
        # no obs corruption; no falling auto-reset
        if hasattr(env_cfg.observations.policy, "enable_corruption"):
            env_cfg.observations.policy.enable_corruption = False
        if hasattr(env_cfg.terminations, "base_contact"):
            env_cfg.terminations.base_contact = None

        env = gym.make(a.task, cfg=env_cfg)
        dev = env.unwrapped.device
        env.reset()
        robot = env.unwrapped.scene["robot"]
        d = robot.data

        # ---- deterministic canonical initial state ----
        jp = to_t(d.default_joint_pos, dev).clone()
        jv = torch.zeros_like(jp)
        robot.write_joint_state_to_sim(jp, jv)
        try:
            root_pose = to_t(d.default_root_pose, dev).clone()    # (1,7) pos+quat
        except Exception:
            root_pose = to_t(d.default_root_state, dev)[:, :7].clone()
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))
        robot.write_data_to_sim()

        # ---- tape (load if present so both backends share it) ----
        if os.path.exists(a.tape):
            tape = np.load(a.tape)["tape"]
        else:
            tape = make_tape(a.horizon, env.action_space.shape[-1])
            np.savez(a.tape, tape=tape)
        H = tape.shape[0]

        # ---- parsed model params dump ----
        def grab(name):
            try:
                v = getattr(d, name)
                if torch.is_tensor(v) or isinstance(v, wp.array):
                    return to_np(v).tolist()
                return v
            except Exception as e:
                return f"MISSING:{type(e).__name__}"
        model = {k: grab(k) for k in
                 ["default_mass", "joint_stiffness", "joint_damping", "joint_armature",
                  "joint_friction_coefficient", "joint_pos_limits", "default_joint_pos"]}
        model["joint_names"] = list(getattr(d, "joint_names", []))
        model["body_names"] = list(getattr(d, "body_names", []))
        with open(os.path.join(a.outdir, f"model_{a.tag}.json"), "w") as f:
            json.dump(model, f, indent=2)

        # ---- remap tables (env order <-> canonical order) ----
        jn = list(getattr(d, "joint_names", []))
        body_names = list(getattr(d, "body_names", []))
        # action fed in env joint order: env_tape[:,k] = canonical action for joint at env pos k
        env_perm = [CANON_JOINTS.index(j) for j in jn]      # len 12
        env_tape = tape[:, env_perm]
        # state remap env->canonical: canon[:,c] = env[:, joint_names.index(CANON[c])]
        inv_perm = [jn.index(cj) for cj in CANON_JOINTS]
        foot_idx_canon = [body_names.index(f) for f in FOOT_CANON]

        rec = {k: [] for k in ["root_pos", "root_quat", "joint_pos", "joint_vel", "foot_force"]}
        cs = env.unwrapped.scene["contact_forces"]
        for ti in range(H):
            # record state BEFORE applying action (t = ti), remapped to canonical order
            rec["root_pos"].append(to_np(d.root_pos_w)[0].copy())
            rec["root_quat"].append(to_np(d.root_quat_w)[0].copy())
            rec["joint_pos"].append(to_np(d.joint_pos)[0][inv_perm].copy())
            rec["joint_vel"].append(to_np(d.joint_vel)[0][inv_perm].copy())
            nf = to_np(cs.data.net_forces_w)[0]  # (bodies,3)
            ff = np.linalg.norm(nf[foot_idx_canon], axis=-1)   # canonical foot order
            rec["foot_force"].append(ff.copy())
            act = torch.from_numpy(env_tape[ti:ti + 1]).to(dev)
            with torch.inference_mode():
                env.step(act)

        out = {k: np.asarray(v) for k, v in rec.items()}
        out["joint_names_canon"] = np.asarray(CANON_JOINTS)
        out["foot_names_canon"] = np.asarray(FOOT_CANON)
        np.savez(os.path.join(a.outdir, f"traj_{a.tag}.npz"), **out)
        os.write(2, f"[DYN] {a.tag}: H={H} env_joint_order={jn[:3]}... "
                    f"final_root_pos={out['root_pos'][-1]} done\n".encode())
        env.close()


if __name__ == "__main__":
    main()
