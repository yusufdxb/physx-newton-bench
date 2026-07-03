#!/usr/bin/env python3
"""Pillar 1 analysis: PhysX vs Newton open-loop trajectory divergence."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(
    os.environ.get("BENCH_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "pillar1")
DT = 0.02            # control step (sim.dt 0.005 * decimation 4)
SETTLE = 50
P = np.load(f"{D}/traj_physx.npz", allow_pickle=True)
N = np.load(f"{D}/traj_newton.npz", allow_pickle=True)
joints = [str(x) for x in P["joint_names_canon"]]
feet = [str(x) for x in P["foot_names_canon"]]
H = P["root_pos"].shape[0]
t = np.arange(H) * DT


def quat_angle_deg(q1, q2):
    d = np.abs(np.sum(q1 * q2, axis=-1)).clip(-1, 1)
    return np.degrees(2 * np.arccos(d))


# --- base divergence ---
base_pos_err = np.linalg.norm(P["root_pos"] - N["root_pos"], axis=-1)      # (H,)
base_ori_err = quat_angle_deg(P["root_quat"], N["root_quat"])             # (H,)
# --- joint divergence ---
jp_err = np.abs(P["joint_pos"] - N["joint_pos"])                          # (H,12)
jp_rmse = np.sqrt((jp_err ** 2).mean(axis=0))                             # (12,)
jv_err = np.abs(P["joint_vel"] - N["joint_vel"])
# --- contact force ---
ff_p, ff_n = P["foot_force"], N["foot_force"]                             # (H,4)
foot_corr = [np.corrcoef(ff_p[:, i], ff_n[:, i])[0, 1] for i in range(4)]
tot_p, tot_n = ff_p.sum(1), ff_n.sum(1)
tot_corr = np.corrcoef(tot_p, tot_n)[0, 1]

# --- divergence time ---
def first_cross(arr, thr):
    idx = np.where(arr > thr)[0]
    return float(idx[0] * DT) if len(idx) else None
div_pos_5cm = first_cross(base_pos_err, 0.05)
div_ori_5deg = first_cross(base_ori_err, 5.0)

# --- model param diff (remap masses to canonical body order) ---
mp = json.load(open(f"{D}/model_physx.json")); mn = json.load(open(f"{D}/model_newton.json"))
def remap_mass(m):
    names = m["body_names"]; mass = np.array(m["default_mass"], dtype=float).flatten()
    order = sorted(range(len(names)), key=lambda i: names[i])
    return [names[i] for i in order], mass[order]
bp, mass_p = remap_mass(mp); bn, mass_n = remap_mass(mn)
mass_match = (bp == bn)
mass_maxdiff = float(np.abs(mass_p - mass_n).max()) if mass_match else None

print("=== Pillar 1: dynamics equivalence ===")
print(f"horizon {H} steps ({t[-1]:.1f}s), settle 0-{SETTLE*DT:.1f}s then sinusoid")
print(f"base pos err: settle-end={base_pos_err[SETTLE]:.4f}m  final={base_pos_err[-1]:.4f}m  max={base_pos_err.max():.4f}m")
print(f"base ori err: settle-end={base_ori_err[SETTLE]:.2f}deg final={base_ori_err[-1]:.2f}deg max={base_ori_err.max():.2f}deg")
print(f"divergence time: >5cm at {div_pos_5cm}s, >5deg at {div_ori_5deg}s")
print(f"joint pos RMSE (rad) mean={jp_rmse.mean():.4f} max={jp_rmse.max():.4f} ({joints[jp_rmse.argmax()]})")
print(f"settle-phase joint pos max-abs-err (rad)={jp_err[:SETTLE].max():.4f}  (PD-hold agreement)")
print(f"foot force corr per foot {dict(zip(feet,[round(c,3) for c in foot_corr]))}  total={tot_corr:.3f}")
print(f"mass canonical match={mass_match} maxdiff={mass_maxdiff} kg ; total mass P={mass_p.sum():.3f} N={mass_n.sum():.3f}")
print(f"joint_stiffness physx={mp['joint_stiffness'][0][0]} newton={mn['joint_stiffness'][0][0]} ; "
      f"damping physx={mp['joint_damping'][0][0]} newton={mn['joint_damping'][0][0]}")

summary = {
    "horizon_steps": int(H), "horizon_s": float(t[-1]),
    "base_pos_err_settle_m": float(base_pos_err[SETTLE]),
    "base_pos_err_final_m": float(base_pos_err[-1]), "base_pos_err_max_m": float(base_pos_err.max()),
    "base_ori_err_final_deg": float(base_ori_err[-1]), "base_ori_err_max_deg": float(base_ori_err.max()),
    "div_time_5cm_s": div_pos_5cm, "div_time_5deg_s": div_ori_5deg,
    "joint_pos_rmse_mean_rad": float(jp_rmse.mean()), "joint_pos_rmse_max_rad": float(jp_rmse.max()),
    "joint_pos_rmse_max_joint": joints[int(jp_rmse.argmax())],
    "settle_joint_maxabs_rad": float(jp_err[:SETTLE].max()),
    "foot_force_corr": dict(zip(feet, [float(c) for c in foot_corr])),
    "foot_force_corr_total": float(tot_corr),
    "mass_canonical_match": bool(mass_match), "mass_maxdiff_kg": mass_maxdiff,
    "total_mass_physx": float(mass_p.sum()), "total_mass_newton": float(mass_n.sum()),
    "joint_stiffness_physx": mp["joint_stiffness"][0][0], "joint_stiffness_newton": mn["joint_stiffness"][0][0],
    "joint_damping_physx": mp["joint_damping"][0][0], "joint_damping_newton": mn["joint_damping"][0][0],
}
json.dump(summary, open(f"{D}/dynamics_summary.json", "w"), indent=2)

# --- plots ---
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].plot(t, base_pos_err, "k"); ax[0, 0].axvline(SETTLE*DT, color="gray", ls=":")
ax[0, 0].axhline(0.05, color="r", ls="--", lw=0.8); ax[0, 0].set_title("Base position error |PhysX - Newton|")
ax[0, 0].set_xlabel("s"); ax[0, 0].set_ylabel("m")
ax[0, 1].plot(t, base_ori_err, "k"); ax[0, 1].axvline(SETTLE*DT, color="gray", ls=":")
ax[0, 1].axhline(5, color="r", ls="--", lw=0.8); ax[0, 1].set_title("Base orientation error")
ax[0, 1].set_xlabel("s"); ax[0, 1].set_ylabel("deg")
j = int(jp_rmse.argmax())
ax[1, 0].plot(t, P["joint_pos"][:, j], "b", label="PhysX"); ax[1, 0].plot(t, N["joint_pos"][:, j], "r", label="Newton")
ax[1, 0].axvline(SETTLE*DT, color="gray", ls=":"); ax[1, 0].legend()
ax[1, 0].set_title(f"Worst joint: {joints[j]}"); ax[1, 0].set_xlabel("s"); ax[1, 0].set_ylabel("rad")
ax[1, 1].bar(range(12), jp_rmse); ax[1, 1].set_xticks(range(12))
ax[1, 1].set_xticklabels([j.replace("_joint", "") for j in joints], rotation=90, fontsize=7)
ax[1, 1].set_title("Per-joint position RMSE"); ax[1, 1].set_ylabel("rad")
fig.suptitle("Pillar 1: PhysX vs Newton open-loop dynamics divergence (Go2, identical tape, no training)")
fig.tight_layout()
fig.savefig(f"{D}/dynamics_divergence.png", dpi=140)
print(f"\n[plot] {D}/dynamics_divergence.png")
print(f"[json] {D}/dynamics_summary.json")
