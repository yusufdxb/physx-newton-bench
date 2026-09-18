#!/usr/bin/env python3
"""Semantic guardrail for the PhysX/Newton strengthening configs.

This script compares the committed resolved configs for the two benchmark arms
and writes a machine-readable manifest plus a compact Markdown report. It is
intentionally conservative: backend implementation blocks are allowed to differ,
but task semantics, randomization, rewards, terminations, policy, and PPO fields
must match for publication-style aggregate claims.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import yaml


BENCH = os.environ.get("BENCH_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PARAMS = os.path.join(BENCH, "results", "strengthen", "params")
DEFAULT_OUT = os.path.join(BENCH, "results", "strengthen")
BACKENDS = ("physx", "newton")
EXPECTED_SEEDS = set(range(10))
EXPECTED_ITERS = 300


@dataclass(frozen=True)
class Difference:
    path: str
    physx: Any
    newton: Any
    classification: str
    rationale: str


def load_yaml(path: str) -> Any:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def normalize(value: Any) -> Any:
    if isinstance(value, slice):
        return {"__slice__": [normalize(value.start), normalize(value.stop), normalize(value.step)]}
    if isinstance(value, tuple):
        return [normalize(v) for v in value]
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    return value


def get_path(root: Any, path: str, default: Any = None) -> Any:
    cur = root
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def recursive_diff(a: Any, b: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    out: list[tuple[str, Any, Any]] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            next_path = f"{path}.{key}" if path else key
            if key not in a:
                out.append((next_path, "<missing>", b[key]))
            elif key not in b:
                out.append((next_path, a[key], "<missing>"))
            else:
                out.extend(recursive_diff(a[key], b[key], next_path))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((f"{path}.length", len(a), len(b)))
        for idx, (x, y) in enumerate(zip(a, b)):
            out.extend(recursive_diff(x, y, f"{path}[{idx}]"))
        return out
    if a != b:
        out.append((path, a, b))
    return out


def summarize_value(value: Any) -> str:
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else repr(value)
    return text if len(text) <= 180 else text[:177] + "..."


def classify(path: str, physx: Any, newton: Any) -> tuple[str, str]:
    if path.startswith("sim.physics"):
        return "required backend-specific", "Backend manager and solver/contact buffer settings are selected by the physics preset."
    if path == "scene.contact_forces.class_type":
        return "required backend-specific", "Contact sensor implementation follows the selected backend package."
    if path.startswith("scene.contact_forces.") and path.endswith("_shape_prim_expr"):
        return "harmless representation", "Newton dumps explicit empty contact sensor shape filters where PhysX omits them."
    if path == "log_dir":
        return "harmless representation", "Run timestamp and backend name identify output directories only."
    if path == "run_name":
        return "harmless representation", "Run name labels the TensorBoard/log directory only."
    if path == "events.base_com":
        return "harmless representation", "PhysX dumps a disabled event as null while Newton omits it."
    if path == "events.add_base_mass":
        return "uncontrolled confound", "PhysX randomizes base mass at startup with add range [-1, 3], while Newton has no matching event."
    if path == "events.physics_material":
        return "uncontrolled confound", "PhysX randomizes robot rigid-body material at startup, while Newton has no matching event."
    return "unknown", "Difference is outside the explicit allowlist and needs manual review."


def classified_diffs(kind: str, physx_cfg: Any, newton_cfg: Any) -> list[Difference]:
    diffs = []
    for path, physx, newton in recursive_diff(physx_cfg, newton_cfg):
        classification, rationale = classify(path, physx, newton)
        diffs.append(Difference(f"{kind}.{path}", physx, newton, classification, rationale))
    return diffs


def section_status(env: Any, agent: Any, manifest_rows: list[dict[str, str]], result_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    p_env, n_env = env["physx"], env["newton"]
    p_agent, n_agent = agent["physx"], agent["newton"]

    def same(paths: list[str], root_p: Any = p_env, root_n: Any = n_env) -> bool:
        return all(get_path(root_p, path, "<missing>") == get_path(root_n, path, "<missing>") for path in paths)

    sim_dt = get_path(p_env, "sim.dt")
    decimation = get_path(p_env, "decimation")
    control_hz = 1.0 / (float(sim_dt) * int(decimation)) if sim_dt and decimation else None

    seeds_by_backend: dict[str, set[int]] = {b: set() for b in BACKENDS}
    exits_by_backend: dict[str, set[str]] = {b: set() for b in BACKENDS}
    env_counts = {get_path(p_env, "scene.num_envs"), get_path(n_env, "scene.num_envs")}
    for row in manifest_rows:
        if row.get("backend") in seeds_by_backend:
            seeds_by_backend[row["backend"]].add(int(row["seed"]))
            exits_by_backend[row["backend"]].add(row.get("exit_code", ""))

    iters_by_run: dict[tuple[str, int], set[int]] = {}
    for row in result_rows:
        key = (row["backend"], int(row["seed"]))
        iters_by_run.setdefault(key, set()).add(int(row["iteration"]))

    def complete_iters(backend: str) -> bool:
        for seed in EXPECTED_SEEDS:
            values = iters_by_run.get((backend, seed), set())
            if values != set(range(EXPECTED_ITERS)):
                return False
        return True

    physics_material = get_path(p_env, "events.physics_material") == get_path(n_env, "events.physics_material")
    mass_randomization = get_path(p_env, "events.add_base_mass") == get_path(n_env, "events.add_base_mass")

    rows = [
        {
            "field": "task",
            "physx": "Isaac-Velocity-Flat-Unitree-Go2-v0",
            "newton": "Isaac-Velocity-Flat-Unitree-Go2-v0",
            "classification": "harmless representation",
            "verdict": "Same task is specified by run_strengthen.sh and probe artifacts; task id is not embedded in the committed env YAML.",
        },
        {
            "field": "observations",
            "physx": summarize_value(get_path(p_env, "observations")),
            "newton": summarize_value(get_path(n_env, "observations")),
            "classification": "harmless representation" if same(["observations"]) else "uncontrolled confound",
            "verdict": "Policy observation terms, noise, and concatenation match." if same(["observations"]) else "Observation configs differ.",
        },
        {
            "field": "actions",
            "physx": summarize_value(get_path(p_env, "actions")),
            "newton": summarize_value(get_path(n_env, "actions")),
            "classification": "harmless representation" if same(["actions"]) else "uncontrolled confound",
            "verdict": "Joint position action config matches." if same(["actions"]) else "Action configs differ.",
        },
        {
            "field": "control frequency",
            "physx": f"{control_hz:.1f} Hz" if control_hz else None,
            "newton": f"{control_hz:.1f} Hz" if same(["sim.dt", "decimation"]) and control_hz else None,
            "classification": "harmless representation" if same(["sim.dt", "decimation"]) else "uncontrolled confound",
            "verdict": "Same policy rate from dt=0.005 and decimation=4." if same(["sim.dt", "decimation"]) else "Control rate differs.",
        },
        {
            "field": "timestep",
            "physx": get_path(p_env, "sim.dt"),
            "newton": get_path(n_env, "sim.dt"),
            "classification": "harmless representation" if same(["sim.dt"]) else "uncontrolled confound",
            "verdict": "Same simulation timestep.",
        },
        {
            "field": "decimation",
            "physx": get_path(p_env, "decimation"),
            "newton": get_path(n_env, "decimation"),
            "classification": "harmless representation" if same(["decimation"]) else "uncontrolled confound",
            "verdict": "Same action decimation.",
        },
        {
            "field": "solver/contact",
            "physx": summarize_value(get_path(p_env, "sim.physics")),
            "newton": summarize_value(get_path(n_env, "sim.physics")),
            "classification": "required backend-specific",
            "verdict": "Physics manager and solver/contact buffer settings intentionally differ by backend.",
        },
        {
            "field": "actuator",
            "physx": summarize_value(get_path(p_env, "scene.robot.actuators")),
            "newton": summarize_value(get_path(n_env, "scene.robot.actuators")),
            "classification": "harmless representation" if same(["scene.robot.actuators"]) else "uncontrolled confound",
            "verdict": "DCMotor stiffness, damping, limits, friction, and saturation match." if same(["scene.robot.actuators"]) else "Actuator configs differ.",
        },
        {
            "field": "rewards",
            "physx": summarize_value(get_path(p_env, "rewards")),
            "newton": summarize_value(get_path(n_env, "rewards")),
            "classification": "harmless representation" if same(["rewards"]) else "uncontrolled confound",
            "verdict": "Reward terms, weights, and params match." if same(["rewards"]) else "Reward configs differ.",
        },
        {
            "field": "terminations",
            "physx": summarize_value(get_path(p_env, "terminations")),
            "newton": summarize_value(get_path(n_env, "terminations")),
            "classification": "harmless representation" if same(["terminations"]) else "uncontrolled confound",
            "verdict": "Timeout and base-contact termination configs match." if same(["terminations"]) else "Termination configs differ.",
        },
        {
            "field": "reset distributions",
            "physx": "reset_base/reset_robot_joints match; physics_material and add_base_mass startup events present",
            "newton": "reset_base/reset_robot_joints match; physics_material and add_base_mass startup events missing",
            "classification": "uncontrolled confound" if not (physics_material and mass_randomization) else "harmless representation",
            "verdict": "Startup material and mass randomization are not controlled across backends.",
        },
        {
            "field": "policy architecture",
            "physx": summarize_value({k: p_agent[k] for k in ("actor", "critic")}),
            "newton": summarize_value({k: n_agent[k] for k in ("actor", "critic")}),
            "classification": "harmless representation" if same(["actor", "critic"], p_agent, n_agent) else "uncontrolled confound",
            "verdict": "Actor/critic MLP shapes, activations, and distribution config match." if same(["actor", "critic"], p_agent, n_agent) else "Policy architecture differs.",
        },
        {
            "field": "PPO hyperparams",
            "physx": summarize_value(get_path(p_agent, "algorithm")),
            "newton": summarize_value(get_path(n_agent, "algorithm")),
            "classification": "harmless representation" if same(["algorithm", "num_steps_per_env"], p_agent, n_agent) else "uncontrolled confound",
            "verdict": "PPO algorithm config and rollout length match." if same(["algorithm", "num_steps_per_env"], p_agent, n_agent) else "PPO config differs.",
        },
        {
            "field": "normalization",
            "physx": summarize_value({k: p_agent[k] for k in ("empirical_normalization", "obs_groups")}),
            "newton": summarize_value({k: n_agent[k] for k in ("empirical_normalization", "obs_groups")}),
            "classification": "harmless representation" if same(["empirical_normalization", "obs_groups"], p_agent, n_agent) else "uncontrolled confound",
            "verdict": "No empirical normalization and no agent obs groups in either arm.",
        },
        {
            "field": "seeds",
            "physx": sorted(seeds_by_backend["physx"]),
            "newton": sorted(seeds_by_backend["newton"]),
            "classification": "harmless representation" if all(seeds_by_backend[b] == EXPECTED_SEEDS and exits_by_backend[b] == {"0"} for b in BACKENDS) else "uncontrolled confound",
            "verdict": "Manifest has seeds 0-9 and exit code 0 for both backends." if all(seeds_by_backend[b] == EXPECTED_SEEDS and exits_by_backend[b] == {"0"} for b in BACKENDS) else "Seed coverage or exits are incomplete.",
        },
        {
            "field": "env count",
            "physx": get_path(p_env, "scene.num_envs"),
            "newton": get_path(n_env, "scene.num_envs"),
            "classification": "harmless representation" if env_counts == {2048} else "uncontrolled confound",
            "verdict": "Training resolved env count is 2048 for both arms.",
        },
        {
            "field": "training horizon",
            "physx": {"max_iterations": p_agent["max_iterations"], "csv_complete": complete_iters("physx")},
            "newton": {"max_iterations": n_agent["max_iterations"], "csv_complete": complete_iters("newton")},
            "classification": "harmless representation" if p_agent["max_iterations"] == n_agent["max_iterations"] == EXPECTED_ITERS and all(complete_iters(b) for b in BACKENDS) else "uncontrolled confound",
            "verdict": "Both arms have 300 configured iterations and 300 CSV rows per seed.",
        },
        {
            "field": "eval protocol",
            "physx": "No separate evaluation rollouts in committed artifacts",
            "newton": "No separate evaluation rollouts in committed artifacts",
            "classification": "harmless representation",
            "verdict": "Claims must be limited to training reward, throughput, and probes; there is no held-out policy evaluation protocol.",
        },
        {
            "field": "checkpoint selection",
            "physx": {"save_interval": p_agent["save_interval"], "resume": p_agent["resume"], "load_checkpoint": p_agent["load_checkpoint"]},
            "newton": {"save_interval": n_agent["save_interval"], "resume": n_agent["resume"], "load_checkpoint": n_agent["load_checkpoint"]},
            "classification": "harmless representation" if same(["save_interval", "resume", "load_checkpoint"], p_agent, n_agent) else "uncontrolled confound",
            "verdict": "Training starts fresh; load_checkpoint is inert because resume=false. Final CSV iteration is used for reported training reward.",
        },
        {
            "field": "software versions",
            "physx": "Shared env_capture.txt",
            "newton": "Shared env_capture.txt",
            "classification": "harmless representation",
            "verdict": "Both arms are from one strengthening suite under the same captured Isaac Lab SHA, package freeze, driver, and CUDA stack.",
        },
    ]
    return rows


def read_csv(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Semantic configuration comparison",
        "",
        f"Publication safe: **{summary['publication_safe']}**",
        "",
        "## Recursive differences",
        "",
        "| path | classification | PhysX | Newton | rationale |",
        "|---|---|---|---|---|",
    ]
    for diff in summary["differences"]:
        lines.append(
            f"| `{diff['path']}` | {diff['classification']} | "
            f"`{summarize_value(diff['physx'])}` | `{summarize_value(diff['newton'])}` | {diff['rationale']} |"
        )
    lines.extend([
        "",
        "## Field comparison",
        "",
        "| field | classification | verdict |",
        "|---|---|---|",
    ])
    for row in summary["fields"]:
        lines.append(f"| {row['field']} | {row['classification']} | {row['verdict']} |")
    lines.extend([
        "",
        "## Required rerun",
        "",
        "The committed training artifacts are not publication safe because the Newton arm is missing startup mass and robot material randomization events present in the PhysX arm. Existing results cannot be repaired post hoc.",
        "",
        "Exact rerun commands after fixing the task preset so both resolved configs contain identical non-backend events:",
        "",
        "```bash",
        "export ISAACLAB_PATH=/path/to/IsaacLab",
        "export ISAACLAB_PYTHON=/path/to/IsaacLab/_isaac_sim/python.sh  # or the Isaac Lab venv python used for the original suite",
        "./scripts/run_strengthen.sh",
        "python3 scripts/compare_semantic_configs.py --fail-on-unsafe",
        "python3 scripts/analyze_strengthen.py",
        "```",
    ])
    return "\n".join(lines) + "\n"


def check_run_script_task() -> str | None:
    path = os.path.join(BENCH, "scripts", "run_strengthen.sh")
    if not os.path.exists(path):
        return None
    text = open(path).read()
    match = re.search(r"^TASK=(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def build_summary(params_dir: str, out_dir: str) -> dict[str, Any]:
    env = {
        b: normalize(load_yaml(os.path.join(params_dir, f"env_{b}.yaml")))
        for b in BACKENDS
    }
    agent = {
        b: normalize(load_yaml(os.path.join(params_dir, f"agent_{b}.yaml")))
        for b in BACKENDS
    }
    differences = classified_diffs("env", env["physx"], env["newton"])
    differences.extend(classified_diffs("agent", agent["physx"], agent["newton"]))

    manifest_rows = read_csv(os.path.join(out_dir, "manifest10.csv"))
    result_rows = read_csv(os.path.join(out_dir, "results10.csv"))
    fields = section_status(env, agent, manifest_rows, result_rows)

    counts: dict[str, int] = {}
    for item in differences:
        counts[item.classification] = counts.get(item.classification, 0) + 1
    unsafe_fields = [row for row in fields if row["classification"] in {"uncontrolled confound", "unknown"}]
    publication_safe = not unsafe_fields and counts.get("uncontrolled confound", 0) == 0 and counts.get("unknown", 0) == 0

    return {
        "publication_safe": publication_safe,
        "task_from_run_script": check_run_script_task(),
        "classification_counts": counts,
        "unsafe_fields": unsafe_fields,
        "differences": [item.__dict__ for item in differences],
        "fields": fields,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params-dir", default=DEFAULT_PARAMS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--fail-on-unsafe", action="store_true")
    args = parser.parse_args()

    summary = build_summary(args.params_dir, args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "config_semantic_comparison.json")
    md_path = os.path.join(args.out_dir, "config_semantic_comparison.md")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(md_path, "w") as f:
        f.write(render_markdown(summary))

    print(f"[CONFIG] wrote {json_path}")
    print(f"[CONFIG] wrote {md_path}")
    print(f"[CONFIG] publication_safe={summary['publication_safe']} counts={summary['classification_counts']}")
    if args.fail_on_unsafe and not summary["publication_safe"]:
        print("[CONFIG] unsafe semantic config comparison; rerun after resolving uncontrolled confounds")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
