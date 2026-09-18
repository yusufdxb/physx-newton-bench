#!/usr/bin/env python3
"""Fail closed when two backend evaluations expose different policy tensors."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_contract = left["runtime_contract"]
    right_contract = right["runtime_contract"]

    def action_semantics(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "dimension": term["dimension"],
                "joint_names": term["joint_names"],
            }
            for name, term in contract["actions"].items()
        }

    checks = {
        "robot_joint_order": (
            left_contract["robot_joint_names"] == right_contract["robot_joint_names"]
        ),
        "robot_body_order": (
            left_contract["robot_body_names"] == right_contract["robot_body_names"]
        ),
        "action_contract": action_semantics(left_contract) == action_semantics(right_contract),
        "observation_schema": (
            left_contract["observations"] == right_contract["observations"]
        ),
    }
    def observation_joint_semantics(contract: dict[str, Any]) -> dict[str, list[str]]:
        semantics = {}
        for group in contract["observations"].values():
            for term in group["terms"]:
                if "joint_" not in term["name"]:
                    continue
                semantics[term["name"]] = (
                    term["joint_names"]
                    if term["joint_names"] is not None
                    else contract["robot_joint_names"]
                )
        return semantics

    left_obs_semantics = observation_joint_semantics(left_contract)
    right_obs_semantics = observation_joint_semantics(right_contract)
    policy_joint_observation_safe = left_obs_semantics == right_obs_semantics
    publication_safe = (
        checks["action_contract"]
        and policy_joint_observation_safe
    )
    return {
        "schema_version": 1,
        "publication_safe": publication_safe,
        "checks": checks,
        "policy_joint_observation_safe": policy_joint_observation_safe,
        "left_observation_joint_semantics": left_obs_semantics,
        "right_observation_joint_semantics": right_obs_semantics,
        "left_backend": left["backend"],
        "right_backend": right["backend"],
        "left_robot_joint_names": left_contract["robot_joint_names"],
        "right_robot_joint_names": right_contract["robot_joint_names"],
        "interpretation": (
            "The runtime policy interface matches across backends."
            if publication_safe
            else (
                "The cross-backend fixed-checkpoint result is invalid: action and/or "
                "joint-observation semantics differ at runtime."
            )
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Runtime policy-contract comparison",
        "",
        f"- Publication safe: **{summary['publication_safe']}**",
        f"- Left backend: {summary['left_backend']}",
        f"- Right backend: {summary['right_backend']}",
        "",
        "| check | match |",
        "|---|---:|",
    ]
    for name, passes in summary["checks"].items():
        lines.append(f"| {name} | {passes} |")
    lines.extend(
        [
            "",
            "## Left joint order",
            "",
            "`" + ", ".join(summary["left_robot_joint_names"]) + "`",
            "",
            "## Right joint order",
            "",
            "`" + ", ".join(summary["right_robot_joint_names"]) + "`",
            "",
            summary["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--fail-on-unsafe", action="store_true")
    args = parser.parse_args()
    with open(args.left) as stream:
        left = json.load(stream)
    with open(args.right) as stream:
        right = json.load(stream)
    summary = compare(left, right)
    for path in (args.output_json, args.output_md):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(args.output_json, "w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with open(args.output_md, "w") as stream:
        stream.write(render_markdown(summary))
    print(
        f"[RUNTIME-CONTRACT] publication_safe={summary['publication_safe']} "
        f"action_match={summary['checks']['action_contract']} "
        f"joint_order_match={summary['checks']['robot_joint_order']}"
    )
    return 2 if args.fail_on_unsafe and not summary["publication_safe"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
