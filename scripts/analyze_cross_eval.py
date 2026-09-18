#!/usr/bin/env python3
"""Analyze whether fixed checkpoints retain their ranking across backends."""
from __future__ import annotations

import argparse
import glob
import json
import os
from itertools import combinations
from typing import Any

import numpy as np


def rank_map(scores: dict[str, float]) -> dict[str, int]:
    """Return descending ordinal ranks, breaking exact ties by policy id."""
    ordered = sorted(scores, key=lambda policy: (-scores[policy], policy))
    return {policy: rank for rank, policy in enumerate(ordered, start=1)}


def spearman_from_ranks(left: dict[str, int], right: dict[str, int]) -> float:
    """Compute Spearman correlation for two complete rank maps."""
    policies = sorted(left)
    if policies != sorted(right):
        raise ValueError("rank maps must contain the same policies")
    if len(policies) < 2:
        return float("nan")
    x = np.asarray([left[p] for p in policies], dtype=float)
    y = np.asarray([right[p] for p in policies], dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def preference_reversals(
    physx_scores: dict[str, float], newton_scores: dict[str, float]
) -> list[dict[str, Any]]:
    """List policy pairs whose strict mean-return preference reverses."""
    reversals = []
    for first, second in combinations(sorted(physx_scores), 2):
        physx_delta = physx_scores[first] - physx_scores[second]
        newton_delta = newton_scores[first] - newton_scores[second]
        if physx_delta * newton_delta < 0:
            reversals.append(
                {
                    "first": first,
                    "second": second,
                    "physx_delta": physx_delta,
                    "newton_delta": newton_delta,
                }
            )
    return reversals


def bootstrap_mean_difference(
    physx: list[float],
    newton: list[float],
    *,
    seed: int = 20260726,
    n_resamples: int = 5000,
) -> dict[str, float]:
    """Return an unpaired bootstrap CI for Newton minus PhysX mean return."""
    left = np.asarray(physx, dtype=float)
    right = np.asarray(newton, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        samples[index] = (
            rng.choice(right, size=len(right), replace=True).mean()
            - rng.choice(left, size=len(left), replace=True).mean()
        )
    return {
        "difference": float(right.mean() - left.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def load_results(results_dir: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Load one artifact for every policy/backend cell."""
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as stream:
            item = json.load(stream)
        if item.get("schema_version") != 1 or "policy_id" not in item:
            continue
        key = (item["policy_id"], item["backend"])
        if key in results:
            raise ValueError(f"duplicate result cell {key}")
        results[key] = item
    return results


def analyze(results: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """Build the rank-stability result payload."""
    policies = sorted({policy for policy, _ in results})
    missing = [
        (policy, backend)
        for policy in policies
        for backend in ("physx", "newton")
        if (policy, backend) not in results
    ]
    if missing:
        raise ValueError(f"missing result cells: {missing}")
    if len(policies) < 2:
        raise ValueError("at least two policies are required for rank analysis")

    scores = {
        backend: {policy: float(results[(policy, backend)]["mean_return"]) for policy in policies}
        for backend in ("physx", "newton")
    }
    ranks = {backend: rank_map(scores[backend]) for backend in ("physx", "newton")}
    reversals = preference_reversals(scores["physx"], scores["newton"])
    total_pairs = len(policies) * (len(policies) - 1) // 2
    per_policy = []
    for index, policy in enumerate(policies):
        physx = results[(policy, "physx")]
        newton = results[(policy, "newton")]
        effect = bootstrap_mean_difference(
            physx["episode_return"],
            newton["episode_return"],
            seed=20260726 + index,
        )
        per_policy.append(
            {
                "policy_id": policy,
                "physx_mean_return": scores["physx"][policy],
                "newton_mean_return": scores["newton"][policy],
                "physx_success_rate": float(physx["success_rate"]),
                "newton_success_rate": float(newton["success_rate"]),
                "newton_minus_physx_return": effect,
                "physx_rank": ranks["physx"][policy],
                "newton_rank": ranks["newton"][policy],
            }
        )

    max_rank_shift = max(abs(ranks["physx"][p] - ranks["newton"][p]) for p in policies)
    return {
        "schema_version": 1,
        "num_policies": len(policies),
        "total_policy_pairs": total_pairs,
        "preference_reversal_count": len(reversals),
        "preference_reversal_fraction": len(reversals) / total_pairs,
        "spearman_rank_correlation": spearman_from_ranks(ranks["physx"], ranks["newton"]),
        "max_rank_shift": max_rank_shift,
        "ranks": ranks,
        "reversals": reversals,
        "per_policy": per_policy,
        "probe_survives": bool(reversals),
        "uncertainty_note": (
            "Per-policy return intervals bootstrap vectorized episodes within one simulator process. "
            "They do not replace process-level replication."
        ),
        "interpretation": (
            "At least one fixed-checkpoint preference reverses across backends."
            if reversals
            else "No fixed-checkpoint preference reversal was observed in this bounded probe."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a compact result table."""
    lines = [
        "# Controlled cross-backend checkpoint evaluation",
        "",
        f"- Policies: {summary['num_policies']}",
        f"- Pairwise preference reversals: {summary['preference_reversal_count']} / {summary['total_policy_pairs']}",
        f"- Spearman rank correlation: {summary['spearman_rank_correlation']:.4f}",
        f"- Maximum rank shift: {summary['max_rank_shift']}",
        f"- Probe survives: **{summary['probe_survives']}**",
        f"- Uncertainty: {summary['uncertainty_note']}",
        "",
        "| policy | PhysX return | Newton return | Newton - PhysX [95% CI] | PhysX success | Newton success | ranks P/N |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_policy"]:
        effect = row["newton_minus_physx_return"]
        lines.append(
            f"| {row['policy_id']} | {row['physx_mean_return']:.3f} | "
            f"{row['newton_mean_return']:.3f} | {effect['difference']:.3f} "
            f"[{effect['ci_low']:.3f}, {effect['ci_high']:.3f}] | "
            f"{row['physx_success_rate']:.3f} | {row['newton_success_rate']:.3f} | "
            f"{row['physx_rank']} / {row['newton_rank']} |"
        )
    lines.extend(["", summary["interpretation"], ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    summary = analyze(load_results(args.results_dir))
    for path in (args.output_json, args.output_md):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(args.output_json, "w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with open(args.output_md, "w") as stream:
        stream.write(render_markdown(summary))
    print(
        f"[CROSS-EVAL] reversals={summary['preference_reversal_count']}/"
        f"{summary['total_policy_pairs']} spearman={summary['spearman_rank_correlation']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
