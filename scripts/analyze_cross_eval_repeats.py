#!/usr/bin/env python3
"""Compare cross-backend rank instability with evaluation-seed instability."""
from __future__ import annotations

import argparse
import json
import os

from analyze_cross_eval import (
    load_results,
    preference_reversals,
    rank_map,
    spearman_from_ranks,
)


def _scores(results, policies, backend):
    return {policy: float(results[(policy, backend)]["mean_return"]) for policy in policies}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-dir", required=True)
    parser.add_argument("--second-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    first = load_results(args.first_dir)
    second = load_results(args.second_dir)
    policies = sorted({policy for policy, _ in first})
    if policies != sorted({policy for policy, _ in second}):
        raise ValueError("evaluation repeats must contain the same policies")

    same_backend = {}
    for backend in ("physx", "newton"):
        first_scores = _scores(first, policies, backend)
        second_scores = _scores(second, policies, backend)
        reversals = preference_reversals(first_scores, second_scores)
        same_backend[backend] = {
            "spearman": spearman_from_ranks(rank_map(first_scores), rank_map(second_scores)),
            "preference_reversals": len(reversals),
            "mean_success_first": sum(first[(p, backend)]["success_rate"] for p in policies) / len(policies),
            "mean_success_second": sum(second[(p, backend)]["success_rate"] for p in policies) / len(policies),
        }

    cross_backend = []
    for repeat_name, results in (("first", first), ("second", second)):
        physx_scores = _scores(results, policies, "physx")
        newton_scores = _scores(results, policies, "newton")
        reversals = preference_reversals(physx_scores, newton_scores)
        cross_backend.append(
            {
                "repeat": repeat_name,
                "spearman": spearman_from_ranks(rank_map(physx_scores), rank_map(newton_scores)),
                "preference_reversals": len(reversals),
            }
        )

    total_pairs = len(policies) * (len(policies) - 1) // 2
    summary = {
        "schema_version": 1,
        "num_policies": len(policies),
        "total_policy_pairs": total_pairs,
        "same_backend": same_backend,
        "cross_backend": cross_backend,
        "interpretation": (
            "The deployability collapse repeats, but return ranking is also evaluation-seed sensitive. "
            "Do not attribute every pairwise reversal to the backend."
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    lines = [
        "# Cross-backend seed-repeatability control",
        "",
        f"- Policies: {len(policies)}",
        f"- Policy pairs: {total_pairs}",
        "",
        "| comparison | Spearman | preference reversals | mean success first / second |",
        "|---|---:|---:|---:|",
        (
            f"| PhysX seed repeat | {same_backend['physx']['spearman']:.4f} | "
            f"{same_backend['physx']['preference_reversals']} / {total_pairs} | "
            f"{same_backend['physx']['mean_success_first']:.4f} / "
            f"{same_backend['physx']['mean_success_second']:.4f} |"
        ),
        (
            f"| Newton seed repeat | {same_backend['newton']['spearman']:.4f} | "
            f"{same_backend['newton']['preference_reversals']} / {total_pairs} | "
            f"{same_backend['newton']['mean_success_first']:.4f} / "
            f"{same_backend['newton']['mean_success_second']:.4f} |"
        ),
    ]
    for item in cross_backend:
        lines.append(
            f"| Cross-backend {item['repeat']} | {item['spearman']:.4f} | "
            f"{item['preference_reversals']} / {total_pairs} | n/a |"
        )
    lines.extend(["", summary["interpretation"], ""])
    with open(args.output_md, "w") as stream:
        stream.write("\n".join(lines))
    print(
        "[CROSS-EVAL] repeatability "
        f"physx_rho={same_backend['physx']['spearman']:.4f} "
        f"newton_rho={same_backend['newton']['spearman']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
