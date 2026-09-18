from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_cross_eval import analyze, preference_reversals, rank_map


def _cell(policy: str, backend: str, score: float) -> dict:
    return {
        "policy_id": policy,
        "backend": backend,
        "mean_return": score,
        "success_rate": float(score > 0),
        "episode_return": [score - 0.1, score + 0.1],
    }


def test_preference_reversal_is_detected() -> None:
    physx = {"a": 2.0, "b": 1.0, "c": 0.0}
    newton = {"a": 1.0, "b": 2.0, "c": 0.0}

    reversals = preference_reversals(physx, newton)

    assert [(item["first"], item["second"]) for item in reversals] == [("a", "b")]


def test_rank_map_is_descending_and_deterministic_on_ties() -> None:
    assert rank_map({"b": 1.0, "a": 1.0, "c": 2.0}) == {"c": 1, "a": 2, "b": 3}


def test_analysis_reports_rank_shift() -> None:
    results = {
        ("a", "physx"): _cell("a", "physx", 2.0),
        ("a", "newton"): _cell("a", "newton", 1.0),
        ("b", "physx"): _cell("b", "physx", 1.0),
        ("b", "newton"): _cell("b", "newton", 2.0),
    }

    summary = analyze(results)

    assert summary["preference_reversal_count"] == 1
    assert summary["preference_reversal_fraction"] == 1.0
    assert summary["max_rank_shift"] == 1
    assert summary["probe_survives"] is True


def test_analysis_refuses_incomplete_backend_cells() -> None:
    with pytest.raises(ValueError, match="missing result cells"):
        analyze(
            {
                ("a", "physx"): _cell("a", "physx", 2.0),
                ("a", "newton"): _cell("a", "newton", 1.0),
                ("b", "physx"): _cell("b", "physx", 1.0),
            }
        )
