from __future__ import annotations

from copy import deepcopy

from compare_runtime_contracts import compare


def _result(backend: str) -> dict:
    return {
        "backend": backend,
        "runtime_contract": {
            "robot_joint_names": ["left", "right"],
            "robot_body_names": ["base", "foot"],
            "actions": {
                "joint_pos": {
                    "dimension": 2,
                    "joint_names": ["left", "right"],
                }
            },
            "observations": {
                "policy": {
                    "terms": [
                        {"name": "joint_pos", "joint_names": None},
                        {"name": "joint_vel", "joint_names": None},
                    ]
                }
            },
        },
    }


def test_equal_contract_is_publication_safe():
    assert compare(_result("physx"), _result("newton"))["publication_safe"]


def test_joint_reordering_fails_closed_even_if_observation_schema_matches():
    physx = _result("physx")
    newton = deepcopy(_result("newton"))
    newton["runtime_contract"]["robot_joint_names"] = ["right", "left"]
    newton["runtime_contract"]["actions"]["joint_pos"]["joint_names"] = ["right", "left"]
    summary = compare(physx, newton)
    assert not summary["publication_safe"]
    assert not summary["checks"]["robot_joint_order"]
    assert summary["checks"]["observation_schema"]


def test_explicit_joint_names_make_different_asset_orders_safe():
    physx = _result("physx")
    newton = deepcopy(_result("newton"))
    newton["runtime_contract"]["robot_joint_names"] = ["right", "left"]
    for result in (physx, newton):
        for term in result["runtime_contract"]["observations"]["policy"]["terms"]:
            term["joint_names"] = ["left", "right"]
    assert compare(physx, newton)["publication_safe"]
