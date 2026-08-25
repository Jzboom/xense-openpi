"""Configuration-contract tests that do not connect to robot hardware."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from examples.dreamtac_bi_flexiv.observation import CAMERA_KEYS
from examples.dreamtac_bi_flexiv.robot_config import validate_dreamtac_robot_config


def _config(
    *,
    use_force: bool = False,
    left_gripper: bool = True,
    right_gripper: bool = True,
    auto_discover_cameras: bool = True,
    enable_tactile: bool = True,
    cameras: tuple[str, ...] = ("head",),
):
    gripper = SimpleNamespace(
        auto_discover_cameras=auto_discover_cameras,
        enable_tactile=enable_tactile,
    )
    return cast(
        Any,
        SimpleNamespace(
            use_force=use_force,
            gripper=gripper,
            left_gripper=gripper if left_gripper else None,
            right_gripper=gripper if right_gripper else None,
            cameras=dict.fromkeys(cameras),
        ),
    )


def test_accepts_current_auto_discovery_recipe() -> None:
    validate_dreamtac_robot_config(_config())


def test_rejects_force_action_space() -> None:
    with pytest.raises(ValueError, match="use_force"):
        validate_dreamtac_robot_config(_config(use_force=True))


def test_rejects_a_missing_gripper() -> None:
    with pytest.raises(ValueError, match="missing: right"):
        validate_dreamtac_robot_config(_config(right_gripper=False))


def test_rejects_auto_discovery_without_tactile() -> None:
    with pytest.raises(ValueError, match="enable_tactile"):
        validate_dreamtac_robot_config(_config(enable_tactile=False))


def test_rejects_auto_discovery_without_head_camera() -> None:
    with pytest.raises(ValueError, match="pin head"):
        validate_dreamtac_robot_config(_config(cameras=()))


def test_manual_camera_recipe_must_pin_all_seven_views() -> None:
    with pytest.raises(ValueError, match="left_wrist"):
        validate_dreamtac_robot_config(_config(auto_discover_cameras=False))
    validate_dreamtac_robot_config(_config(auto_discover_cameras=False, cameras=CAMERA_KEYS))
