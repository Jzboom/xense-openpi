"""Pure validation of the LeRobot-Xense bench contract Dream-Tac requires."""

from __future__ import annotations

from typing import Any

from examples.dreamtac_bi_flexiv.observation import CAMERA_KEYS


def validate_dreamtac_robot_config(robot_config: Any) -> None:
    """Reject a bench that cannot provide Dream-Tac's fixed 20D/seven-view contract."""
    if robot_config.use_force:
        raise ValueError("Dream-Tac uses a 20D pose/gripper action space; recipe use_force must be false")

    missing_grippers = [
        side
        for side, config in (("left", robot_config.left_gripper), ("right", robot_config.right_gripper))
        if config is None
    ]
    if missing_grippers:
        raise ValueError(f"Dream-Tac requires grippers on both arms; missing: {', '.join(missing_grippers)}")

    gripper = robot_config.gripper
    if gripper is None:  # Per-side validation above already catches this; explicit for unusual config objects.
        raise ValueError("Dream-Tac requires a typed gripper block in the robot recipe")

    if gripper.auto_discover_cameras:
        if not gripper.enable_tactile:
            raise ValueError(
                "Dream-Tac requires four tactile cameras; set robot.gripper.enable_tactile: true in the recipe"
            )
        if "head" not in robot_config.cameras:
            raise ValueError("Dream-Tac camera discovery supplies wrists/tactiles, but the recipe must pin head")
    else:
        missing_cameras = [name for name in CAMERA_KEYS if name not in robot_config.cameras]
        if missing_cameras:
            raise ValueError(
                "Dream-Tac camera auto-discovery is disabled and the recipe does not pin all seven views; "
                f"missing: {missing_cameras}"
            )
