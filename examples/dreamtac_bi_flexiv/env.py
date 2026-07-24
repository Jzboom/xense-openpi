"""xense-client Environment for Dream-Tac on the dual Flexiv robot."""

from __future__ import annotations

from typing import override

from lerobot.utils.robot_utils import get_logger
import numpy as np
from xense_client.runtime import environment as _environment

from examples.dreamtac_bi_flexiv.observation import ACTION_DIM
from examples.dreamtac_bi_flexiv.observation import DEFAULT_IMAGE_SIZE
from examples.dreamtac_bi_flexiv.observation import TactileGateTracker
from examples.dreamtac_bi_flexiv.observation import coerce_hwc_uint8
from examples.dreamtac_bi_flexiv.observation import prepare_policy_images
from examples.dreamtac_bi_flexiv.real_env import DreamTacBiFlexivRealEnv

logger = get_logger("DreamTacBiFlexivEnv")

_ACTION_LABELS = (
    "L.x",
    "L.y",
    "L.z",
    "L.r1",
    "L.r2",
    "L.r3",
    "L.r4",
    "L.r5",
    "L.r6",
    "R.x",
    "R.y",
    "R.z",
    "R.r1",
    "R.r2",
    "R.r3",
    "R.r4",
    "R.r5",
    "R.r6",
    "L.grip",
    "R.grip",
)


class DreamTacBiFlexivEnvironment(_environment.Environment):
    """Build Dream-Tac requests while keeping action execution in LeRobot-Xense."""

    def __init__(
        self,
        *,
        bi_mount_type: str = "side",
        use_force: bool = False,
        go_to_start: bool = True,
        stiffness_ratio: float = 0.2,
        inner_control_hz: int = 1000,
        interpolate_cmds: bool = True,
        log_level: str = "INFO",
        image_size: int = DEFAULT_IMAGE_SIZE,
        include_raw_images: bool = False,
        setup_robot: bool = True,
    ) -> None:
        self._env = DreamTacBiFlexivRealEnv(
            bi_mount_type=bi_mount_type,
            use_force=use_force,
            go_to_start=go_to_start,
            stiffness_ratio=stiffness_ratio,
            inner_control_hz=inner_control_hz,
            interpolate_cmds=interpolate_cmds,
            enable_tactile_sensors=True,
            log_level=log_level,
            setup_robot=setup_robot,
        )
        self._image_size = image_size
        self._include_raw_images = include_raw_images
        self._gate_tracker = TactileGateTracker()
        self._observation_seq = 0
        self._action_step = 0

    @override
    def reset(self) -> None:
        self._env.reset()
        self._gate_tracker.reset()
        self._observation_seq = 0
        self._action_step = 0

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        raw_obs = self._env.get_observation()
        raw_images = raw_obs["images"]
        tactile_gate = self._gate_tracker.update(raw_images)
        processed_images = prepare_policy_images(raw_images, image_size=self._image_size)

        self._observation_seq += 1
        observation = {
            "observation_seq": self._observation_seq,
            "state": np.ascontiguousarray(raw_obs["qpos"], dtype=np.float32),
            "images": processed_images,
            "tactile_self_attn_gate": tactile_gate,
        }
        if self._include_raw_images:
            observation["images_raw"] = {name: coerce_hwc_uint8(image, name=name) for name, image in raw_images.items()}
        return observation

    @override
    def apply_action(self, action: dict) -> None:
        actions = np.asarray(action.get("actions"), dtype=np.float32)
        if actions.shape != (ACTION_DIM,):
            raise ValueError(f"Robot action must have shape ({ACTION_DIM},), got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Robot action contains NaN or Inf")

        self._action_step += 1
        action_text = " | ".join(f"{label}={value:+.4f}" for label, value in zip(_ACTION_LABELS, actions))
        logger.debug(f"Step {self._action_step}: {action_text}")
        self._env.send_action(actions)

    def disconnect(self) -> None:
        self._env.disconnect()
