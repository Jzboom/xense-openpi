"""Seven-camera BiFlexiv hardware wrapper for Dream-Tac inference."""

from __future__ import annotations

from lerobot.robots.bi_flexiv_rizon4_rt.config_bi_flexiv_rizon4_rt import BiFlexivRizon4RTConfig
from lerobot.utils.robot_utils import get_logger

from examples.bi_flexiv_rizon4_rt.real_env import BiFlexivRizon4RTRealEnv
from examples.dreamtac_bi_flexiv.observation import CAMERA_KEYS
from examples.dreamtac_bi_flexiv.robot_config import validate_dreamtac_robot_config

logger = get_logger("DreamTacBiFlexivRealEnv")


class DreamTacBiFlexivRealEnv(BiFlexivRizon4RTRealEnv):
    """Reuse BiFlexiv state/action handling while retaining all seven model views."""

    def __init__(self, robot_config: BiFlexivRizon4RTConfig, setup_robot: bool = True) -> None:
        validate_dreamtac_robot_config(robot_config)
        super().__init__(robot_config=robot_config, setup_robot=setup_robot)

    def get_images(self, obs: dict) -> dict:
        missing = [name for name in CAMERA_KEYS if name not in obs]
        if missing:
            available = sorted(key for key, value in obs.items() if hasattr(value, "shape"))
            raise RuntimeError(f"Missing Dream-Tac camera frames: {missing}. Available array fields: {available}")
        return {name: obs[name] for name in CAMERA_KEYS}
