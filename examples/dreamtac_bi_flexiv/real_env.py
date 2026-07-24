"""Seven-camera BiFlexiv hardware wrapper for Dream-Tac inference."""

from __future__ import annotations

from lerobot.utils.robot_utils import get_logger

from examples.bi_flexiv_rizon4_rt.real_env import BiFlexivRizon4RTRealEnv
from examples.dreamtac_bi_flexiv.observation import CAMERA_KEYS

logger = get_logger("DreamTacBiFlexivRealEnv")


class DreamTacBiFlexivRealEnv(BiFlexivRizon4RTRealEnv):
    """Reuse BiFlexiv state/action handling while retaining all seven model views."""

    def __init__(self, *args, enable_tactile_sensors: bool = True, **kwargs) -> None:
        if not enable_tactile_sensors:
            raise ValueError("Dream-Tac requires enable_tactile_sensors=True")
        super().__init__(*args, enable_tactile_sensors=True, **kwargs)

    def get_images(self, obs: dict) -> dict:
        missing = [name for name in CAMERA_KEYS if name not in obs]
        if missing:
            available = sorted(key for key, value in obs.items() if hasattr(value, "shape"))
            raise RuntimeError(f"Missing Dream-Tac camera frames: {missing}. Available array fields: {available}")
        return {name: obs[name] for name in CAMERA_KEYS}
