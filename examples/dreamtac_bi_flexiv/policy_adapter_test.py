from typing import Any

import numpy as np
import pytest

from examples.dreamtac_bi_flexiv.observation import ACTION_DIM
from examples.dreamtac_bi_flexiv.observation import ACTION_HORIZON
from examples.dreamtac_bi_flexiv.observation import CAMERA_KEYS
from examples.dreamtac_bi_flexiv.observation import RGB_CAMERA_KEYS
from examples.dreamtac_bi_flexiv.observation import STATE_DIM
from examples.dreamtac_bi_flexiv.observation import TACTILE_IMAGE_SHAPE
from examples.dreamtac_bi_flexiv.observation import TACTILE_KEYS
from examples.dreamtac_bi_flexiv.policy_adapter import DreamTacRemotePolicy
from examples.dreamtac_bi_flexiv.policy_adapter import build_policy_payload
from examples.dreamtac_bi_flexiv.policy_adapter import validate_server_metadata


def _metadata() -> dict[str, Any]:
    return {
        "service": "dreamtac-bi_flexiv",
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "action_space": "absolute_tcp18_absolute_gripper2",
        "normalization_mode": "q99",
        "camera_keys": CAMERA_KEYS,
        "image_shape": (224, 224, 3),
        "future_image_horizon": ACTION_HORIZON,
        "state_t": 11,
        "num_conditional_frames": 7,
        "rtc_supported": False,
    }


def _observation() -> dict[str, Any]:
    images = {name: np.zeros((224, 224, 3), dtype=np.uint8) for name in RGB_CAMERA_KEYS}
    images.update({name: np.zeros(TACTILE_IMAGE_SHAPE, dtype=np.uint8) for name in TACTILE_KEYS})
    return {
        "observation_seq": 3,
        "state": np.zeros((STATE_DIM,), dtype=np.float32),
        "images": images,
        "images_raw": {"debug": np.zeros((2, 2, 3), dtype=np.uint8)},
        "tactile_self_attn_gate": np.asarray([0.15, 0.8], dtype=np.float32),
    }


def test_accepts_current_dreamtac_server_metadata() -> None:
    validate_server_metadata(_metadata())


def test_rejects_old_20_step_server() -> None:
    metadata = _metadata()
    metadata["action_horizon"] = 20

    with pytest.raises(ValueError, match="action_horizon=20"):
        validate_server_metadata(metadata)


def test_payload_keeps_only_model_inputs_and_adds_prompt() -> None:
    payload = build_policy_payload(_observation(), default_prompt="task prompt")

    assert set(payload) == {"state", "images", "tactile_self_attn_gate", "observation_seq", "prompt"}
    assert tuple(payload["images"]) == CAMERA_KEYS
    assert payload["images"]["left_tactile_left"].shape == TACTILE_IMAGE_SHAPE
    assert payload["prompt"] == "task prompt"


class _FakePolicy:
    def infer(self, _obs: dict) -> dict[str, Any]:
        return {
            "actions": np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32),
            "action_space": "absolute_tcp18_absolute_gripper2",
        }

    def reset(self) -> None:
        pass


def test_remote_policy_accepts_a_30_step_action_chunk() -> None:
    policy = DreamTacRemotePolicy(_FakePolicy(), default_prompt="task prompt")
    result = policy.infer(_observation())

    assert result["actions"].shape == (30, 20)
