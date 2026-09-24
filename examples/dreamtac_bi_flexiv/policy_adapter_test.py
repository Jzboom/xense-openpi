from typing import Any

import numpy as np
import pytest
from xense_client import action_chunk_broker
from xense_client import msgpack_numpy

from examples.dreamtac_bi_flexiv.main import Args
from examples.dreamtac_bi_flexiv.observation import ACTION_DIM
from examples.dreamtac_bi_flexiv.observation import ACTION_HORIZON
from examples.dreamtac_bi_flexiv.observation import FUTURE_IMAGE_OFFSETS
from examples.dreamtac_bi_flexiv.observation import HISTORY_FRAMES
from examples.dreamtac_bi_flexiv.observation import RGB_HISTORY_OFFSETS
from examples.dreamtac_bi_flexiv.observation import STATE_DIM
from examples.dreamtac_bi_flexiv.observation import TACTILE_HISTORY_OFFSETS
from examples.dreamtac_bi_flexiv.observation import TRANSPORT_CAMERA_KEYS
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
        "camera_keys": TRANSPORT_CAMERA_KEYS,
        "image_shape": (HISTORY_FRAMES, 224, 224, 3),
        "camera_history_shape": (HISTORY_FRAMES, 224, 224, 3),
        "future_image_shape": (HISTORY_FRAMES, 224, 224, 3),
        "history_frames": HISTORY_FRAMES,
        "rgb_history_offsets": RGB_HISTORY_OFFSETS,
        "tactile_history_offsets": TACTILE_HISTORY_OFFSETS,
        "future_image_offsets": FUTURE_IMAGE_OFFSETS,
        "future_image_horizon": ACTION_HORIZON,
        "state_t": 11,
        "num_conditional_frames": 7,
        "rtc_supported": False,
    }


def _observation() -> dict[str, Any]:
    images = {
        name: np.zeros((HISTORY_FRAMES, 224, 224, 3), dtype=np.uint8)
        for name in TRANSPORT_CAMERA_KEYS
    }
    return {
        "observation_seq": 3,
        "state": np.zeros((STATE_DIM,), dtype=np.float32),
        "images": images,
        "images_raw": {"debug": np.zeros((2, 2, 3), dtype=np.uint8)},
        "tactile_self_attn_gate": np.asarray([0.15, 0.8], dtype=np.float32),
    }


def test_accepts_current_dreamtac_server_metadata() -> None:
    metadata = _metadata()
    # WebSocket serialization turns tuples into lists.
    for key in (
        "camera_keys",
        "image_shape",
        "camera_history_shape",
        "future_image_shape",
        "rgb_history_offsets",
        "tactile_history_offsets",
        "future_image_offsets",
    ):
        metadata[key] = list(metadata[key])

    validate_server_metadata(metadata)


def test_rejects_old_30_step_server() -> None:
    metadata = _metadata()
    metadata["action_horizon"] = 30

    with pytest.raises(ValueError, match="action_horizon=30"):
        validate_server_metadata(metadata)


def test_payload_keeps_only_model_inputs_and_adds_prompt() -> None:
    payload = build_policy_payload(_observation(), default_prompt="task prompt")

    assert set(payload) == {"state", "images", "tactile_self_attn_gate", "observation_seq", "prompt"}
    assert tuple(payload["images"]) == TRANSPORT_CAMERA_KEYS
    assert payload["images"]["head"].shape == (HISTORY_FRAMES, 224, 224, 3)
    assert payload["images"]["left_tactile_merged"].shape == (HISTORY_FRAMES, 224, 224, 3)
    assert payload["prompt"] == "task prompt"


def test_compact_payload_is_about_three_megabytes() -> None:
    payload = build_policy_payload(_observation(), default_prompt="task prompt")

    packed_size = len(msgpack_numpy.packb(payload))

    assert 3_010_560 < packed_size < 3_020_000


def test_payload_rejects_legacy_single_frames() -> None:
    observation = _observation()
    observation["images"]["head"] = np.zeros((224, 224, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="image history 'head' must have shape"):
        build_policy_payload(observation)


class _FakePolicy:
    def infer(self, _obs: dict) -> dict[str, Any]:
        return {
            "actions": np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32),
            "action_space": "absolute_tcp18_absolute_gripper2",
        }

    def reset(self) -> None:
        pass


def test_remote_policy_accepts_a_40_step_action_chunk() -> None:
    policy = DreamTacRemotePolicy(_FakePolicy(), default_prompt="task prompt")
    result = policy.infer(_observation())

    assert result["actions"].shape == (40, 20)


class _CountingChunkPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, _obs: dict) -> dict[str, np.ndarray]:
        offset = 100 * self.calls
        self.calls += 1
        steps = (np.arange(ACTION_HORIZON, dtype=np.float32) + offset)[:, None]
        return {"actions": np.repeat(steps, ACTION_DIM, axis=1)}

    def reset(self) -> None:
        pass


def test_client_executes_the_full_40_step_chunk_by_default() -> None:
    execution_horizon = Args().action_execution_horizon
    assert execution_horizon == 40

    inner = _CountingChunkPolicy()
    broker = action_chunk_broker.ActionChunkBroker(
        policy=inner,
        action_horizon=execution_horizon,
    )
    executed = [float(broker.infer({})["actions"][0]) for _ in range(41)]

    assert executed[:40] == list(range(40))
    assert executed[40] == 100.0
    assert inner.calls == 2


def test_rtc_mode_requires_supported_server() -> None:
    metadata = _metadata()
    with pytest.raises(ValueError, match="rtc_supported=False"):
        validate_server_metadata(metadata, require_rtc=True)
    metadata["rtc_supported"] = True
    validate_server_metadata(metadata, require_rtc=True)


class _FakeRTCPolicy:
    def __init__(self) -> None:
        self.kwargs = None
        self.cancelled = False
        self.prefix_length = 14

    def infer(self, _obs: dict, **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        return {
            "actions": np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32),
            "action_space": "absolute_tcp18_absolute_gripper2",
            "rtc_prefix_length": self.prefix_length,
        }

    def cancel_pending(self) -> None:
        self.cancelled = True


def test_rtc_policy_forwards_prefix_and_checks_server_response() -> None:
    inner = _FakeRTCPolicy()
    policy = DreamTacRemotePolicy(inner)
    prefix = np.zeros((14, ACTION_DIM), dtype=np.float32)
    response = policy.infer(
        _observation(), prev_chunk_left_over=prefix,
        inference_delay=14, execution_horizon=40,
    )
    assert inner.kwargs["inference_delay"] == 14
    np.testing.assert_array_equal(inner.kwargs["prev_chunk_left_over"], prefix)
    assert response["rtc_prefix_length"] == 14
    policy.cancel_pending()
    assert inner.cancelled

    inner.prefix_length = 13
    with pytest.raises(ValueError, match="RTC prefix length"):
        policy.infer(_observation(), prev_chunk_left_over=prefix, inference_delay=14)
