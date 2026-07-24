"""Robot-side adapter from a full BiFlexiv observation to Dream-Tac RPC."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, override

import numpy as np
from xense_client import base_policy as _base_policy

from examples.dreamtac_bi_flexiv.observation import ACTION_DIM
from examples.dreamtac_bi_flexiv.observation import ACTION_HORIZON
from examples.dreamtac_bi_flexiv.observation import CAMERA_KEYS
from examples.dreamtac_bi_flexiv.observation import DEFAULT_IMAGE_SIZE
from examples.dreamtac_bi_flexiv.observation import STATE_DIM


def validate_server_metadata(metadata: Mapping[str, Any]) -> None:
    """Fail fast when the robot connects to an incompatible inference server."""
    expected = {
        "service": "dreamtac-earbud",
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "action_space": "absolute_tcp18_absolute_gripper2",
        "rtc_supported": False,
    }
    errors = [
        f"{key}={metadata.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]

    camera_keys = tuple(metadata.get("camera_keys", ()))
    if camera_keys != CAMERA_KEYS:
        errors.append(f"camera_keys={camera_keys!r}, expected {CAMERA_KEYS!r}")
    image_shape = tuple(metadata.get("image_shape", ()))
    if image_shape != (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3):
        errors.append(f"image_shape={image_shape!r}, expected {(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3)!r}")
    if errors:
        raise ValueError("Incompatible Dream-Tac server metadata: " + "; ".join(errors))


def build_policy_payload(observation: Mapping[str, Any], *, default_prompt: str | None = None) -> dict[str, Any]:
    """Select only fields required by Dream-Tac; local raw/debug fields stay local."""
    state = np.asarray(observation.get("state"), dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"state must have shape ({STATE_DIM},), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("state contains NaN or Inf")

    images = observation.get("images")
    if not isinstance(images, Mapping):
        raise ValueError("observation must contain an 'images' mapping")
    missing = [name for name in CAMERA_KEYS if name not in images]
    if missing:
        raise ValueError(f"observation is missing Dream-Tac cameras: {missing}")

    gate = np.asarray(observation.get("tactile_self_attn_gate"), dtype=np.float32)
    if gate.shape != (2,):
        raise ValueError(f"tactile_self_attn_gate must have shape (2,), got {gate.shape}")
    if not np.isfinite(gate).all():
        raise ValueError("tactile_self_attn_gate contains NaN or Inf")

    payload: dict[str, Any] = {
        "state": np.ascontiguousarray(state),
        "images": {name: images[name] for name in CAMERA_KEYS},
        "tactile_self_attn_gate": np.ascontiguousarray(gate),
    }
    if "observation_seq" in observation:
        payload["observation_seq"] = observation["observation_seq"]

    prompt = observation.get("prompt") or default_prompt
    if prompt:
        payload["prompt"] = str(prompt)
    return payload


class DreamTacRemotePolicy(_base_policy.BasePolicy):
    """Filter robot observations and validate Dream-Tac action chunks."""

    def __init__(self, policy: _base_policy.BasePolicy, *, default_prompt: str | None = None) -> None:
        self._policy = policy
        self._default_prompt = default_prompt

    @override
    def infer(self, obs: dict) -> dict:
        payload = build_policy_payload(obs, default_prompt=self._default_prompt)
        response = self._policy.infer(payload)
        if not isinstance(response, Mapping):
            raise ValueError(f"Dream-Tac response must be a mapping, got {type(response).__name__}")

        actions = np.asarray(response.get("actions"), dtype=np.float32)
        if actions.shape != (ACTION_HORIZON, ACTION_DIM):
            raise ValueError(f"Dream-Tac actions must have shape ({ACTION_HORIZON}, {ACTION_DIM}), got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Dream-Tac actions contain NaN or Inf")

        action_space = response.get("action_space")
        if action_space != "absolute_tcp18_absolute_gripper2":
            raise ValueError(
                "Robot client requires absolute Dream-Tac actions; start the server with "
                f"--action-output absolute_from_state, got action_space={action_space!r}"
            )

        result: dict[str, Any] = {
            "actions": np.ascontiguousarray(actions),
            "action_space": action_space,
        }
        for key in ("server_timing", "normalized_action_clipped_fraction", "observation_seq"):
            if key in response:
                result[key] = response[key]
        return result

    @override
    def reset(self) -> None:
        self._policy.reset()

    @override
    def warmup(self, obs: dict) -> None:
        self._policy.warmup(build_policy_payload(obs, default_prompt=self._default_prompt))
