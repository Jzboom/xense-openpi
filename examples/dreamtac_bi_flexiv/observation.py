"""Dream-Tac observation preprocessing for the BiFlexiv robot client."""

from __future__ import annotations

from collections.abc import Mapping
import math

import cv2
import numpy as np

CAMERA_KEYS = (
    "head",
    "left_wrist",
    "right_wrist",
    "left_tactile_0",
    "left_tactile_1",
    "right_tactile_0",
    "right_tactile_1",
)
TACTILE_KEYS = CAMERA_KEYS[3:]
LEFT_TACTILE_KEYS = ("left_tactile_0", "left_tactile_1")
RIGHT_TACTILE_KEYS = ("right_tactile_0", "right_tactile_1")

STATE_DIM = 20
ACTION_DIM = 20
ACTION_HORIZON = 20
DEFAULT_IMAGE_SIZE = 224


def coerce_hwc_uint8(image: object, *, name: str) -> np.ndarray:
    """Return a contiguous HWC uint8 RGB image without resizing it."""
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Image {name!r} must have three dimensions, got {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError(f"Image {name!r} must be HWC or CHW RGB, got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"Image {name!r} must have dtype uint8, got {array.dtype}")
    return np.ascontiguousarray(array)


def prepare_policy_images(
    images: Mapping[str, object], *, image_size: int = DEFAULT_IMAGE_SIZE
) -> dict[str, np.ndarray]:
    """Validate and directly resize the seven Dream-Tac views to square HWC images."""
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    missing = [name for name in CAMERA_KEYS if name not in images]
    if missing:
        raise ValueError(f"Missing Dream-Tac cameras: {missing}")

    prepared: dict[str, np.ndarray] = {}
    for name in CAMERA_KEYS:
        image = coerce_hwc_uint8(images[name], name=name)
        if image.shape[:2] != (image_size, image_size):
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        prepared[name] = np.ascontiguousarray(image)
    return prepared


def mean_abs_frame_delta(current: np.ndarray, previous: np.ndarray) -> float:
    """Mean absolute RGB difference on the normalized uint8 [0, 1] scale."""
    if current.shape != previous.shape:
        raise ValueError(
            f"Consecutive tactile frames must have the same shape, got {current.shape} and {previous.shape}"
        )
    return float(np.abs(current.astype(np.float32) - previous.astype(np.float32)).mean() / 255.0)


def scalar_gate_from_raw(
    raw: float,
    *,
    median: float = 0.002,
    mad: float = 0.001,
    sigmoid_k: float = 4.0,
    g_min: float = 0.15,
    g_max: float = 1.0,
) -> float:
    """Map a tactile frame-delta event to the fixed Dream-Tac gate range."""
    z = sigmoid_k * (raw - median) / (mad + 1e-6)
    z = max(-30.0, min(30.0, z))
    g01 = 1.0 / (1.0 + math.exp(-z))
    return g_min + (g_max - g_min) * g01


class TactileGateTracker:
    """Compute the two-arm Dream-Tac gate from consecutive raw tactile frames."""

    def __init__(self) -> None:
        self._previous: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._previous.clear()

    def update(self, images: Mapping[str, object]) -> np.ndarray:
        current = {name: coerce_hwc_uint8(images[name], name=name) for name in TACTILE_KEYS if name in images}
        missing = [name for name in TACTILE_KEYS if name not in current]
        if missing:
            raise ValueError(f"Missing tactile cameras required for the Dream-Tac gate: {missing}")

        def arm_raw(keys: tuple[str, str]) -> float:
            if any(name not in self._previous for name in keys):
                return 0.0
            return max(mean_abs_frame_delta(current[name], self._previous[name]) for name in keys)

        left_raw = arm_raw(LEFT_TACTILE_KEYS)
        right_raw = arm_raw(RIGHT_TACTILE_KEYS)
        self._previous = {name: image.copy() for name, image in current.items()}
        return np.asarray([scalar_gate_from_raw(left_raw), scalar_gate_from_raw(right_raw)], dtype=np.float32)
