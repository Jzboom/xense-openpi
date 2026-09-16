"""Dream-Tac observation preprocessing for the BiFlexiv robot client."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import math

import cv2
import numpy as np

RGB_CAMERA_KEYS = (
    "head",
    "left_wrist",
    "right_wrist",
)
LEFT_TACTILE_KEYS = ("left_tactile_left", "left_tactile_right")
RIGHT_TACTILE_KEYS = ("right_tactile_left", "right_tactile_right")
TACTILE_KEYS = LEFT_TACTILE_KEYS + RIGHT_TACTILE_KEYS
CAMERA_KEYS = RGB_CAMERA_KEYS + TACTILE_KEYS
MERGED_TACTILE_KEYS = ("left_tactile_merged", "right_tactile_merged")
TRANSPORT_CAMERA_KEYS = RGB_CAMERA_KEYS + MERGED_TACTILE_KEYS

STATE_DIM = 20
ACTION_DIM = 20
ACTION_HORIZON = 40
DEFAULT_IMAGE_SIZE = 224
TACTILE_IMAGE_SHAPE = (400, 700, 3)
TACTILE_CONTENT_WIDTH = 196
TACTILE_HORIZONTAL_PADDING = 14
HISTORY_FRAMES = 4
RGB_HISTORY_OFFSETS = (-3, -2, -1, 0)
TACTILE_HISTORY_OFFSETS = (-3, -2, -1, 0)
FUTURE_IMAGE_OFFSETS = (10, 20, 30, 40)


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
    """Build the five compact condition views sent to the policy server."""
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if image_size != DEFAULT_IMAGE_SIZE:
        raise ValueError(f"Dream-Tac merged tactile preprocessing requires image_size={DEFAULT_IMAGE_SIZE}")
    missing = [name for name in CAMERA_KEYS if name not in images]
    if missing:
        raise ValueError(f"Missing Dream-Tac cameras: {missing}")

    prepared: dict[str, np.ndarray] = {}
    for name in RGB_CAMERA_KEYS:
        image = coerce_hwc_uint8(images[name], name=name)
        if image.shape[:2] != (image_size, image_size):
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        prepared[name] = np.ascontiguousarray(image)

    for arm, pair in (
        ("left", LEFT_TACTILE_KEYS),
        ("right", RIGHT_TACTILE_KEYS),
    ):
        first = coerce_hwc_uint8(images[pair[0]], name=pair[0])
        second = coerce_hwc_uint8(images[pair[1]], name=pair[1])
        for name, image in zip(pair, (first, second), strict=True):
            if image.shape != TACTILE_IMAGE_SHAPE:
                raise ValueError(
                    f"Tactile image {name!r} must retain raw shape {TACTILE_IMAGE_SHAPE}, got {image.shape}"
                )
        stacked = np.concatenate((first, second), axis=0)
        resized = cv2.resize(
            stacked,
            (TACTILE_CONTENT_WIDTH, image_size),
            interpolation=cv2.INTER_AREA,
        )
        merged = cv2.copyMakeBorder(
            resized,
            0,
            0,
            TACTILE_HORIZONTAL_PADDING,
            TACTILE_HORIZONTAL_PADDING,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        expected_shape = (image_size, image_size, 3)
        if merged.shape != expected_shape:
            raise RuntimeError(f"Unexpected merged tactile shape: {merged.shape}, expected {expected_shape}")
        prepared[f"{arm}_tactile_merged"] = np.ascontiguousarray(merged)
    return prepared


class CameraHistoryBuffer:
    """Assemble the recent RGB and tactile histories used in training."""

    def __init__(self) -> None:
        self._frames = {
            name: deque(maxlen=-min(RGB_HISTORY_OFFSETS) + 1) for name in RGB_CAMERA_KEYS
        }
        self._frames.update(
            {name: deque(maxlen=-min(TACTILE_HISTORY_OFFSETS) + 1) for name in MERGED_TACTILE_KEYS}
        )

    def reset(self) -> None:
        for frames in self._frames.values():
            frames.clear()

    def update(self, images: Mapping[str, object]) -> dict[str, np.ndarray]:
        """Append one synchronized observation and return chronological THWC histories.

        Before enough observations exist, offsets before the beginning are
        clamped to the first frame, matching the training dataset.
        """
        missing = [name for name in TRANSPORT_CAMERA_KEYS if name not in images]
        if missing:
            raise ValueError(f"Missing Dream-Tac cameras for history: {missing}")

        for name in TRANSPORT_CAMERA_KEYS:
            current = coerce_hwc_uint8(images[name], name=name)
            frames = self._frames[name]
            if frames and current.shape != frames[-1].shape:
                raise ValueError(
                    f"Camera {name!r} changed shape from {frames[-1].shape} to {current.shape}"
                )
            frames.append(current.copy())

        histories: dict[str, np.ndarray] = {}
        for name in TRANSPORT_CAMERA_KEYS:
            offsets = RGB_HISTORY_OFFSETS if name in RGB_CAMERA_KEYS else TACTILE_HISTORY_OFFSETS
            frames = self._frames[name]
            latest = len(frames) - 1
            histories[name] = np.ascontiguousarray(
                np.stack([frames[max(0, latest + offset)] for offset in offsets], axis=0)
            )
        return histories


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
