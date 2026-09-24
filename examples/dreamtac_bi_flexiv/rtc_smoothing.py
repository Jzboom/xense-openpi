"""Bound velocity and acceleration of generated absolute RTC targets."""

import math

import numpy as np
from lerobot.utils.robot_utils import quaternion_to_rotation_6d, rotation_6d_to_quaternion

# At 30 Hz: 8 mm/step and 3 mm/step^2 for each TCP, 2 deg/step and
# 1 deg/step^2 for orientation, 0.1 and 0.04 units/step^2 for grippers.
# These preserve the old speed ceiling while removing instantaneous reversals.
MAX_TCP_SPEED_M_S = 0.24
MAX_TCP_ACCEL_M_S2 = 2.7
MAX_ROTATION_SPEED_RAD_S = math.radians(60.0)
MAX_ROTATION_ACCEL_RAD_S2 = math.radians(900.0)
MAX_GRIPPER_SPEED_S = 3.0
MAX_GRIPPER_ACCEL_S2 = 36.0


def _unit_quaternion(r6d: np.ndarray) -> np.ndarray:
    first = np.asarray(r6d[:3], dtype=np.float64)
    second = np.asarray(r6d[3:6], dtype=np.float64)
    first_norm = float(np.linalg.norm(first))
    if not np.isfinite(r6d).all() or first_norm < 1e-8:
        raise ValueError("RTC action has a degenerate 6D rotation")
    unit_first = first / first_norm
    if np.linalg.norm(second - np.dot(unit_first, second) * unit_first) < 1e-8:
        raise ValueError("RTC action has a degenerate 6D rotation")
    quaternion = np.asarray(rotation_6d_to_quaternion(r6d), dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(quaternion).all() or not math.isfinite(norm) or norm < 1e-8:
        raise ValueError("RTC action has a degenerate 6D rotation")
    return quaternion / norm


def _quaternion_product(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    scalar = first[0] * second[0] - np.dot(first[1:], second[1:])
    vector = first[0] * second[1:] + second[0] * first[1:] + np.cross(first[1:], second[1:])
    return np.concatenate(([scalar], vector))


def _rotation_vector(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    if np.dot(before, after) < 0:
        after = -after
    relative = _quaternion_product(after, before * [1, -1, -1, -1])
    norm = float(np.linalg.norm(relative[1:]))
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(norm, max(0.0, float(relative[0])))
    return relative[1:] * (angle / norm)


def _quaternion_from_rotation_vector(rotation: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.concatenate(([math.cos(angle / 2)], rotation * (math.sin(angle / 2) / angle)))


def _braking_speed(distance: float, speed: float, acceleration: float) -> float:
    """Largest next step that can stop at a stationary target with bounded deceleration."""
    upper = min(distance, speed)
    if upper <= 0:
        return 0.0

    def stopping_distance(step: float) -> float:
        count = math.ceil(step / acceleration)
        return count * step - acceleration * count * (count - 1) / 2

    if stopping_distance(upper) <= distance:
        return upper
    lower = 0.0
    for _ in range(24):
        middle = (lower + upper) / 2
        if stopping_distance(middle) <= distance:
            lower = middle
        else:
            upper = middle
    return lower


def _limit_step(desired: np.ndarray, previous: np.ndarray, speed: float, acceleration: float):
    magnitude = float(np.linalg.norm(desired))
    allowed = _braking_speed(magnitude, speed, acceleration)
    requested = desired * (allowed / magnitude) if magnitude else desired
    change = requested - previous
    change_magnitude = float(np.linalg.norm(change))
    if change_magnitude > acceleration:
        change = change * (acceleration / change_magnitude)
    return previous + change, magnitude > allowed + 1e-12 or change_magnitude > acceleration


def limit_action_chunk(
    actions: np.ndarray,
    anchor: np.ndarray,
    *,
    start_index: int,
    frequency_hz: float,
    previous_anchor: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Filter only the generated suffix, seeded by the last two frozen targets.

    The returned chunk is used for robot execution and the next RTC prefix.
    Freezing those exact targets keeps the model and queue aligned.
    """
    chunk = np.asarray(actions, dtype=np.float32)
    anchor = np.asarray(anchor, dtype=np.float32)
    before_anchor = anchor if previous_anchor is None else np.asarray(previous_anchor, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != 20 or anchor.shape != (20,) or before_anchor.shape != (20,):
        raise ValueError("RTC smoothing expects (horizon, 20) actions and two 20D anchors")
    if not np.isfinite(chunk).all() or not np.isfinite(anchor).all() or not np.isfinite(before_anchor).all():
        raise ValueError("RTC smoothing requires finite actions and anchors")
    if not isinstance(start_index, int) or not 0 <= start_index <= len(chunk):
        raise ValueError("RTC smoothing start_index is outside the action chunk")
    if not math.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("RTC smoothing frequency_hz must be positive")

    max_position = MAX_TCP_SPEED_M_S / frequency_hz
    max_position_accel = MAX_TCP_ACCEL_M_S2 / frequency_hz**2
    max_rotation = MAX_ROTATION_SPEED_RAD_S / frequency_hz
    max_rotation_accel = MAX_ROTATION_ACCEL_RAD_S2 / frequency_hz**2
    max_gripper = MAX_GRIPPER_SPEED_S / frequency_hz
    max_gripper_accel = MAX_GRIPPER_ACCEL_S2 / frequency_hz**2
    smoothed = chunk.copy()
    previous = anchor.copy()
    position_steps = {base: np.asarray(anchor[base:base+3] - before_anchor[base:base+3], dtype=np.float64) for base in (0, 9)}
    rotation_steps = {
        base: _rotation_vector(_unit_quaternion(before_anchor[base+3:base+9]), _unit_quaternion(anchor[base+3:base+9]))
        for base in (0, 9)
    }
    gripper_steps = {index: float(anchor[index] - before_anchor[index]) for index in (18, 19)}
    metrics: dict[str, float | int] = {
        "limited_steps": 0,
        "max_raw_translation_m": 0.0,
        "max_raw_rotation_deg": 0.0,
        "max_raw_gripper": 0.0,
    }
    for index in range(start_index, len(smoothed)):
        target = smoothed[index].copy()
        limited = False
        for base in (0, 9):
            raw_step = np.asarray(target[base:base+3] - previous[base:base+3], dtype=np.float64)
            distance = float(np.linalg.norm(raw_step))
            metrics["max_raw_translation_m"] = max(metrics["max_raw_translation_m"], distance)
            step, clipped = _limit_step(raw_step, position_steps[base], max_position, max_position_accel)
            if clipped:
                target[base:base+3] = previous[base:base+3] + step
                limited = True
            position_steps[base] = np.asarray(target[base:base+3] - previous[base:base+3], dtype=np.float64)

            before_quaternion = _unit_quaternion(previous[base+3:base+9])
            after_quaternion = _unit_quaternion(target[base+3:base+9])
            requested_rotation = _rotation_vector(before_quaternion, after_quaternion)
            metrics["max_raw_rotation_deg"] = max(
                metrics["max_raw_rotation_deg"], math.degrees(float(np.linalg.norm(requested_rotation)))
            )
            rotation_step, clipped = _limit_step(
                requested_rotation, rotation_steps[base], max_rotation, max_rotation_accel
            )
            if clipped:
                result = _quaternion_product(_quaternion_from_rotation_vector(rotation_step), before_quaternion)
                result /= np.linalg.norm(result)
                target[base+3:base+9] = quaternion_to_rotation_6d(*result)
                limited = True
            rotation_steps[base] = rotation_step

        for grip_index in (18, 19):
            desired = float(np.clip(target[grip_index], 0.0, 1.0))
            raw_step = desired - float(previous[grip_index])
            metrics["max_raw_gripper"] = max(metrics["max_raw_gripper"], abs(raw_step))
            step, _ = _limit_step(
                np.array([raw_step]), np.array([gripper_steps[grip_index]]),
                max_gripper, max_gripper_accel,
            )
            next_value = float(np.clip(float(previous[grip_index]) + step[0], 0.0, 1.0))
            if next_value != float(target[grip_index]):
                limited = True
            target[grip_index] = next_value
            gripper_steps[grip_index] = next_value - float(previous[grip_index])

        smoothed[index] = target
        previous = target
        if limited:
            metrics["limited_steps"] += 1
    return smoothed, metrics
