import math
import time

import numpy as np
import pytest
from lerobot.utils.robot_utils import quaternion_to_rotation_6d, rotation_6d_to_quaternion

from examples.dreamtac_bi_flexiv.rtc_client import DreamTacRTCActionChunkBroker
from examples.dreamtac_bi_flexiv.rtc_smoothing import limit_action_chunk


def action(x=0.0, angle=0.0, grip=0.0):
    target = np.zeros(20, dtype=np.float32)
    target[0] = x
    target[9] = -x
    rotation = quaternion_to_rotation_6d(math.cos(angle / 2), math.sin(angle / 2), 0, 0)
    target[3:9] = rotation
    target[12:18] = rotation
    target[18:] = grip
    return target


def assert_bounded(sequence, frequency_hz):
    position_limit = 0.24 / frequency_hz + 1e-6
    position_accel = 2.7 / frequency_hz**2 + 2e-6
    rotation_limit = math.radians(60) / frequency_hz + 3e-5
    grip_limit = 3.0 / frequency_hz + 1e-6
    grip_accel = 36 / frequency_hz**2 + 2e-6
    for before, after in zip(sequence[:-1], sequence[1:]):
        for base in (0, 9):
            assert np.linalg.norm(after[base:base + 3] - before[base:base + 3]) <= position_limit
            q1 = rotation_6d_to_quaternion(before[base + 3:base + 9])
            q2 = rotation_6d_to_quaternion(after[base + 3:base + 9])
            angle = 2 * math.acos(np.clip(abs(float(np.dot(q1, q2))), -1, 1))
            assert angle <= rotation_limit
        assert np.max(np.abs(after[18:] - before[18:])) <= grip_limit
    for before, middle, after in zip(sequence[:-2], sequence[1:-1], sequence[2:]):
        for base in (0, 9):
            previous_step = middle[base:base + 3] - before[base:base + 3]
            current_step = after[base:base + 3] - middle[base:base + 3]
            assert np.linalg.norm(current_step - previous_step) <= position_accel
        assert np.max(np.abs(after[18:] - 2 * middle[18:] + before[18:])) <= grip_accel


@pytest.mark.parametrize('frequency_hz', [30, 60])
def test_generated_suffix_is_bounded_and_frozen_prefix_is_exact(frequency_hz):
    anchor = action()
    chunk = np.repeat(anchor[None], 40, axis=0)
    chunk[11:] = action(x=0.12, angle=math.pi / 2, grip=1.0)
    original = chunk.copy()
    smoothed, metrics = limit_action_chunk(chunk, chunk[10], start_index=11, frequency_hz=frequency_hz)
    np.testing.assert_array_equal(chunk, original)
    np.testing.assert_array_equal(smoothed[:11], chunk[:11])
    assert_bounded(np.concatenate([chunk[9:11], smoothed[11:]], axis=0), frequency_hz)
    assert metrics['limited_steps'] > 0
    assert metrics['max_raw_translation_m'] >= 0.12 - 1e-5
    assert metrics['max_raw_rotation_deg'] >= 89.9
    assert metrics['max_raw_gripper'] == 1.0


def test_direction_reversal_decelerates_before_changing_direction():
    before = action(x=-0.008, angle=-math.radians(2.0), grip=0.4)
    anchor = action(x=0.0, angle=0.0, grip=0.5)
    chunk = np.repeat(action(x=-0.12, angle=-math.pi / 2, grip=0.0)[None], 40, axis=0)
    smoothed, _ = limit_action_chunk(
        chunk, anchor, previous_anchor=before, start_index=0, frequency_hz=30,
    )
    assert smoothed[0, 0] > anchor[0]
    assert smoothed[0, 8] > anchor[8]  # orientation is still moving forward
    assert smoothed[0, 18] > anchor[18]
    sequence = np.concatenate([before[None], anchor[None], smoothed])
    assert_bounded(sequence, 30)
    # For rotations about x, the second 6D column gives the signed angle.
    angles = np.unwrap(np.arctan2(sequence[:, 8], sequence[:, 7]))
    assert np.max(np.abs(np.diff(angles, n=2))) <= math.radians(1.0) + 1e-5


def test_invalid_rotation_fails_before_queue_merge():
    chunk = np.repeat(action()[None], 40, axis=0)
    chunk[11, 3:9] = 0
    with pytest.raises(ValueError, match='degenerate'):
        limit_action_chunk(chunk, chunk[10], start_index=11, frequency_hz=30)


def test_broker_smooths_warmup_and_live_chunk_and_reuses_smoothed_prefix():
    class JumpPolicy:
        def __init__(self):
            self.calls = []

        def infer(self, obs, **kwargs):
            previous = kwargs['prev_chunk_left_over']
            delay = kwargs['inference_delay']
            actions = np.repeat(action(x=0.12 if previous is None else -0.12,
                                       angle=math.pi / 2, grip=1.0)[None], 40, axis=0)
            if previous is not None:
                actions[:delay] = previous[:delay]
            self.calls.append((None if previous is None else previous.copy(), actions.copy()))
            return {'actions': actions}

        def reset(self):
            pass

    policy = JumpPolicy()
    broker = DreamTacRTCActionChunkBroker(policy, smooth_actions=True)
    observation = {'state': action()}
    try:
        broker.warmup(observation)
        assert_bounded(np.concatenate([observation['state'][None], broker._action_queue.queue]), 30)
        np.testing.assert_array_equal(policy.calls[1][0], broker._action_queue.queue[:14])
        for _ in range(26):
            broker.infer(observation)
        deadline = time.monotonic() + 3
        while broker._inference_seq < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert broker._inference_seq >= 1
        np.testing.assert_array_equal(broker._action_queue.queue, broker._action_queue.original_queue)
        assert_bounded(broker._action_queue.queue, 30)
        assert len(policy.calls) >= 3
        np.testing.assert_array_equal(policy.calls[2][0], policy.calls[2][1][:len(policy.calls[2][0])])
    finally:
        broker.stop()
