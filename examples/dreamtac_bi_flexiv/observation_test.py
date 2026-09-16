import numpy as np
import pytest

from examples.dreamtac_bi_flexiv.observation import ACTION_HORIZON
from examples.dreamtac_bi_flexiv.observation import CAMERA_KEYS
from examples.dreamtac_bi_flexiv.observation import CameraHistoryBuffer
from examples.dreamtac_bi_flexiv.observation import HISTORY_FRAMES
from examples.dreamtac_bi_flexiv.observation import MERGED_TACTILE_KEYS
from examples.dreamtac_bi_flexiv.observation import RGB_CAMERA_KEYS
from examples.dreamtac_bi_flexiv.observation import RGB_HISTORY_OFFSETS
from examples.dreamtac_bi_flexiv.observation import TACTILE_IMAGE_SHAPE
from examples.dreamtac_bi_flexiv.observation import TACTILE_HISTORY_OFFSETS
from examples.dreamtac_bi_flexiv.observation import TACTILE_KEYS
from examples.dreamtac_bi_flexiv.observation import TRANSPORT_CAMERA_KEYS
from examples.dreamtac_bi_flexiv.observation import prepare_policy_images


def _images() -> dict[str, np.ndarray]:
    images = {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in RGB_CAMERA_KEYS}
    images.update(
        {
            name: np.full(TACTILE_IMAGE_SHAPE, value, dtype=np.uint8)
            for value, name in enumerate(TACTILE_KEYS, start=1)
        }
    )
    return images


def test_deployment_contract_is_40_steps_with_four_frame_history() -> None:
    assert ACTION_HORIZON == 40
    assert HISTORY_FRAMES == 4
    assert RGB_HISTORY_OFFSETS == (-3, -2, -1, 0)
    assert TACTILE_HISTORY_OFFSETS == (-3, -2, -1, 0)
    assert CAMERA_KEYS == (
        "head",
        "left_wrist",
        "right_wrist",
        "left_tactile_left",
        "left_tactile_right",
        "right_tactile_left",
        "right_tactile_right",
    )
    assert TRANSPORT_CAMERA_KEYS == (
        "head",
        "left_wrist",
        "right_wrist",
        "left_tactile_merged",
        "right_tactile_merged",
    )


def test_prepares_rgb_and_merges_each_raw_tactile_pair_before_transport() -> None:
    prepared = prepare_policy_images(_images())

    assert tuple(prepared) == TRANSPORT_CAMERA_KEYS
    for name in RGB_CAMERA_KEYS:
        assert prepared[name].shape == (224, 224, 3)
        assert prepared[name].flags.c_contiguous
    for name in MERGED_TACTILE_KEYS:
        assert prepared[name].shape == (224, 224, 3)
        assert prepared[name].flags.c_contiguous
        np.testing.assert_array_equal(prepared[name][:, :14], 0)
        np.testing.assert_array_equal(prepared[name][:, 210:], 0)
    np.testing.assert_array_equal(prepared["left_tactile_merged"][40, 40], np.full(3, 1, dtype=np.uint8))
    np.testing.assert_array_equal(prepared["left_tactile_merged"][180, 40], np.full(3, 2, dtype=np.uint8))
    np.testing.assert_array_equal(prepared["right_tactile_merged"][40, 40], np.full(3, 3, dtype=np.uint8))
    np.testing.assert_array_equal(prepared["right_tactile_merged"][180, 40], np.full(3, 4, dtype=np.uint8))


def test_rejects_pre_resized_tactile_frames() -> None:
    images = _images()
    images["left_tactile_left"] = np.zeros((224, 224, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="must retain raw shape"):
        prepare_policy_images(images)


def _tiny_images(value: int) -> dict[str, np.ndarray]:
    return {name: np.full((2, 3, 3), value, dtype=np.uint8) for name in TRANSPORT_CAMERA_KEYS}


def test_history_buffer_uses_recent_rgb_and_tactile_offsets() -> None:
    history = CameraHistoryBuffer()
    for step in range(5):
        result = history.update(_tiny_images(step))

    assert result["head"][:, 0, 0, 0].tolist() == [1, 2, 3, 4]
    assert result["left_tactile_merged"][:, 0, 0, 0].tolist() == [1, 2, 3, 4]
    assert all(value.shape == (HISTORY_FRAMES, 2, 3, 3) for value in result.values())
    assert all(value.flags.c_contiguous for value in result.values())


def test_history_buffer_clamps_to_first_frame_and_reset_clears_history() -> None:
    history = CameraHistoryBuffer()
    for step in range(3):
        result = history.update(_tiny_images(step))

    assert result["head"][:, 0, 0, 0].tolist() == [0, 0, 1, 2]
    assert result["left_tactile_merged"][:, 0, 0, 0].tolist() == [0, 0, 1, 2]

    history.reset()
    result = history.update(_tiny_images(9))
    assert all(value[:, 0, 0, 0].tolist() == [9, 9, 9, 9] for value in result.values())
