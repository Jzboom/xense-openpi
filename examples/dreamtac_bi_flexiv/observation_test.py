import numpy as np
import pytest

from examples.dreamtac_bi_flexiv.observation import ACTION_HORIZON
from examples.dreamtac_bi_flexiv.observation import CAMERA_KEYS
from examples.dreamtac_bi_flexiv.observation import RGB_CAMERA_KEYS
from examples.dreamtac_bi_flexiv.observation import TACTILE_IMAGE_SHAPE
from examples.dreamtac_bi_flexiv.observation import TACTILE_KEYS
from examples.dreamtac_bi_flexiv.observation import prepare_policy_images


def _images() -> dict[str, np.ndarray]:
    images = {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in RGB_CAMERA_KEYS}
    images.update({name: np.zeros(TACTILE_IMAGE_SHAPE, dtype=np.uint8) for name in TACTILE_KEYS})
    return images


def test_deployment_contract_is_30_steps_with_physical_tactile_names() -> None:
    assert ACTION_HORIZON == 30
    assert CAMERA_KEYS == (
        "head",
        "left_wrist",
        "right_wrist",
        "left_tactile_left",
        "left_tactile_right",
        "right_tactile_left",
        "right_tactile_right",
    )


def test_prepares_rgb_but_preserves_raw_tactile_frames() -> None:
    prepared = prepare_policy_images(_images())

    for name in RGB_CAMERA_KEYS:
        assert prepared[name].shape == (224, 224, 3)
        assert prepared[name].flags.c_contiguous
    for name in TACTILE_KEYS:
        assert prepared[name].shape == TACTILE_IMAGE_SHAPE
        assert prepared[name].flags.c_contiguous


def test_rejects_pre_resized_tactile_frames() -> None:
    images = _images()
    images["left_tactile_left"] = np.zeros((224, 224, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="must retain raw shape"):
        prepare_policy_images(images)
