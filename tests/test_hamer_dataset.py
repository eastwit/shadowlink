"""Tests for the lightweight HaMeR crop preprocessing."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from shadowlink.perception.hamer_dataset import expand_to_aspect_ratio, prepare_batch


def make_cfg(image_size: int = 64) -> SimpleNamespace:
    return SimpleNamespace(
        MODEL=SimpleNamespace(
            IMAGE_SIZE=image_size,
            IMAGE_MEAN=[0.485, 0.456, 0.406],
            IMAGE_STD=[0.229, 0.224, 0.225],
            BBOX_SHAPE=[48, 64],
        )
    )


def test_batch_shapes_and_normalisation():
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    # Centre the box and keep the expanded 3:4 crop inside the frame so the
    # whole patch is white (out-of-frame pixels would be black borders).
    batch = prepare_batch(
        make_cfg(), image, np.array([[75.0, 75.0, 125.0, 125.0]]), np.array([True])
    )
    assert batch.images.shape == (1, 3, 64, 64)
    assert batch.images.dtype == np.float32
    assert batch.box_center.shape == (1, 2)
    assert batch.box_size.shape == (1,)
    assert batch.img_size.tolist() == [[200.0, 200.0]]
    assert batch.right.tolist() == [1.0]
    assert batch.flipped.tolist() == [False]
    assert np.isfinite(batch.images).all()
    # A white image normalises to (255 - mean*255) / (std*255).
    expected = (255.0 - 255.0 * np.array([0.485, 0.456, 0.406])) / (
        255.0 * np.array([0.229, 0.224, 0.225])
    )
    np.testing.assert_allclose(batch.images[0].mean(axis=(1, 2)), expected, rtol=1e-4)


def test_left_hands_are_flipped_inside_the_crop():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[:, :100] = 255  # bright left half
    box = np.array([[75.0, 75.0, 125.0, 125.0]])
    right = prepare_batch(make_cfg(), image, box, np.array([True]))
    left = prepare_batch(make_cfg(), image, box, np.array([False]))

    assert right.flipped.tolist() == [False]
    assert left.flipped.tolist() == [True]
    # The left-hand crop is mirrored: bright pixels move from left to right.
    assert left.images[0, :, :, :32].mean() < left.images[0, :, :, 32:].mean()
    assert right.images[0, :, :, :32].mean() > right.images[0, :, :, 32:].mean()


def test_multiple_boxes_are_batched():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    boxes = np.array([[10.0, 10.0, 110.0, 110.0], [100.0, 100.0, 190.0, 190.0]])
    batch = prepare_batch(make_cfg(), image, boxes, np.array([True, False]))
    assert len(batch) == 2
    assert batch.flipped.tolist() == [False, True]


def test_empty_batch_is_rejected():
    with pytest.raises(ValueError):
        prepare_batch(make_cfg(), np.zeros((10, 10, 3)), np.zeros((0, 4)), np.zeros(0))


def test_mismatched_boxes_and_rights_are_rejected():
    with pytest.raises(ValueError):
        prepare_batch(
            make_cfg(),
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.array([[0.0, 0.0, 5.0, 5.0]]),
            np.array([True, False]),
        )


def test_expand_to_aspect_ratio():
    np.testing.assert_allclose(
        expand_to_aspect_ratio(np.array([100.0, 100.0]), [192, 256]),
        [100.0, 400.0 / 3.0],
    )
    np.testing.assert_allclose(
        expand_to_aspect_ratio(np.array([100.0, 300.0]), [192, 256]),
        [225.0, 300.0],
    )
    np.testing.assert_allclose(expand_to_aspect_ratio(np.array([4.0, 2.0]), None), [4.0, 2.0])
