"""Tests for detection filtering and picking."""

from __future__ import annotations

import numpy as np
import pytest

from shadowlink.perception.selector import DetectionSelector, filter_by_side, iou
from shadowlink.types import HandDetection, Side


def detection(
    box,
    *,
    apparent: Side = Side.RIGHT,
    anatomical: Side | None = None,
    score: float = 1.0,
) -> HandDetection:
    return HandDetection(
        bbox_xyxy=np.asarray(box, dtype=np.float64),
        apparent_side=apparent,
        anatomical_side=anatomical if anatomical is not None else apparent,
        score=score,
    )


def test_center_picks_the_image_center_not_the_top_left_corner():
    # Both boxes have the same area.  The old bug minimised cx^2 + cy^2, which
    # would have picked the box at the origin.
    top_left = detection((0, 0, 20, 20))
    middle = detection((80, 40, 100, 60))
    selector = DetectionSelector("center")
    chosen = selector.select([top_left, middle], width=200, height=100)
    assert chosen is middle


def test_largest_picks_the_largest_box():
    small = detection((0, 0, 10, 10))
    large = detection((50, 50, 90, 90))
    selected = DetectionSelector("largest").select([small, large], 200, 100)
    assert selected is large


def test_track_follows_the_previous_hand_even_when_another_grows():
    selector = DetectionSelector("track")
    first = detection((10, 10, 40, 40))
    assert selector.select([first], 200, 100) is first

    same = detection((12, 12, 42, 42))
    bigger_elsewhere = detection((100, 20, 180, 90))
    chosen = selector.select([same, bigger_elsewhere], 200, 100)
    assert chosen is same


def test_track_falls_back_to_largest_and_resets():
    selector = DetectionSelector("track")
    assert selector.select([], 200, 100) is None
    large = detection((10, 10, 80, 80))
    small = detection((120, 10, 140, 30))
    assert selector.select([small, large], 200, 100) is large


def test_filter_uses_anatomical_side():
    mirrored_right = detection((0, 0, 10, 10), apparent=Side.LEFT, anatomical=Side.RIGHT)
    left = detection((20, 0, 30, 10), apparent=Side.RIGHT, anatomical=Side.LEFT)
    assert filter_by_side([mirrored_right, left], "any") == [mirrored_right, left]
    assert filter_by_side([mirrored_right, left], "left") == [left]
    assert filter_by_side([mirrored_right, left], "right") == [mirrored_right]
    with pytest.raises(ValueError):
        filter_by_side([], "both")


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1.0 / 3.0)
