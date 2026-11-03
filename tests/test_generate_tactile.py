"""Tests for the offline tactile layout generator."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from shadowlink.paths import TACTILE_DIR
from shadowlink.tools.generate_tactile import COLS, FINGERS, ROWS, check_output, generate

SITE_RE = re.compile(r'<site name="(\w+_taxel_r\d_c\d)"')
TOUCH_RE = re.compile(r'<touch name="touch_(\w+_taxel_r\d_c\d)"')


@pytest.fixture(scope="module")
def generated() -> dict[str, str]:
    return generate()


def test_generates_five_finger_files_and_sensors(generated):
    assert set(generated) == {f"{finger}_sites.xml" for finger in FINGERS} | {"sensors.xml"}


def test_each_finger_has_a_full_grid(generated):
    for finger in FINGERS:
        names = SITE_RE.findall(generated[f"{finger}_sites.xml"])
        assert len(names) == ROWS * COLS
        assert names == sorted(names)
        assert all(name.startswith(f"{finger}_taxel_") for name in names)


def test_sensors_match_the_sites(generated):
    sites = [
        name for finger in FINGERS for name in SITE_RE.findall(generated[f"{finger}_sites.xml"])
    ]
    touches = TOUCH_RE.findall(generated["sensors.xml"])
    assert touches == sites
    assert len(touches) == len(FINGERS) * ROWS * COLS == 120


def test_sites_are_well_formed_includes(generated):
    for content in generated.values():
        assert content.startswith("<mujocoinclude>")
        assert content.rstrip().endswith("</mujocoinclude>")
        assert "nan" not in content.lower()


def test_committed_assets_are_up_to_date(generated):
    assert check_output(generated, TACTILE_DIR) == []


def test_check_detects_stale_and_missing_files(generated, tmp_path: Path):
    for name, content in generated.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    assert check_output(generated, tmp_path) == []

    (tmp_path / "ff_sites.xml").write_text("<mujocoinclude/>\n", encoding="utf-8")
    (tmp_path / "sensors.xml").unlink()
    assert sorted(check_output(generated, tmp_path)) == ["ff_sites.xml", "sensors.xml"]
