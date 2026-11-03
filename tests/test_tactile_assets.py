"""Tests that the packaged tactile XML files are complete and consistent."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from shadowlink.paths import TACTILE_DIR

FINGERS = ("ff", "mf", "rf", "lf", "th")
SITE_RE = re.compile(r"(ff|mf|rf|lf|th)_taxel_r(\d+)_c(\d+)")
TOUCH_RE = re.compile(r"touch_(ff|mf|rf|lf|th)_taxel_r(\d+)_c(\d+)")


def parse(path):
    return ET.parse(path).getroot()


def test_every_finger_has_a_full_taxel_grid():
    for finger in FINGERS:
        root = parse(TACTILE_DIR / f"{finger}_sites.xml")
        sites = root.findall("site")
        assert len(sites) == 24, finger
        names = [site.get("name") for site in sites]
        assert len(set(names)) == 24
        for name in names:
            match = SITE_RE.fullmatch(name)
            assert match, name
            assert match.group(1) == finger
            assert 0 <= int(match.group(2)) < 6
            assert 0 <= int(match.group(3)) < 4


def test_touch_sensors_match_the_sites():
    sites = set()
    for finger in FINGERS:
        root = parse(TACTILE_DIR / f"{finger}_sites.xml")
        sites.update(site.get("name") for site in root.findall("site"))

    sensors = parse(TACTILE_DIR / "sensors.xml").findall("touch")
    assert len(sensors) == 120
    touched = set()
    for sensor in sensors:
        name = sensor.get("name")
        site = sensor.get("site")
        match = TOUCH_RE.fullmatch(name or "")
        assert match, name
        assert site == name[len("touch_") :]
        touched.add(site)
    assert touched == sites


def test_sites_have_normals_and_initial_colors():
    for finger in FINGERS:
        root = parse(TACTILE_DIR / f"{finger}_sites.xml")
        for site in root.findall("site"):
            assert site.get("type") == "ellipsoid"
            quat = [float(value) for value in (site.get("quat") or "").split()]
            assert len(quat) == 4
            assert abs(sum(value * value for value in quat) - 1.0) < 1e-4
            rgba = [float(value) for value in (site.get("rgba") or "").split()]
            assert rgba[:3] == [0.15, 0.65, 1.0]


def test_right_hand_includes_all_tactile_files():
    text = (TACTILE_DIR.parent / "right_hand.xml").read_text(encoding="utf-8")
    for finger in FINGERS:
        assert f"tactile/{finger}_sites.xml" in text, finger
    assert "tactile/sensors.xml" in text


def test_left_hand_model_still_parses():
    # The left hand is shipped for reference but is not used by v0.1; it has no
    # tactile arrays, so only the MJCF structure is checked.
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(TACTILE_DIR.parent / "left_hand.xml"))
    assert model.nu == 20
    assert model.nq == 24
