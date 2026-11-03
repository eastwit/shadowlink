"""Guard rails: the installed package must not rely on path tricks."""

from __future__ import annotations

import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "shadowlink"

FORBIDDEN = {
    "os.chdir": re.compile(r"\bos\.chdir\s*\("),
    "sys.path.insert": re.compile(r"\bsys\.path\.insert\s*\("),
    "sys.path.append": re.compile(r"\bsys\.path\.append\s*\("),
    "user.home": re.compile(r"[Cc]:[\\/]Users[\\/]"),
    "anaconda": re.compile(r"anaconda3", re.IGNORECASE),
}

HAMER_IMPORT = re.compile(r"^\s*(?:import\s+hamer\b|from\s+hamer(?:\.|\s+import))", re.MULTILINE)


def package_files() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def test_package_has_no_path_hacks():
    offenders = []
    for path in package_files():
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN.items():
            if pattern.search(text):
                offenders.append(f"{path.relative_to(PACKAGE)}: {label}")
    assert offenders == []


def test_hamer_is_imported_lazily_and_only_by_its_adapters():
    offenders = []
    for path in package_files():
        text = path.read_text(encoding="utf-8")
        if HAMER_IMPORT.search(text):
            relative = path.relative_to(PACKAGE)
            if relative.parent.name != "perception" or not relative.name.startswith("hamer_"):
                offenders.append(str(relative))
    assert offenders == []


def test_assets_are_resolved_relative_to_the_package():
    from shadowlink.paths import ASSETS_DIR, PACKAGE_DIR, SCENE_XML

    assert PACKAGE_DIR.is_dir()
    assert ASSETS_DIR.is_relative_to(PACKAGE_DIR)
    assert SCENE_XML.is_file()
    assert SCENE_XML.is_relative_to(PACKAGE_DIR)
