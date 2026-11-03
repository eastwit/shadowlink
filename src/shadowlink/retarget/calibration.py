"""Neutral-pose calibration and its on-disk representation."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from shadowlink.retarget.geometry import ALL_FINGERS, FINGERS, SPREAD_REST, HandMetrics

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_FOUR_FINGERS = tuple(FINGERS)


class CalibrationError(ValueError):
    """Raised when stored calibration data is invalid or incompatible."""


@dataclass(frozen=True, slots=True)
class Neutral:
    flex: dict[str, tuple[float, float, float]]
    spread: dict[str, float]
    opposition: float

    @classmethod
    def default(cls) -> Neutral:
        return cls(
            flex={finger: (0.10, 0.10, 0.10) for finger in ALL_FINGERS},
            spread=dict(SPREAD_REST),
            opposition=0.0,
        )

    @classmethod
    def from_metrics(cls, metrics: HandMetrics) -> Neutral:
        return cls(
            flex={finger: tuple(metrics.flex[finger]) for finger in ALL_FINGERS},
            spread={finger: float(metrics.spread[finger]) for finger in _FOUR_FINGERS},
            opposition=float(metrics.opposition),
        )

    def to_dict(self) -> dict:
        return {
            "flex": {finger: list(self.flex[finger]) for finger in ALL_FINGERS},
            "spread": {finger: float(self.spread[finger]) for finger in _FOUR_FINGERS},
            "opposition": float(self.opposition),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Neutral:
        if not isinstance(data, dict):
            raise CalibrationError("neutral must be an object")
        try:
            raw_flex = data["flex"]
            raw_spread = data["spread"]
            opposition = float(data["opposition"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"invalid neutral: {exc}") from exc
        if not isinstance(raw_flex, dict) or not isinstance(raw_spread, dict):
            raise CalibrationError("neutral.flex and neutral.spread must be objects")
        missing_flex = set(ALL_FINGERS) - set(raw_flex)
        missing_spread = set(_FOUR_FINGERS) - set(raw_spread)
        if missing_flex or missing_spread:
            raise CalibrationError(
                f"neutral is missing fingers: flex={sorted(missing_flex)}, "
                f"spread={sorted(missing_spread)}"
            )
        flex: dict[str, tuple[float, float, float]] = {}
        for finger in ALL_FINGERS:
            values = raw_flex[finger]
            if not isinstance(values, (list, tuple)) or len(values) != 3:
                raise CalibrationError(f"neutral.flex[{finger!r}] must have 3 values")
            try:
                flex[finger] = tuple(float(v) for v in values)  # type: ignore[assignment]
            except (TypeError, ValueError) as exc:
                raise CalibrationError(f"neutral.flex[{finger!r}] is not numeric") from exc
        spread: dict[str, float] = {}
        for finger in _FOUR_FINGERS:
            try:
                spread[finger] = float(raw_spread[finger])
            except (TypeError, ValueError) as exc:
                raise CalibrationError(f"neutral.spread[{finger!r}] is not numeric") from exc
        return cls(flex=flex, spread=spread, opposition=opposition)

    def with_flex(self, finger: str, values: tuple[float, float, float]) -> Neutral:
        flex = dict(self.flex)
        flex[finger] = values
        return replace(self, flex=flex)


@dataclass(frozen=True, slots=True)
class Calibration:
    schema_version: int
    created_at: str
    backend: str
    side: str
    neutral: Neutral

    @classmethod
    def create(cls, backend: str, side: str, neutral: Neutral) -> Calibration:
        return cls(
            schema_version=SCHEMA_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            backend=backend,
            side=side,
            neutral=neutral,
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "backend": self.backend,
            "side": self.side,
            "neutral": self.neutral.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Calibration:
        if not isinstance(data, dict):
            raise CalibrationError("calibration root must be an object")
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise CalibrationError(
                f"unsupported calibration schema {version!r}; expected {SCHEMA_VERSION}"
            )
        try:
            return cls(
                schema_version=int(version),
                created_at=str(data.get("created_at", "")),
                backend=str(data.get("backend", "")),
                side=str(data.get("side", "")),
                neutral=Neutral.from_dict(data["neutral"]),
            )
        except KeyError as exc:
            raise CalibrationError(f"calibration is missing {exc}") from exc


class CalibrationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def load(self, backend: str | None = None, *, strict: bool = False) -> Calibration | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CalibrationError(f"cannot read calibration {self.path}: {exc}") from exc
        calibration = Calibration.from_dict(data)
        if backend and calibration.backend and calibration.backend != backend:
            message = (
                f"calibration at {self.path} was saved for backend "
                f"{calibration.backend!r}, not {backend!r}"
            )
            if strict:
                raise CalibrationError(message)
            log.warning("%s; ignoring it (pass --calibration to use another file)", message)
            return None
        return calibration

    def save(self, calibration: Calibration) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(calibration.to_dict(), indent=2, sort_keys=True) + "\n"
        fd, temp_name = tempfile.mkstemp(
            prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                temp_path.unlink(missing_ok=True)
            raise

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)
