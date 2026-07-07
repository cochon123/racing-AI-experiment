"""Historical experiment profiles for archive replays."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from racing.physics import Car


@dataclass(frozen=True, slots=True)
class ExperimentProfile:
    """Physics and reward settings frozen at the time an archived run was trained."""

    slug: str
    title: str
    track_profile: str
    decel_coef: float
    reverse_accel: float
    max_reverse_speed: float | None


FLAT_TRACKS = ExperimentProfile(
    slug="flat-tracks",
    title="Run 1 — Flat Tracks",
    track_profile="gentle",
    decel_coef=0.6,
    reverse_accel=8.0,
    max_reverse_speed=None,
)

REVERSE_EXPLOIT = ExperimentProfile(
    slug="reverse-exploit",
    title="Run 2 — Reverse Exploit",
    track_profile="default",
    decel_coef=2.0,
    reverse_accel=8.0,
    max_reverse_speed=None,
)

PROFILES = {p.slug: p for p in (FLAT_TRACKS, REVERSE_EXPLOIT)}


@contextmanager
def physics_profile(profile: ExperimentProfile) -> Iterator[None]:
    """Temporarily restore Car constants used when an archive was recorded."""

    saved = (
        Car.reverse_accel,
        Car.max_reverse_speed,
    )
    Car.reverse_accel = profile.reverse_accel
    Car.max_reverse_speed = profile.max_reverse_speed if profile.max_reverse_speed is not None else 999.0
    try:
        yield
    finally:
        Car.reverse_accel, Car.max_reverse_speed = saved
