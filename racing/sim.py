"""Race simulation glue for tracks, cars, progress, laps, and telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from racing.physics import Car, DT
from racing.track import Track


@dataclass(slots=True)
class CarProgress:
    """Progress and lap timing state for one car."""

    raw_s: float
    progress_s: float
    lap_count: int = 0
    lap_time: float = 0.0
    last_lap: float | None = None
    lateral_offset: float = 0.0
    on_track: bool = True


class RaceSim:
    """Simulation state for a generated track and one or more cars."""

    def __init__(
        self,
        track: Track,
        cars: Sequence[Car] | None = None,
        *,
        dt: float = DT,
        max_laps: int | None = None,
    ) -> None:
        self.track = track
        self.dt = float(dt)
        self.max_laps = max_laps
        if cars is None:
            cars = [Car(track.point_at(0.0), track.heading_at(0.0))]
        self.cars = list(cars)
        self.t = 0.0
        self.progress: list[CarProgress] = []
        self.last_telemetry: list[dict[str, float | bool | None]] = []
        for car in self.cars:
            s, lateral = self.track.localize(car.state.position)
            # A car on the start line can localize to s ~= length instead of 0,
            # which would count as a completed lap at spawn; wrap it negative.
            progress_s = s - track.length if s > track.length * 0.5 else s
            self.progress.append(
                CarProgress(raw_s=s, progress_s=progress_s, lateral_offset=lateral, on_track=abs(lateral) <= track.half_width)
            )

    def reset(self) -> None:
        """Reset all cars to the start line and clear race timers."""

        start = self.track.point_at(0.0)
        heading = self.track.heading_at(0.0)
        for car in self.cars:
            car.reset(start, heading)
        self.t = 0.0
        self.progress = [
            CarProgress(raw_s=0.0, progress_s=0.0, lateral_offset=0.0, on_track=True) for _ in self.cars
        ]
        self.last_telemetry = []

    def step(self, controls: Sequence[tuple[float, float]]) -> list[dict[str, float | bool | None]]:
        """Advance the simulation by one step using ``(steer, throttle)`` per car."""

        if len(controls) != len(self.cars):
            raise ValueError("controls length must match number of cars")

        telemetry: list[dict[str, float | bool | None]] = []
        for i, (car, control) in enumerate(zip(self.cars, controls, strict=True)):
            prog = self.progress[i]
            steer, throttle = control
            car.step(steer, throttle, prog.on_track)
            raw_s, lateral = self.track.localize(car.state.position)
            delta = raw_s - prog.raw_s
            if delta > self.track.length * 0.5:
                delta -= self.track.length
            elif delta < -self.track.length * 0.5:
                delta += self.track.length
            if delta > -self.track.length * 0.02:
                prog.progress_s += delta
            prog.raw_s = raw_s
            prog.lateral_offset = lateral
            prog.on_track = abs(lateral) <= self.track.half_width

            old_lap = prog.lap_count
            prog.lap_count = max(0, int(prog.progress_s // self.track.length))
            prog.lap_time += self.dt
            if prog.lap_count > old_lap:
                prog.last_lap = prog.lap_time
                prog.lap_time = 0.0

            st = car.state
            telemetry.append(
                {
                    "t": self.t + self.dt,
                    "x": float(st.position[0]),
                    "y": float(st.position[1]),
                    "speed": car.speed,
                    "v_long": float(st.v_long),
                    "v_lat": float(st.v_lat),
                    "steer": float(steer),
                    "throttle": float(throttle),
                    "drift_angle": car.drift_angle,
                    "is_drifting": car.is_drifting,
                    "on_track": prog.on_track,
                    "s": float(prog.progress_s),
                    "lateral_offset": float(lateral),
                    "lap_count": prog.lap_count,
                    "lap_time": float(prog.lap_time),
                    "last_lap": prog.last_lap,
                }
            )

        self.t += self.dt
        self.last_telemetry = telemetry
        return telemetry

    @classmethod
    def from_seed(
        cls, seed: int, difficulty: float = 0.5, car_count: int = 1, *, dt: float = DT
    ) -> "RaceSim":
        """Create a simulation with a generated track and cars on the start line."""

        track = Track.generate(seed, difficulty)
        start = track.point_at(0.0)
        heading = track.heading_at(0.0)
        cars = [Car(np.array(start, dtype=np.float64), heading) for _ in range(car_count)]
        return cls(track, cars, dt=dt)
