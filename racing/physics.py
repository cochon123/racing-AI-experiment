"""Deterministic top-down car physics for the racing experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


DT = 1.0 / 60.0


@dataclass(slots=True)
class CarState:
    """Mutable car state in world and car-body coordinates."""

    position: np.ndarray
    heading: float
    v_long: float = 0.0
    v_lat: float = 0.0
    yaw_rate: float = 0.0


class Car:
    """A deterministic drift-capable top-down car model."""

    dt: float = DT
    engine_accel: float = 30.0
    brake_accel: float = 48.0
    reverse_accel: float = 3.0
    max_reverse_speed: float = 6.0
    lateral_stiffness: float = 24.0
    max_lateral_grip: float = 28.0
    rolling_drag: float = 0.22
    air_drag: float = 0.012
    offtrack_factor: float = 0.35
    offtrack_extra_linear_drag: float = 0.90
    offtrack_extra_quad_drag: float = 0.025
    max_yaw_rate: float = 2.75
    yaw_response: float = 8.0

    def __init__(self, position: np.ndarray, heading: float = 0.0) -> None:
        self.state = CarState(np.asarray(position, dtype=np.float64).copy(), float(heading))
        self.last_steer = 0.0
        self.last_throttle = 0.0

    def reset(self, position: np.ndarray, heading: float = 0.0) -> None:
        """Reset the car to ``position`` and ``heading`` with zero velocity."""

        self.state = CarState(np.asarray(position, dtype=np.float64).copy(), float(heading))
        self.last_steer = 0.0
        self.last_throttle = 0.0

    def step(self, steer: float, throttle: float, on_track: bool) -> None:
        """Advance the car by one fixed ``dt`` step."""

        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip(throttle, -1.0, 1.0))
        st = self.state
        grip_scale = 1.0 if on_track else self.offtrack_factor
        engine_scale = 1.0 if on_track else self.offtrack_factor

        speed_abs = abs(st.v_long)
        speed_total = self.speed
        steering_scale = 0.38 + 0.62 / (1.0 + speed_total / 18.0)
        target_yaw = steer * self.max_yaw_rate * steering_scale
        st.yaw_rate += (target_yaw - st.yaw_rate) * min(1.0, self.yaw_response * self.dt)

        if throttle >= 0.0:
            long_accel = throttle * self.engine_accel * engine_scale
        elif st.v_long > 1.0:
            long_accel = throttle * self.brake_accel
        else:
            long_accel = throttle * self.reverse_accel * engine_scale

        lateral_demand = -self.lateral_stiffness * st.v_lat
        max_lat = self.max_lateral_grip * grip_scale
        lateral_accel = float(np.clip(lateral_demand, -max_lat, max_lat))
        slide_ratio = min(1.0, abs(lateral_demand) / max(max_lat, 1e-9))
        long_accel *= 1.0 - 0.35 * slide_ratio

        drag = self.rolling_drag * st.v_long + self.air_drag * st.v_long * abs(st.v_long)
        if not on_track:
            drag += (
                self.offtrack_extra_linear_drag * st.v_long
                + self.offtrack_extra_quad_drag * st.v_long * abs(st.v_long)
            )
        long_accel -= drag

        # Body-frame velocity derivatives include rotating-frame coupling.
        st.v_long += (long_accel + st.yaw_rate * st.v_lat) * self.dt
        # Reverse gear is for getting unstuck only; without this cap, driving
        # the whole track backwards becomes a way to dodge deceleration.
        st.v_long = max(st.v_long, -self.max_reverse_speed)
        st.v_lat += (lateral_accel - st.yaw_rate * st.v_long) * self.dt
        if abs(st.v_long) < 0.03 and abs(throttle) < 0.05:
            st.v_long = 0.0
        if abs(st.v_lat) < 0.01:
            st.v_lat = 0.0

        st.heading = (st.heading + st.yaw_rate * self.dt + math.pi) % (2.0 * math.pi) - math.pi
        forward = np.array([math.cos(st.heading), math.sin(st.heading)], dtype=np.float64)
        right = np.array([-math.sin(st.heading), math.cos(st.heading)], dtype=np.float64)
        st.position += (forward * st.v_long + right * st.v_lat) * self.dt

        self.last_steer = steer
        self.last_throttle = throttle

    @property
    def velocity_world(self) -> np.ndarray:
        """Return current velocity as a world-space vector."""

        st = self.state
        forward = np.array([math.cos(st.heading), math.sin(st.heading)], dtype=np.float64)
        right = np.array([-math.sin(st.heading), math.cos(st.heading)], dtype=np.float64)
        return forward * st.v_long + right * st.v_lat

    @property
    def speed(self) -> float:
        """Return scalar speed magnitude."""

        st = self.state
        return float(math.hypot(st.v_long, st.v_lat))

    @property
    def drift_angle(self) -> float:
        """Return absolute angle between velocity and car heading in radians."""

        if self.speed < 1e-6:
            return 0.0
        return float(abs(math.atan2(self.state.v_lat, self.state.v_long)))

    @property
    def is_drifting(self) -> bool:
        """Return whether the car is visibly sliding."""

        return self.drift_angle > 0.15 and self.speed > 5.0
