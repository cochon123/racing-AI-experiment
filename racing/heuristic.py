"""Observation-only heuristic baseline for the racing environment."""

from __future__ import annotations

import numpy as np


def heuristic_action(obs: np.ndarray) -> np.ndarray:
    """Return a simple lookahead steering and target-speed action."""

    obs = np.asarray(obs, dtype=np.float32)
    heading_error = float(obs[11])
    lateral_offset = float(obs[12])
    speed = float(obs[14] * 40.0)
    lookahead_curves = obs[15:21]

    near_curve = float(np.mean(lookahead_curves[:3]))
    far_curve = float(np.mean(lookahead_curves[3:]))
    curvature_abs = float(np.max(np.abs(lookahead_curves)))

    steer = 1.15 * near_curve + 0.45 * far_curve - 1.7 * heading_error - 0.55 * lateral_offset
    steer = float(np.clip(steer, -1.0, 1.0))

    target_speed = 34.0 - 22.0 * curvature_abs
    target_speed = float(np.clip(target_speed, 10.0, 34.0))
    throttle = (target_speed - speed) / 10.0
    if speed > target_speed + 5.0:
        throttle -= 0.35
    throttle = float(np.clip(throttle, -1.0, 1.0))
    return np.array([steer, throttle], dtype=np.float32)
