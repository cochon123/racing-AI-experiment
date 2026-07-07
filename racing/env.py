"""Gymnasium environment for the 2D racing reinforcement-learning experiment."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from racing.physics import Car, DT
from racing.sim import RaceSim
from racing.track import Track


RAYCAST_MAX_DIST = 80.0
RAYCAST_ANGLES_DEG = tuple(np.linspace(-100.0, 100.0, 9, dtype=np.float64).tolist())
LOOKAHEAD_DISTANCES = (8.0, 18.0, 30.0, 45.0, 65.0, 90.0)
DEFAULT_DECEL_COEF = 1.0
PROGRESS_REWARD_SCALE = 0.5
TIME_PENALTY = 0.02
LAP_BONUS = 50.0
OFF_TRACK_PENALTY = 0.05
STUCK_SECONDS = 5.0
STUCK_PENALTY = 10.0
MAX_LAPS = 3

OBSERVATION_LAYOUT = (
    "raycast_0_left_100_deg / 80",
    "raycast_1_left_75_deg / 80",
    "raycast_2_left_50_deg / 80",
    "raycast_3_left_25_deg / 80",
    "raycast_4_forward / 80",
    "raycast_5_right_25_deg / 80",
    "raycast_6_right_50_deg / 80",
    "raycast_7_right_75_deg / 80",
    "raycast_8_right_100_deg / 80",
    "v_long / 40",
    "v_lat / 15",
    "heading_error_vs_track_tangent / pi",
    "lateral_offset / half_width",
    "drift_angle / 1.5",
    "speed / 40",
    "curvature_s_plus_8 * 15",
    "curvature_s_plus_18 * 15",
    "curvature_s_plus_30 * 15",
    "curvature_s_plus_45 * 15",
    "curvature_s_plus_65 * 15",
    "curvature_s_plus_90 * 15",
)

REWARD_CONSTANTS = {
    "progress_reward_scale": PROGRESS_REWARD_SCALE,
    "time_penalty_per_action_step": TIME_PENALTY,
    "lap_bonus": LAP_BONUS,
    "off_track_penalty_per_action_step": OFF_TRACK_PENALTY,
    "default_decel_coef": DEFAULT_DECEL_COEF,
    "stuck_seconds": STUCK_SECONDS,
    "stuck_penalty": STUCK_PENALTY,
    "max_laps": MAX_LAPS,
}


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def shared_reward_core(delta_s: float, lap_delta: int, on_track: bool) -> float:
    """Return the reward shared byte-for-byte by both experimental agents."""

    reward = float(delta_s) * PROGRESS_REWARD_SCALE
    reward -= TIME_PENALTY
    reward += int(lap_delta) * LAP_BONUS
    if not on_track:
        reward -= OFF_TRACK_PENALTY
    return float(reward)


class RacingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-car Gymnasium environment backed by the deterministic race core."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_mode: str = "time",
        decel_coef: float = DEFAULT_DECEL_COEF,
        difficulty: float = 0.5,
        seed_pool: tuple[int, int] | None = (0, 400),
        fixed_seed: int | None = None,
        max_episode_seconds: float = 60.0,
        frame_skip: int = 2,
        random_start_offset: bool = True,
        track_profile: str = "default",
    ) -> None:
        if reward_mode not in {"time", "nobrakes"}:
            raise ValueError("reward_mode must be 'time' or 'nobrakes'")
        if frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")
        if seed_pool is not None and seed_pool[1] <= seed_pool[0]:
            raise ValueError("seed_pool must be (low, high) with high > low")

        self.reward_mode = reward_mode
        self.decel_coef = float(decel_coef)
        self.difficulty = float(difficulty)
        self.seed_pool = seed_pool
        self.fixed_seed = fixed_seed
        self.max_episode_seconds = float(max_episode_seconds)
        self.frame_skip = int(frame_skip)
        self.action_dt = self.frame_skip * DT
        self.random_start_offset = bool(random_start_offset)
        self.track_profile = str(track_profile)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(OBSERVATION_LAYOUT),),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng()
        self.track = Track.generate(
            self.fixed_seed if self.fixed_seed is not None else 0,
            self.difficulty,
            profile=self.track_profile,
        )
        self.car = Car(self.track.point_at(0.0), self.track.heading_at(0.0))
        self.sim = RaceSim(self.track, [self.car], dt=DT)
        self._elapsed_seconds = 0.0
        self._best_progress_s = 0.0
        self._last_forward_progress_time = 0.0
        self._last_info: dict[str, Any] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset onto a deterministic or sampled track and return the initial observation."""

        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        track_seed = self._choose_track_seed()
        self.track = Track.generate(track_seed, self.difficulty, profile=self.track_profile)
        use_offset = self.random_start_offset
        if options is not None and "random_start_offset" in options:
            use_offset = bool(options["random_start_offset"])
        start_s = float(self._rng.uniform(0.0, 0.05 * self.track.length)) if use_offset else 0.0
        self.car = Car(self.track.point_at(start_s), self.track.heading_at(start_s))
        self.sim = RaceSim(self.track, [self.car], dt=DT)
        self._elapsed_seconds = 0.0
        self._best_progress_s = self.sim.progress[0].progress_s
        self._last_forward_progress_time = 0.0

        obs = self._observation()
        info = self._info(decel_penalty=0.0)
        info["track_seed"] = track_seed
        self._last_info = info
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the environment by one action step with frame skipping."""

        steer, throttle = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        prev_speed = self.car.speed
        prev_progress_s = self.sim.progress[0].progress_s
        prev_laps = self.sim.progress[0].lap_count

        any_off_track = False
        for _ in range(self.frame_skip):
            telemetry = self.sim.step([(float(steer), float(throttle))])
            any_off_track = any_off_track or not bool(telemetry[0]["on_track"])

        self._elapsed_seconds += self.action_dt
        progress = self.sim.progress[0]
        delta_s = progress.progress_s - prev_progress_s
        lap_delta = progress.lap_count - prev_laps
        reward = shared_reward_core(delta_s, lap_delta, not any_off_track)

        decel_penalty = 0.0
        if self.reward_mode == "nobrakes":
            decel_penalty = self.decel_coef * max(0.0, prev_speed - self.car.speed)
            reward -= decel_penalty

        if progress.progress_s > self._best_progress_s + 1.0:
            self._best_progress_s = progress.progress_s
            self._last_forward_progress_time = self._elapsed_seconds

        terminated = progress.lap_count >= MAX_LAPS
        truncated = self._elapsed_seconds >= self.max_episode_seconds
        if self._elapsed_seconds - self._last_forward_progress_time >= STUCK_SECONDS:
            terminated = True
            reward -= STUCK_PENALTY

        obs = self._observation()
        info = self._info(decel_penalty=decel_penalty)
        self._last_info = info
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _choose_track_seed(self) -> int:
        if self.fixed_seed is not None:
            return int(self.fixed_seed)
        if self.seed_pool is None:
            return int(self._rng.integers(0, np.iinfo(np.int32).max))
        low, high = self.seed_pool
        return int(self._rng.integers(low, high))

    def _observation(self) -> np.ndarray:
        st = self.car.state
        progress = self.sim.progress[0]
        ray_angles = st.heading + np.deg2rad(np.asarray(RAYCAST_ANGLES_DEG, dtype=np.float64))
        raycasts = self.track.raycast_edges(st.position, ray_angles, RAYCAST_MAX_DIST) / RAYCAST_MAX_DIST

        track_heading = self.track.heading_at(progress.raw_s)
        heading_error = _wrap_angle(st.heading - track_heading) / math.pi
        curvatures = np.array(
            [self.track.curvature(progress.raw_s + offset) * 15.0 for offset in LOOKAHEAD_DISTANCES],
            dtype=np.float64,
        )
        obs = np.concatenate(
            (
                raycasts,
                np.array(
                    [
                        st.v_long / 40.0,
                        st.v_lat / 15.0,
                        heading_error,
                        progress.lateral_offset / self.track.half_width,
                        self.car.drift_angle / 1.5,
                        self.car.speed / 40.0,
                    ],
                    dtype=np.float64,
                ),
                curvatures,
            )
        )
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _info(self, decel_penalty: float) -> dict[str, Any]:
        progress = self.sim.progress[0]
        return {
            "speed": float(self.car.speed),
            "drift_angle": float(self.car.drift_angle),
            "is_drifting": bool(self.car.is_drifting),
            "on_track": bool(progress.on_track),
            "s": float(progress.progress_s),
            "lap_count": int(progress.lap_count),
            "lap_time": float(progress.lap_time),
            "decel_penalty": float(decel_penalty),
        }
