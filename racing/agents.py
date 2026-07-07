"""Stable-Baselines3 policy helpers for trained racing agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from racing.env import RacingEnv


AGENT_SPECS = {
    "time": {"label": "Time Optimizer", "color": "#22d3ee"},
    "nobrakes": {"label": "No-Brakes Optimizer", "color": "#fb923c"},
}


def make_env(reward_mode: str, **kwargs: Any) -> Callable[[], gym.Env]:
    """Return a picklable Monitor-wrapped RacingEnv factory for vector training."""

    def _init() -> gym.Env:
        return Monitor(RacingEnv(reward_mode=reward_mode, **kwargs))

    return _init


def load_policy(reward_mode: str, runs_dir: str = "runs") -> PPO:
    """Load a trained PPO policy from ``runs/<reward_mode>/model.zip``."""

    if reward_mode not in AGENT_SPECS:
        raise ValueError(f"unknown reward_mode {reward_mode!r}")
    model_path = Path(runs_dir) / reward_mode / "model.zip"
    if not model_path.exists():
        raise SystemExit(
            f"no trained model at {model_path} - run "
            f"`python -m racing train --agent {reward_mode}` first, or pass --policy heuristic"
        )
    return PPO.load(model_path)
