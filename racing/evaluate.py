"""Build evaluation telemetry and report data for trained racing agents."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import date
import json
from pathlib import Path
from typing import Any, Protocol
import warnings

import numpy as np
import pandas as pd

from racing.agents import AGENT_SPECS, load_policy
from racing.env import DEFAULT_DECEL_COEF, RacingEnv
from racing.heuristic import heuristic_action


AGENT_ORDER = ("time", "nobrakes")
MAX_SPEED = 42.0
# Speed drop per 1/30s action step marking a hard brake; full braking produces
# ~1.6 u/s per step, coasting drag at top speed ~0.9, so 1.0 isolates real braking.
HARD_BRAKE_DECEL = 1.0


class Policy(Protocol):
    """Minimal policy interface shared by SB3 and heuristic drivers."""

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]:
        """Return an action for one observation."""


class HeuristicPolicy:
    """SB3-like wrapper around the observation-only heuristic driver."""

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        """Return the deterministic heuristic action."""

        del deterministic
        return heuristic_action(obs), None


def evaluate(
    runs_dir: str = "runs",
    out_dir: str = "dataset",
    eval_seeds: Iterable[int] = range(1000, 1016),
    difficulty: float = 0.5,
    policy: str = "model",
    track_profile: str = "default",
    decel_coef: float | None = None,
    archive_meta: dict[str, Any] | None = None,
    report_data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run held-out deterministic episodes and write telemetry/report artifacts."""

    seeds = [int(seed) for seed in eval_seeds]
    out_path = Path(out_dir)
    report_dir = Path(report_data_dir) if report_data_dir is not None else Path("report/assets/data")
    out_path.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    nobrakes_decel = float(decel_coef if decel_coef is not None else DEFAULT_DECEL_COEF)
    policies = _load_policies(policy, runs_dir)
    rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for agent in AGENT_ORDER:
            telemetry, episode = _run_episode(
                agent,
                seed,
                difficulty,
                policies[agent],
                track_profile=track_profile,
                decel_coef=nobrakes_decel,
            )
            rows.extend(telemetry)
            episode_rows.append(episode)

    telemetry_df = pd.DataFrame(rows)
    telemetry_df.to_csv(out_path / "telemetry.csv", index=False)

    episodes_df = pd.DataFrame(episode_rows)
    summary = _build_summary(
        telemetry_df,
        episodes_df,
        seeds,
        runs_dir,
        decel_coef=nobrakes_decel,
        archive_meta=archive_meta,
    )
    _write_json(out_path / "summary.json", summary)
    _write_json(report_dir / "summary.json", summary)
    _write_json(report_dir / "speed_histograms.json", _build_histograms(telemetry_df))
    _write_json(report_dir / "speed_profiles.json", _build_speed_profiles(telemetry_df))
    _write_json(report_dir / "learning_curves.json", _load_learning_curves(runs_dir))
    return telemetry_df


def cmd_evaluate(args: argparse.Namespace) -> int:
    """CLI entrypoint for dataset building."""

    seeds = parse_seeds(args.seeds)
    evaluate(
        runs_dir=args.runs_dir,
        out_dir=args.out_dir,
        eval_seeds=seeds,
        difficulty=args.difficulty,
        policy=args.policy,
    )
    print(f"wrote evaluation data for {len(seeds)} tracks to {args.out_dir}")
    return 0


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the evaluate subcommand."""

    parser = subparsers.add_parser("evaluate", help="build telemetry CSV and report JSON data")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default="dataset")
    parser.add_argument("--seeds", default="1000-1015", help="seed, comma list, or inclusive range")
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--policy", choices=("model", "heuristic"), default="model")


def parse_seeds(text: str) -> list[int]:
    """Parse ``1,2,5-7`` style seed specifications."""

    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _load_policies(policy: str, runs_dir: str) -> dict[str, Policy]:
    if policy == "heuristic":
        heuristic = HeuristicPolicy()
        return {agent: heuristic for agent in AGENT_ORDER}
    return {agent: load_policy(agent, runs_dir) for agent in AGENT_ORDER}


def _run_episode(
    agent: str,
    seed: int,
    difficulty: float,
    policy: Policy,
    *,
    track_profile: str = "default",
    decel_coef: float = DEFAULT_DECEL_COEF,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = RacingEnv(
        reward_mode=agent,
        decel_coef=decel_coef,
        difficulty=difficulty,
        fixed_seed=seed,
        max_episode_seconds=60.0,
        random_start_offset=False,
        track_profile=track_profile,
    )
    obs, info = env.reset(seed=seed, options={"random_start_offset": False})
    rows: list[dict[str, Any]] = []
    lap_time: float | None = None
    prev_speed = float(info["speed"])
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = policy.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        obs, _reward, terminated, truncated, info = env.step(action)
        decel = max(0.0, prev_speed - float(info["speed"]))
        prev_speed = float(info["speed"])
        car = env.car
        row = {
            "agent": agent,
            "track_seed": int(seed),
            "t": float(env.sim.t),
            "x": float(car.state.position[0]),
            "y": float(car.state.position[1]),
            "s": float(info["s"]),
            "speed": float(info["speed"]),
            "v_long": float(car.state.v_long),
            "v_lat": float(car.state.v_lat),
            "steer": float(action[0]),
            "throttle": float(action[1]),
            "drift_angle": float(info["drift_angle"]),
            "is_drifting": bool(info["is_drifting"]),
            "on_track": bool(info["on_track"]),
            "lap_time": float(info["lap_time"]),
            "decel": float(decel),
            "decel_penalty": float(info["decel_penalty"]),
            "lap_count": int(info["lap_count"]),
            "track_length": float(env.track.length),
        }
        rows.append(row)
        if int(info["lap_count"]) >= 1:
            lap_time = float(env.sim.progress[0].last_lap or env.sim.t)
            break
    return rows, {"agent": agent, "track_seed": int(seed), "lap_time": lap_time, "dnf": lap_time is None}


def _build_summary(
    telemetry: pd.DataFrame,
    episodes: pd.DataFrame,
    seeds: list[int],
    runs_dir: str,
    *,
    decel_coef: float = DEFAULT_DECEL_COEF,
    archive_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_steps = max((max(curve["steps"], default=0) for curve in _load_learning_curves(runs_dir).values()), default=0)
    # Seeds completed by each agent; lap-time means are computed only over seeds
    # BOTH agents completed so DNFs on hard tracks cannot skew the comparison.
    completed_seeds = {
        agent: set(episodes[(episodes["agent"] == agent) & (~episodes["dnf"])]["track_seed"].tolist())
        for agent in AGENT_ORDER
    }
    paired_seeds = set.intersection(*completed_seeds.values()) if completed_seeds else set()
    agents: dict[str, Any] = {}
    for agent in AGENT_ORDER:
        agent_rows = telemetry[telemetry["agent"] == agent]
        agent_eps = episodes[episodes["agent"] == agent]
        complete_laps = [float(v) for v in agent_eps["lap_time"].dropna().tolist()]
        paired_laps = [
            float(v)
            for v in agent_eps[agent_eps["track_seed"].isin(paired_seeds)]["lap_time"].dropna().tolist()
        ]
        # Hard-brake rate over completed episodes only (one lap each); counting
        # braking during DNF episodes against "per lap" would inflate the metric.
        brake_rows = agent_rows[agent_rows["track_seed"].isin(completed_seeds[agent])]
        n_completed = len(completed_seeds[agent])
        brake_events = sum(
            _count_events(group["decel"].to_numpy() > HARD_BRAKE_DECEL)
            for _, group in brake_rows.groupby("track_seed")
        )
        agents[agent] = {
            "label": AGENT_SPECS[agent]["label"],
            "color": AGENT_SPECS[agent]["color"],
            "mean_lap_time": _mean_or_none(paired_laps),
            "best_lap_time": min(complete_laps) if complete_laps else None,
            "mean_speed": _mean_or_none(agent_rows["speed"].tolist()),
            "std_speed": float(agent_rows["speed"].std(ddof=0)) if len(agent_rows) else None,
            "offtrack_fraction": float((~agent_rows["on_track"].astype(bool)).mean()) if len(agent_rows) else None,
            "drift_fraction": float(agent_rows["is_drifting"].astype(bool).mean()) if len(agent_rows) else None,
            "hard_brake_events_per_lap": float(brake_events / n_completed) if n_completed else None,
            "dnf_rate": float(agent_eps["dnf"].mean()) if len(agent_eps) else None,
        }

    per_track: list[dict[str, Any]] = []
    for seed in seeds:
        seed_eps = episodes[episodes["track_seed"] == seed].set_index("agent")
        seed_rows = telemetry[telemetry["track_seed"] == seed]
        per_track.append(
            {
                "seed": int(seed),
                "lap_time": {agent: _episode_lap_time(seed_eps, agent) for agent in AGENT_ORDER},
                "mean_speed": {
                    agent: _mean_or_none(seed_rows[seed_rows["agent"] == agent]["speed"].tolist()) for agent in AGENT_ORDER
                },
                "video": f"assets/videos/track_{seed}_overlay.mp4",
            }
        )

    experiment: dict[str, Any] = {
        "train_steps": int(train_steps),
        "n_eval_tracks": int(len(seeds)),
        "decel_penalty_coef": float(decel_coef),
        "algo": "PPO",
        "date": date.today().isoformat(),
    }
    if archive_meta:
        experiment.update(archive_meta)

    return {
        "experiment": experiment,
        "agents": agents,
        "per_track": per_track,
    }


def _build_histograms(telemetry: pd.DataFrame) -> dict[str, Any]:
    edges = np.linspace(0.0, MAX_SPEED, 41)
    output: dict[str, Any] = {"bin_edges": edges.round(6).tolist()}
    for agent in AGENT_ORDER:
        rows = telemetry[(telemetry["agent"] == agent) & (telemetry["s"] <= telemetry["track_length"])]
        hist, _ = np.histogram(rows["speed"].to_numpy(dtype=np.float64), bins=edges, density=True)
        output[agent] = np.nan_to_num(hist, nan=0.0).round(8).tolist()
    return output


def _build_speed_profiles(telemetry: pd.DataFrame) -> list[dict[str, Any]]:
    centers = (np.arange(200, dtype=np.float64) + 0.5) / 200.0
    profiles: list[dict[str, Any]] = []
    for seed in sorted(int(seed) for seed in telemetry["track_seed"].unique()):
        item: dict[str, Any] = {"seed": int(seed), "s": centers.round(6).tolist()}
        for agent in AGENT_ORDER:
            rows = telemetry[(telemetry["track_seed"] == seed) & (telemetry["agent"] == agent)].copy()
            if rows.empty:
                item[agent] = [None] * len(centers)
                continue
            s_norm = np.clip(rows["s"].to_numpy(dtype=np.float64) / rows["track_length"].to_numpy(dtype=np.float64), 0.0, 1.0)
            bins = np.minimum((s_norm * 200).astype(int), 199)
            speeds = rows["speed"].to_numpy(dtype=np.float64)
            values = np.full(200, np.nan, dtype=np.float64)
            for idx in range(200):
                mask = bins == idx
                if np.any(mask):
                    values[idx] = float(np.mean(speeds[mask]))
            item[agent] = _fill_profile(values).round(6).tolist()
        profiles.append(item)
    return profiles


def _load_learning_curves(runs_dir: str) -> dict[str, dict[str, list[float]]]:
    curves: dict[str, dict[str, list[float]]] = {}
    for agent in AGENT_ORDER:
        path = Path(runs_dir) / agent / "progress.csv"
        if not path.exists():
            warnings.warn(f"missing learning curve {path}", stacklevel=2)
            curves[agent] = {"steps": [], "reward": [], "ep_len": []}
            continue
        df = pd.read_csv(path)
        curves[agent] = {
            "steps": _series_or_empty(df, "time/total_timesteps", int),
            "reward": _series_or_empty(df, "rollout/ep_rew_mean", float),
            "ep_len": _series_or_empty(df, "rollout/ep_len_mean", float),
        }
    return curves


def _series_or_empty(df: pd.DataFrame, column: str, cast: type) -> list[Any]:
    if column not in df:
        return []
    return [cast(value) for value in df[column].dropna().tolist()]


def _fill_profile(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.zeros_like(values)
    x = np.arange(len(values), dtype=np.float64)
    return np.interp(x, x[valid], values[valid])


def _count_events(mask: np.ndarray) -> int:
    """Count runs of consecutive True values (one braking event each)."""

    if mask.size == 0:
        return 0
    return int(np.count_nonzero(mask & ~np.concatenate(([False], mask[:-1]))))


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _episode_lap_time(episodes: pd.DataFrame, agent: str) -> float | None:
    if agent not in episodes.index:
        return None
    value = episodes.loc[agent, "lap_time"]
    return None if pd.isna(value) else float(value)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
