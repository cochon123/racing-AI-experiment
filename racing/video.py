"""Headless video rendering for racing policy comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from racing.agents import AGENT_SPECS, load_policy
from racing.env import RacingEnv
from racing.evaluate import AGENT_ORDER, HeuristicPolicy, Policy, parse_seeds
from racing.render import Renderer
from racing.sim import RaceSim


DEFAULT_SIZE = (1100, 750)


def render_overlay(
    seed: int,
    out_path: str | Path,
    policies: dict[str, Policy],
    max_seconds: float = 60.0,
    fps: int = 30,
    difficulty: float = 0.5,
    track_profile: str = "default",
    decel_coef: float = 1.0,
) -> Path:
    """Render both agents as ghost cars on the same fixed track."""

    envs = {
        agent: _make_eval_env(agent, seed, difficulty, max_seconds, track_profile, decel_coef)
        for agent in AGENT_ORDER
    }
    _check_frame_pacing(envs["time"], fps)
    observations = {agent: envs[agent].reset(seed=seed, options={"random_start_offset": False})[0] for agent in AGENT_ORDER}
    display_sim = RaceSim(envs["time"].track, [envs[agent].car for agent in AGENT_ORDER])
    renderer = Renderer(display_sim, width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], car_colors=AGENT_COLORS, car_labels=AGENT_ORDER)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(
        out,
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=1,
        ffmpeg_log_level="error",
    ) as writer:
        for _ in range(int(max_seconds * fps)):
            all_finished = True
            for agent in AGENT_ORDER:
                if envs[agent].sim.progress[0].lap_count >= 1:
                    continue
                all_finished = False
                action, _ = policies[agent].predict(observations[agent], deterministic=True)
                observations[agent], _, _, _, _ = envs[agent].step(np.asarray(action, dtype=np.float32))
            _sync_display_sim(display_sim, envs)
            writer.append_data(renderer.to_rgb_array())
            if all_finished:
                break
    return out


def render_solo(
    seed: int,
    mode: str,
    out_path: str | Path,
    policies: dict[str, Policy],
    max_seconds: float = 60.0,
    fps: int = 30,
    difficulty: float = 0.5,
    track_profile: str = "default",
    decel_coef: float = 1.0,
) -> Path:
    """Render a single policy on a fixed track."""

    if mode not in AGENT_SPECS:
        raise ValueError("mode must be 'time' or 'nobrakes'")
    env = _make_eval_env(mode, seed, difficulty, max_seconds, track_profile, decel_coef)
    _check_frame_pacing(env, fps)
    obs, _ = env.reset(seed=seed, options={"random_start_offset": False})
    renderer = Renderer(env.sim, width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], car_colors=(_hex_to_rgb(AGENT_SPECS[mode]["color"]),), car_labels=(mode,))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(
        out,
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=1,
        ffmpeg_log_level="error",
    ) as writer:
        for _ in range(int(max_seconds * fps)):
            if env.sim.progress[0].lap_count >= 1:
                break
            action, _ = policies[mode].predict(obs, deterministic=True)
            obs, _, _, _, _ = env.step(np.asarray(action, dtype=np.float32))
            writer.append_data(renderer.to_rgb_array())
    return out


def render_intro(
    out_path: str | Path,
    policies: dict[str, Policy],
    *,
    track_profile: str = "default",
    decel_coef: float = 1.0,
) -> Path:
    """Render the short hero overlay video for the report."""

    return render_overlay(1003, out_path, policies, max_seconds=25.0, track_profile=track_profile, decel_coef=decel_coef)


def cmd_video(args: argparse.Namespace) -> int:
    """CLI entrypoint for report video rendering."""

    policies = load_policies(args.policy, args.runs_dir)
    out_dir = Path(args.out_dir)
    render_intro(out_dir / "intro_overlay.mp4", policies)
    for seed in parse_seeds(args.seeds):
        render_overlay(seed, out_dir / f"track_{seed}_overlay.mp4", policies, difficulty=args.difficulty)
    print(f"wrote videos to {out_dir}")
    return 0


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the video subcommand."""

    parser = subparsers.add_parser("video", help="render report replay videos")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default="report/assets/videos")
    parser.add_argument("--seeds", default="1000-1015", help="seed, comma list, or inclusive range")
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--policy", choices=("model", "heuristic"), default="model")


def load_policies(policy: str, runs_dir: str = "runs") -> dict[str, Policy]:
    """Load model or heuristic policies for both agents."""

    if policy == "heuristic":
        heuristic = HeuristicPolicy()
        return {agent: heuristic for agent in AGENT_ORDER}
    return {agent: load_policy(agent, runs_dir) for agent in AGENT_ORDER}


def _make_eval_env(
    agent: str,
    seed: int,
    difficulty: float,
    max_seconds: float,
    track_profile: str = "default",
    decel_coef: float = 1.0,
) -> RacingEnv:
    return RacingEnv(
        reward_mode=agent,
        decel_coef=decel_coef,
        difficulty=difficulty,
        fixed_seed=int(seed),
        max_episode_seconds=float(max_seconds),
        random_start_offset=False,
        track_profile=track_profile,
    )


def _check_frame_pacing(env: RacingEnv, fps: int) -> None:
    """One video frame is written per env step, so sim time per step must be 1/fps."""

    if abs(env.action_dt - 1.0 / fps) > 1e-9:
        raise ValueError(
            f"fps={fps} does not match env action_dt={env.action_dt:.4f}s "
            "(frame_skip * physics dt); video would play at the wrong speed"
        )


def _sync_display_sim(display_sim: RaceSim, envs: dict[str, RacingEnv]) -> None:
    display_sim.progress = [envs[agent].sim.progress[0] for agent in AGENT_ORDER]
    display_sim.last_telemetry = [envs[agent].sim.last_telemetry[0] for agent in AGENT_ORDER if envs[agent].sim.last_telemetry]
    display_sim.t = max(envs[agent].sim.t for agent in AGENT_ORDER)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.removeprefix("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


AGENT_COLORS = tuple(_hex_to_rgb(AGENT_SPECS[agent]["color"]) for agent in AGENT_ORDER)
