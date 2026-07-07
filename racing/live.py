"""Live and headless race modes for human and AI ghost comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np

from racing.env import RacingEnv
from racing.physics import DT
from racing.render import Renderer
from racing.sim import RaceSim
from racing.video import AGENT_COLORS, DEFAULT_SIZE, load_policies, render_overlay


HUMAN_COLOR = (245, 245, 245)


def cmd_watch(args: argparse.Namespace) -> int:
    """Run or record an AI-vs-AI ghost race."""

    policies = load_policies(args.policy, args.runs_dir)
    if args.headless:
        if args.record is None:
            raise ValueError("--headless requires --record")
        render_overlay(args.seed, args.record, policies, max_seconds=args.seconds, difficulty=args.difficulty)
        print(f"wrote {args.record}")
        return 0
    return _watch_window(args, policies)


def cmd_race(args: argparse.Namespace) -> int:
    """Run a human-vs-AI ghost race, or its headless self-test."""

    if args.selftest:
        _race_selftest(args)
        print("race selftest passed")
        return 0
    policies = load_policies(args.policy, args.runs_dir)
    return _race_window(args, policies[args.vs])


def add_watch_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the watch subcommand."""

    parser = subparsers.add_parser("watch", help="watch AI ghost cars race")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--policy", choices=("model", "heuristic"), default="model")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--record")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=5.0)


def add_race_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the race subcommand."""

    parser = subparsers.add_parser("race", help="race a human car against one AI ghost")
    parser.add_argument("--vs", choices=("time", "nobrakes"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--policy", choices=("model", "heuristic"), default="model")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)


def _watch_window(args: argparse.Namespace, policies: dict[str, object]) -> int:
    from racing.play import require_display

    if not require_display("watch"):
        return 2
    pygame = _pygame()
    pygame.init()
    envs = {
        agent: RacingEnv(
            reward_mode=agent,
            difficulty=args.difficulty,
            fixed_seed=args.seed,
            random_start_offset=False,
        )
        for agent in ("time", "nobrakes")
    }
    observations = {agent: envs[agent].reset(seed=args.seed, options={"random_start_offset": False})[0] for agent in envs}
    display_sim = RaceSim(envs["time"].track, [envs["time"].car, envs["nobrakes"].car])
    screen = pygame.display.set_mode(DEFAULT_SIZE)
    pygame.display.set_caption("AI ghost race")
    renderer = Renderer(display_sim, surface=screen, car_colors=AGENT_COLORS, car_labels=("time", "nobrakes"))
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        for agent in ("time", "nobrakes"):
            action, _ = policies[agent].predict(observations[agent], deterministic=True)  # type: ignore[attr-defined]
            observations[agent], _, _, _, _ = envs[agent].step(np.asarray(action, dtype=np.float32))
        display_sim.progress = [envs["time"].sim.progress[0], envs["nobrakes"].sim.progress[0]]
        display_sim.t = max(envs["time"].sim.t, envs["nobrakes"].sim.t)
        renderer.render_frame(screen)
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()
    return 0


def _race_window(args: argparse.Namespace, ai_policy: object) -> int:
    from racing.play import require_display

    if not require_display("race"):
        return 2
    pygame = _pygame()
    pygame.init()
    ai_env, human_sim, display_sim, ai_obs = _make_race_sims(args.seed, args.vs, args.difficulty)
    screen = pygame.display.set_mode(DEFAULT_SIZE)
    pygame.display.set_caption(f"Human vs {args.vs}")
    renderer = Renderer(display_sim, surface=screen, car_colors=(HUMAN_COLOR, AGENT_COLORS[0]), car_labels=("human", args.vs))
    clock = pygame.time.Clock()
    _countdown(screen, renderer, pygame)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    ai_env, human_sim, display_sim, ai_obs = _make_race_sims(args.seed, args.vs, args.difficulty)
                    renderer = Renderer(display_sim, surface=screen, car_colors=(HUMAN_COLOR, AGENT_COLORS[0]), car_labels=("human", args.vs))
                    _countdown(screen, renderer, pygame)
        keys = pygame.key.get_pressed()
        human_action = (float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT]), float(keys[pygame.K_UP]) - float(keys[pygame.K_DOWN]))
        ai_action, _ = ai_policy.predict(ai_obs, deterministic=True)  # type: ignore[attr-defined]
        human_sim.step([human_action])
        ai_obs, _, _, _, _ = ai_env.step(np.asarray(ai_action, dtype=np.float32))
        _sync_race_display(display_sim, human_sim, ai_env)
        renderer.render_frame(screen)
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()
    return 0


def _race_selftest(args: argparse.Namespace) -> None:
    policies = load_policies(args.policy, args.runs_dir)
    ai_env, human_sim, display_sim, ai_obs = _make_race_sims(args.seed, args.vs, args.difficulty)
    for step in range(int(5.0 / DT)):
        steer = float(np.sin(step * 0.03) * 0.35)
        throttle = 1.0 if step < 240 else 0.25
        ai_action, _ = policies[args.vs].predict(ai_obs, deterministic=True)
        human_sim.step([(steer, throttle)])
        ai_obs, _, _, _, _ = ai_env.step(np.asarray(ai_action, dtype=np.float32))
        _sync_race_display(display_sim, human_sim, ai_env)
    if len(display_sim.cars) != 2 or not np.isfinite(display_sim.cars[0].state.position).all():
        raise RuntimeError("race selftest produced invalid display state")


def record_race_selftest(path: str | Path, seed: int, vs: str, policy: str = "heuristic") -> Path:
    """Record a short scripted human-vs-AI clip for diagnostics."""

    policies = load_policies(policy)
    ai_env, human_sim, display_sim, ai_obs = _make_race_sims(seed, vs, 0.5)
    renderer = Renderer(display_sim, width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], car_colors=(HUMAN_COLOR, AGENT_COLORS[0]), car_labels=("human", vs))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out, fps=30, codec="libx264", quality=7, macro_block_size=1, ffmpeg_log_level="error") as writer:
        for step in range(150):
            ai_action, _ = policies[vs].predict(ai_obs, deterministic=True)
            human_sim.step([(float(np.sin(step * 0.04) * 0.25), 1.0)])
            ai_obs, _, _, _, _ = ai_env.step(np.asarray(ai_action, dtype=np.float32))
            _sync_race_display(display_sim, human_sim, ai_env)
            writer.append_data(renderer.to_rgb_array())
    return out


def _make_race_sims(seed: int, vs: str, difficulty: float) -> tuple[RacingEnv, RaceSim, RaceSim, np.ndarray]:
    ai_env = RacingEnv(reward_mode=vs, difficulty=difficulty, fixed_seed=seed, random_start_offset=False)
    ai_obs, _ = ai_env.reset(seed=seed, options={"random_start_offset": False})
    human_sim = RaceSim(ai_env.track)
    display_sim = RaceSim(ai_env.track, [human_sim.cars[0], ai_env.car])
    _sync_race_display(display_sim, human_sim, ai_env)
    return ai_env, human_sim, display_sim, ai_obs


def _sync_race_display(display_sim: RaceSim, human_sim: RaceSim, ai_env: RacingEnv) -> None:
    display_sim.progress = [human_sim.progress[0], ai_env.sim.progress[0]]
    display_sim.t = max(human_sim.t, ai_env.sim.t)


def _countdown(screen: object, renderer: Renderer, pygame: object) -> None:
    font = pygame.font.Font(None, 92)
    for label in ("3", "2", "1", "GO"):
        renderer.render_frame(screen)
        text = font.render(label, True, (245, 245, 245))
        rect = text.get_rect(center=(DEFAULT_SIZE[0] // 2, DEFAULT_SIZE[1] // 2))
        screen.blit(text, rect)
        pygame.display.flip()
        time.sleep(0.7)


def _pygame() -> object:
    import pygame

    return pygame
