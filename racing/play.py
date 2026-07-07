"""Command line entry points for previewing, playing, and self-testing."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Callable

import numpy as np

from racing.physics import Car
from racing.render import Renderer
from racing.sim import RaceSim
from racing.track import Track


Command = Callable[[argparse.Namespace], int]


def cmd_newtrack(args: argparse.Namespace) -> int:
    """Render a deterministic track preview image."""

    track = Track.generate(args.seed, args.difficulty)
    sim = RaceSim(track)
    pygame = _pygame()
    pygame.init()
    out_dir = Path("runs/previews")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"track_{args.seed}.png"
    surface = pygame.Surface((1100, 750))
    Renderer(sim, surface=surface).render_frame(surface)
    pygame.image.save(surface, out_path)
    print(f"saved {out_path}")

    has_display = bool(os.environ.get("DISPLAY")) and os.environ.get("SDL_VIDEODRIVER") != "dummy"
    if has_display:
        screen = pygame.display.set_mode((1100, 750))
        pygame.display.set_caption(f"racing track seed {args.seed}")
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        start = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start < 2500:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return 0
            pygame.time.wait(16)
    pygame.quit()
    return 0


def require_display(mode_name: str) -> bool:
    """Check for a usable display; print a helpful message when absent."""

    has_display = (
        os.environ.get("SDL_VIDEODRIVER") != "dummy"
        and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    )
    if not has_display:
        print(
            f"`{mode_name}` opens a game window and needs a desktop session; "
            "no display was found. For a recording instead, use: "
            "python -m racing watch --seed N --headless --record out.mp4"
        )
    return has_display


def cmd_play(args: argparse.Namespace) -> int:
    """Play the track with arrow-key controls."""

    if not require_display("play"):
        return 2
    pygame = _pygame()
    pygame.init()
    sim = RaceSim.from_seed(args.seed, args.difficulty)
    screen = pygame.display.set_mode((1100, 750))
    pygame.display.set_caption("racing AI experiment")
    clock = pygame.time.Clock()
    renderer = Renderer(sim, surface=screen)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    sim.reset()
        keys = pygame.key.get_pressed()
        steer = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
        throttle = float(keys[pygame.K_UP]) - float(keys[pygame.K_DOWN])
        sim.step([(steer, throttle)])
        renderer.render_frame(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
    return 0


def cmd_selftest(_args: argparse.Namespace) -> int:
    """Run deterministic headless sanity checks."""

    checks: list[tuple[str, Callable[[], bool]]] = [
        ("generate 20 valid tracks", _check_tracks),
        ("full-throttle straight reaches >30 u/s", _check_straight_speed),
        ("high-speed cornering drifts", _check_drift),
        ("off-track top speed penalty", _check_offtrack_penalty),
        ("localize round-trip accuracy", _check_localize_roundtrip),
        ("raycasts return finite distances", _check_raycasts),
    ]
    ok = True
    for name, fn in checks:
        try:
            passed = bool(fn())
        except Exception as exc:  # noqa: BLE001 - selftest should report failures compactly.
            passed = False
            print(f"FAIL {name}: {exc}")
        else:
            print(f"{'PASS' if passed else 'FAIL'} {name}")
        ok = ok and passed
    return 0 if ok else 1


def cmd_train(args: argparse.Namespace) -> int:
    """Train one or both PPO reward variants."""

    from racing.train import train

    reward_modes = ["time", "nobrakes"] if args.agent == "both" else [args.agent]
    for reward_mode in reward_modes:
        print(f"training {reward_mode} for {args.steps} steps with {args.envs} envs")
        train(reward_mode, total_steps=args.steps, n_envs=args.envs, seed=args.seed, device=args.device)
    return 0


def cmd_archive_report(args: argparse.Namespace) -> int:
    """Build reports for archived training runs."""

    from racing.archive_report import cmd_archive_report as _cmd

    return _cmd(args)


def cmd_evolution_video(args: argparse.Namespace) -> int:
    from racing.evolution_video import cmd_evolution_video as _cmd

    return _cmd(args)


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Build evaluation telemetry and report JSON assets."""

    from racing.evaluate import cmd_evaluate as _cmd_evaluate

    return _cmd_evaluate(args)


def cmd_video(args: argparse.Namespace) -> int:
    """Render report videos."""

    from racing.video import cmd_video as _cmd_video

    return _cmd_video(args)


def cmd_report(args: argparse.Namespace) -> int:
    """Validate and serve the static report."""

    from racing.report_cmd import cmd_report as _cmd_report

    return _cmd_report(args)


def cmd_watch(args: argparse.Namespace) -> int:
    """Watch AI ghost cars race."""

    from racing.live import cmd_watch as _cmd_watch

    return _cmd_watch(args)


def cmd_race(args: argparse.Namespace) -> int:
    """Race a human car against one AI ghost."""

    from racing.live import cmd_race as _cmd_race

    return _cmd_race(args)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser with easily extensible subcommands."""

    parser = argparse.ArgumentParser(prog="python -m racing")
    subparsers = parser.add_subparsers(dest="cmd")
    newtrack = subparsers.add_parser("newtrack", help="render a deterministic track preview")
    newtrack.add_argument("--seed", type=int, required=True)
    newtrack.add_argument("--difficulty", type=float, default=0.5)
    play = subparsers.add_parser("play", help="drive a track with arrow keys")
    play.add_argument("--seed", type=int, required=True)
    play.add_argument("--difficulty", type=float, default=0.5)
    subparsers.add_parser("selftest", help="run headless sanity checks")
    train_parser = subparsers.add_parser("train", help="train PPO racing agents")
    train_parser.add_argument("--agent", choices=("time", "nobrakes", "both"), required=True)
    train_parser.add_argument("--steps", type=int, default=1_500_000)
    train_parser.add_argument("--envs", type=int, default=12)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default="auto")
    from racing.archive_report import add_parser as add_archive_parser
    from racing.evaluate import add_parser as add_evaluate_parser
    from racing.evolution_video import add_parser as add_evolution_parser
    from racing.live import add_race_parser, add_watch_parser
    from racing.report_cmd import add_parser as add_report_parser
    from racing.video import add_parser as add_video_parser

    add_evaluate_parser(subparsers)
    add_archive_parser(subparsers)
    add_evolution_parser(subparsers)
    add_video_parser(subparsers)
    add_report_parser(subparsers)
    add_watch_parser(subparsers)
    add_race_parser(subparsers)
    return parser


COMMANDS: dict[str, Command] = {
    "newtrack": cmd_newtrack,
    "play": cmd_play,
    "selftest": cmd_selftest,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "archive-report": cmd_archive_report,
    "evolution-video": cmd_evolution_video,
    "video": cmd_video,
    "report": cmd_report,
    "watch": cmd_watch,
    "race": cmd_race,
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch the racing CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    command = COMMANDS.get(str(args.cmd))
    if command is None:
        parser.print_help()
        return 2
    return command(args)


def _check_tracks() -> bool:
    return all(Track.generate(seed)._is_valid() for seed in range(20))


def _check_straight_speed() -> bool:
    car = Car(np.zeros(2), 0.0)
    for _ in range(6 * 60):
        car.step(0.0, 1.0, True)
    return car.speed > 30.0


def _check_drift() -> bool:
    car = Car(np.zeros(2), 0.0)
    for _ in range(4 * 60):
        car.step(0.0, 1.0, True)
    max_drift = 0.0
    for _ in range(3 * 60):
        car.step(1.0, 1.0, True)
        max_drift = max(max_drift, car.drift_angle)
    return max_drift > 0.2


def _terminal_speed(on_track: bool) -> float:
    car = Car(np.zeros(2), 0.0)
    for _ in range(12 * 60):
        car.step(0.0, 1.0, on_track)
    return car.speed


def _check_offtrack_penalty() -> bool:
    on_track_speed = _terminal_speed(True)
    off_track_speed = _terminal_speed(False)
    return off_track_speed < 0.45 * on_track_speed


def _check_localize_roundtrip() -> bool:
    track = Track.generate(7)
    rng = np.random.default_rng(12345)
    samples = rng.uniform(0.0, track.length, size=100)
    errors = []
    for s in samples:
        localized, lateral = track.localize(track.point_at(float(s)))
        err = abs(((localized - s + track.length * 0.5) % track.length) - track.length * 0.5)
        errors.append(err)
        if abs(lateral) > 0.05:
            return False
    return max(errors) < 1.0


def _check_raycasts() -> bool:
    track = Track.generate(7)
    origin = track.point_at(100.0)
    heading = track.heading_at(100.0)
    angles = heading + np.linspace(-math.pi * 0.7, math.pi * 0.7, 9)
    distances = track.raycast_edges(origin, angles, 120.0)
    return bool(np.all(np.isfinite(distances)) and np.all(distances > 0.0) and np.any(distances < 60.0))


def _pygame() -> object:
    import pygame

    return pygame


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
