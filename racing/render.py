"""Minimal pygame renderer for the racing simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from racing.sim import RaceSim


Color = tuple[int, int, int]
BACKGROUND: Color = (14, 17, 22)
TRACK_FILL: Color = (42, 47, 58)
TRACK_EDGE: Color = (88, 96, 113)
DEFAULT_COLORS: tuple[Color, ...] = ((230, 230, 230), (34, 211, 238), (251, 146, 60))


@dataclass
class Camera:
    """World-to-screen transform fitting the full track."""

    scale: float
    offset: np.ndarray
    height: int

    def world_to_screen(self, points: np.ndarray) -> np.ndarray:
        """Convert world points to pygame screen coordinates."""

        pts = np.asarray(points, dtype=np.float64)
        out = pts * self.scale + self.offset
        out[..., 1] = self.height - out[..., 1]
        return out


@dataclass
class Renderer:
    """Stateful renderer with car trails and fading skid marks."""

    sim: RaceSim
    width: int = 1100
    height: int = 750
    car_colors: Sequence[Color] = DEFAULT_COLORS
    car_labels: Sequence[str] | None = None
    surface: object | None = None
    camera: Camera = field(init=False)
    trails: list[list[np.ndarray]] = field(init=False)
    skid_marks: list[tuple[np.ndarray, Color, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.camera = self._make_camera()
        self.trails = [[] for _ in self.sim.cars]

    def _make_camera(self) -> Camera:
        points = np.vstack((self.sim.track.left_edge, self.sim.track.right_edge))
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        size = np.maximum(maxs - mins, 1.0)
        margin = 55.0
        scale = min((self.width - margin * 2.0) / size[0], (self.height - margin * 2.0) / size[1])
        center = (mins + maxs) * 0.5
        screen_center = np.array([self.width * 0.5, self.height * 0.5], dtype=np.float64)
        offset = screen_center - center * scale
        return Camera(float(scale), offset, self.height)

    def render_frame(self, surface: object | None = None, sim: RaceSim | None = None) -> object:
        """Draw one frame and return the pygame surface."""

        pygame = _pygame()
        if sim is not None and sim is not self.sim:
            self.sim = sim
            self.camera = self._make_camera()
            self.trails = [[] for _ in self.sim.cars]
        if surface is None:
            surface = self.surface
        if surface is None:
            surface = pygame.Surface((self.width, self.height))
            self.surface = surface

        surface.fill(BACKGROUND)
        self._draw_track(surface)
        self._draw_skids(surface)
        self._draw_cars(surface)
        self._draw_hud(surface)
        return surface

    def to_rgb_array(self) -> np.ndarray:
        """Return the current rendered frame as ``(height, width, 3)`` RGB uint8."""

        pygame = _pygame()
        surface = self.render_frame()
        arr = pygame.surfarray.array3d(surface)
        return np.transpose(arr, (1, 0, 2)).copy()

    def _draw_track(self, surface: object) -> None:
        pygame = _pygame()
        track = self.sim.track
        ribbon = np.vstack((track.left_edge, track.right_edge[::-1]))
        pygame.draw.polygon(surface, TRACK_FILL, self.camera.world_to_screen(ribbon).astype(int).tolist())
        pygame.draw.lines(surface, TRACK_EDGE, True, self.camera.world_to_screen(track.left_edge).astype(int).tolist(), 2)
        pygame.draw.lines(surface, TRACK_EDGE, True, self.camera.world_to_screen(track.right_edge).astype(int).tolist(), 2)
        self._draw_start_line(surface)

    def _draw_start_line(self, surface: object) -> None:
        pygame = _pygame()
        track = self.sim.track
        center = track.point_at(0.0)
        tangent = track.tangent_at(0.0)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        pieces = 9
        for i in range(pieces):
            a = -track.half_width + 2.0 * track.half_width * i / pieces
            b = -track.half_width + 2.0 * track.half_width * (i + 0.55) / pieces
            p0 = center + normal * a
            p1 = center + normal * b
            pygame.draw.line(
                surface,
                (230, 230, 230),
                self.camera.world_to_screen(p0).astype(int),
                self.camera.world_to_screen(p1).astype(int),
                3,
            )

    def _draw_skids(self, surface: object) -> None:
        pygame = _pygame()
        fresh: list[tuple[np.ndarray, Color, int]] = []
        for point, color, life in self.skid_marks:
            if life <= 0:
                continue
            alpha = life / 90.0
            draw_color = tuple(max(0, min(255, int(c * alpha))) for c in color)
            pygame.draw.circle(surface, draw_color, self.camera.world_to_screen(point).astype(int), 2)
            fresh.append((point, color, life - 1))
        self.skid_marks = fresh

    def _draw_cars(self, surface: object) -> None:
        pygame = _pygame()
        for idx, car in enumerate(self.sim.cars):
            color = self.car_colors[idx % len(self.car_colors)]
            pos = car.state.position.copy()
            self.trails[idx].append(pos)
            self.trails[idx] = self.trails[idx][-18:]
            trail = self.trails[idx]
            for i in range(1, len(trail)):
                fade = i / len(trail)
                trail_color = tuple(int(c * 0.35 * fade) for c in color)
                pygame.draw.line(
                    surface,
                    trail_color,
                    self.camera.world_to_screen(trail[i - 1]).astype(int),
                    self.camera.world_to_screen(trail[i]).astype(int),
                    max(1, int(3 * fade)),
                )
            if car.is_drifting:
                self.skid_marks.append((pos.copy(), (28, 30, 35), 90))
            self._draw_car(surface, pos, car.state.heading, color)

    def _draw_car(self, surface: object, position: np.ndarray, heading: float, color: Color) -> None:
        pygame = _pygame()
        forward = np.array([math.cos(heading), math.sin(heading)], dtype=np.float64)
        right = np.array([-math.sin(heading), math.cos(heading)], dtype=np.float64)
        pts = np.array(
            [
                position + forward * 10.5,
                position - forward * 7.5 + right * 5.0,
                position - forward * 4.0,
                position - forward * 7.5 - right * 5.0,
            ]
        )
        pygame.draw.polygon(surface, color, self.camera.world_to_screen(pts).astype(int).tolist())
        nose = np.array([position, position + forward * 8.5])
        pygame.draw.line(surface, (10, 12, 16), *self.camera.world_to_screen(nose).astype(int), 2)

    def _draw_hud(self, surface: object) -> None:
        pygame = _pygame()
        pygame.font.init()
        font = pygame.font.Font(None, 22)
        small = pygame.font.Font(None, 19)
        x, y = 16, 14
        if not self.sim.cars:
            return
        car = self.sim.cars[0]
        speed = car.speed
        pygame.draw.rect(surface, (28, 33, 42), (x, y, 190, 18), border_radius=4)
        pygame.draw.rect(surface, (34, 211, 238), (x, y, min(190, int(speed / 42.0 * 190)), 18), border_radius=4)
        lines = [f"lead speed {speed:5.1f} u/s"]
        for line in lines:
            text = font.render(line, True, (226, 232, 240))
            surface.blit(text, (x, y + 24))
            y += 22
        y += 4
        for i, _car in enumerate(self.sim.cars):
            color = self.car_colors[i % len(self.car_colors)]
            pygame.draw.rect(surface, color, (x, y + 4, 12, 8), border_radius=2)
            if self.car_labels is not None and i < len(self.car_labels):
                label = self.car_labels[i]
            else:
                label = ["human", "time", "nobrakes"][i] if i < 3 else f"car {i}"
            prog = self.sim.progress[i]
            last = f" last {prog.last_lap:5.2f}" if prog.last_lap is not None else ""
            text = f"{label} L{prog.lap_count} {prog.lap_time:5.2f}{last}"
            surface.blit(small.render(text, True, (203, 213, 225)), (x + 18, y))
            y += 18


def render_frame(surface: object, sim: RaceSim) -> object:
    """Draw one stateless frame on ``surface`` for ``sim``."""

    width, height = surface.get_size()
    renderer = Renderer(sim, width=width, height=height, surface=surface)
    return renderer.render_frame(surface, sim)


def _pygame() -> object:
    import pygame

    return pygame
