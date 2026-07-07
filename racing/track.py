"""Procedural closed-loop track generation and geometry queries."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np


Array = np.ndarray


def _normalize(v: Array) -> Array:
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norms, 1e-12)


def _cross2(a: Array, b: Array) -> Array:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _catmull_rom_closed(points: Array, samples_per_segment: int = 24) -> Array:
    samples: list[Array] = []
    n = len(points)
    t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
    t2 = t * t
    t3 = t2 * t
    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]
        seg = 0.5 * (
            2.0 * p1
            + np.outer(t, -p0 + p2)
            + np.outer(t2, 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)
            + np.outer(t3, -p0 + 3.0 * p1 - 3.0 * p2 + p3)
        )
        samples.append(seg)
    return np.vstack(samples)


def _resample_closed(polyline: Array, spacing: float = 2.0) -> tuple[Array, Array, float]:
    seg = np.roll(polyline, -1, axis=0) - polyline
    seg_len = np.linalg.norm(seg, axis=1)
    keep = seg_len > 1e-6
    polyline = polyline[keep]
    seg = np.roll(polyline, -1, axis=0) - polyline
    seg_len = np.linalg.norm(seg, axis=1)
    length = float(seg_len.sum())
    count = max(64, int(round(length / spacing)))
    s_new = np.linspace(0.0, length, count, endpoint=False)
    cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))
    idx = np.searchsorted(cumulative, s_new, side="right") - 1
    idx = np.clip(idx, 0, len(polyline) - 1)
    local = (s_new - cumulative[idx]) / np.maximum(seg_len[idx], 1e-12)
    next_idx = (idx + 1) % len(polyline)
    resampled = polyline[idx] + (polyline[next_idx] - polyline[idx]) * local[:, None]
    return resampled.astype(np.float64), s_new.astype(np.float64), length


def _segments_intersect(a: Array, b: Array, c: Array, d: Array) -> bool:
    r = b - a
    s = d - c
    denom = float(_cross2(r, s))
    qp = c - a
    if abs(denom) < 1e-10:
        if abs(float(_cross2(qp, r))) > 1e-8:
            return False
        rr = float(np.dot(r, r))
        if rr < 1e-12:
            return False
        t0 = float(np.dot(c - a, r) / rr)
        t1 = float(np.dot(d - a, r) / rr)
        return max(min(t0, t1), 0.0) <= min(max(t0, t1), 1.0)
    t = float(_cross2(qp, s) / denom)
    u = float(_cross2(qp, r) / denom)
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


def _has_self_intersection(polyline: Array) -> bool:
    n = len(polyline)
    a_points = polyline
    b_points = np.roll(polyline, -1, axis=0)
    seg_len = np.linalg.norm(b_points - a_points, axis=1)
    cell_size = max(8.0, float(np.percentile(seg_len, 90) * 4.0))
    grid: dict[tuple[int, int], list[int]] = {}
    checked: set[tuple[int, int]] = set()

    for i in range(n):
        a = a_points[i]
        b = b_points[i]
        lo = np.floor(np.minimum(a, b) / cell_size).astype(int)
        hi = np.floor(np.maximum(a, b) / cell_size).astype(int)
        candidate_cells = [
            (gx, gy)
            for gx in range(int(lo[0]), int(hi[0]) + 1)
            for gy in range(int(lo[1]), int(hi[1]) + 1)
        ]
        for cell in candidate_cells:
            for j in grid.get(cell, ()):
                if abs(i - j) <= 1 or (i == n - 1 and j == 0):
                    continue
                pair = (j, i) if j < i else (i, j)
                if pair in checked:
                    continue
                checked.add(pair)
                if _segments_intersect(a, b, a_points[j], b_points[j]):
                    return True
        for cell in candidate_cells:
            grid.setdefault(cell, []).append(i)
    return False


@dataclass(slots=True)
class Track:
    """Closed racing track with precomputed geometry acceleration structures."""

    centerline: Array
    half_width: float
    length: float
    s_values: Array
    tangents: Array
    left_edge: Array
    right_edge: Array
    _curvature_values: Array = field(repr=False)
    _segment_lengths: Array = field(repr=False)
    _center_grid: dict[tuple[int, int], list[int]] = field(repr=False)
    _edge_grid: dict[tuple[int, int], list[int]] = field(repr=False)
    _edge_a: Array = field(repr=False)
    _edge_b: Array = field(repr=False)
    _grid_cell_size: float = field(default=24.0, repr=False)

    @classmethod
    def generate(cls, seed: int, difficulty: float = 0.5, profile: str = "default") -> "Track":
        """Generate a deterministic closed-loop track for ``seed``."""

        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        rng = np.random.default_rng(int(seed))
        gentle = profile == "gentle"
        for _ in range(150):
            control_count = int(rng.integers(10, 15))
            base_angles = np.linspace(0.0, 2.0 * math.pi, control_count, endpoint=False)
            angle_step = 2.0 * math.pi / control_count
            angle_jitter = 0.18 if gentle else 0.35
            angles = base_angles + rng.uniform(
                -angle_jitter * angle_step * (0.25 + difficulty),
                angle_jitter * angle_step * (0.25 + difficulty),
                size=control_count,
            )
            angles.sort()
            if gentle:
                base_radius = float(rng.uniform(220.0, 320.0))
                radial_jitter = rng.normal(0.0, 18.0 + 28.0 * difficulty, size=control_count)
                radii = np.clip(base_radius + radial_jitter, 180.0, 370.0)
            else:
                base_radius = float(rng.uniform(190.0, 280.0))
                radial_jitter = rng.normal(0.0, 20.0 + 90.0 * difficulty, size=control_count)
                radii = np.clip(base_radius + radial_jitter, 110.0, 400.0)
            controls = np.column_stack((np.cos(angles) * radii, np.sin(angles) * radii))
            smooth = _catmull_rom_closed(controls)
            centerline, s_values, length = _resample_closed(smooth, 2.0)
            half_width = float(18.0 - 5.0 * difficulty + rng.uniform(-0.7, 0.7))
            track = cls._from_centerline(centerline, s_values, length, half_width)
            if track._is_valid(profile=profile):
                return track
        raise RuntimeError(f"failed to generate a valid track for seed {seed}")

    @classmethod
    def _from_centerline(
        cls, centerline: Array, s_values: Array, length: float, half_width: float
    ) -> "Track":
        next_points = np.roll(centerline, -1, axis=0)
        segment_vecs = next_points - centerline
        segment_lengths = np.linalg.norm(segment_vecs, axis=1)
        tangents = _normalize(np.roll(centerline, -1, axis=0) - np.roll(centerline, 1, axis=0))
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
        left_edge = centerline + normals * half_width
        right_edge = centerline - normals * half_width
        headings = np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0]))
        heading_delta = np.roll(headings, -1) - headings
        heading_delta = (heading_delta + math.pi) % (2.0 * math.pi) - math.pi
        curvature = heading_delta / np.maximum(segment_lengths, 1e-6)

        cell_size = max(half_width * 2.2, 24.0)
        center_grid = cls._build_segment_grid(centerline, next_points, cell_size)
        edge_a = np.vstack((left_edge, right_edge))
        edge_b = np.vstack((np.roll(left_edge, -1, axis=0), np.roll(right_edge, -1, axis=0)))
        edge_grid = cls._build_segment_grid(edge_a, edge_b, cell_size)
        return cls(
            centerline=centerline,
            half_width=half_width,
            length=length,
            s_values=s_values,
            tangents=tangents,
            left_edge=left_edge,
            right_edge=right_edge,
            _curvature_values=curvature,
            _segment_lengths=segment_lengths,
            _center_grid=center_grid,
            _edge_grid=edge_grid,
            _edge_a=edge_a,
            _edge_b=edge_b,
            _grid_cell_size=cell_size,
        )

    @staticmethod
    def _build_segment_grid(a_points: Array, b_points: Array, cell_size: float) -> dict[tuple[int, int], list[int]]:
        grid: dict[tuple[int, int], list[int]] = {}
        mins = np.minimum(a_points, b_points)
        maxs = np.maximum(a_points, b_points)
        lo = np.floor(mins / cell_size).astype(int)
        hi = np.floor(maxs / cell_size).astype(int)
        for idx, (lo_xy, hi_xy) in enumerate(zip(lo, hi, strict=True)):
            for gx in range(int(lo_xy[0]), int(hi_xy[0]) + 1):
                for gy in range(int(lo_xy[1]), int(hi_xy[1]) + 1):
                    grid.setdefault((gx, gy), []).append(idx)
        return grid

    def _is_valid(self, profile: str = "default") -> bool:
        if _has_self_intersection(self.centerline):
            return False
        if _has_self_intersection(self.left_edge) or _has_self_intersection(self.right_edge):
            return False
        max_curvature = float(np.max(np.abs(self._curvature_values)))
        min_radius = 1.0 / max(max_curvature, 1e-9)
        if profile == "gentle":
            return min_radius > self.half_width * 1.45 and self.length > 1200.0
        return min_radius > self.half_width * 1.25 and self.length > 1000.0

    def curvature(self, s: float) -> float:
        """Return linearly interpolated signed centerline curvature at arc length ``s``."""

        return float(self._interp_scalar(self._curvature_values, s))

    def point_at(self, s: float) -> Array:
        """Return the centerline point at wrapped arc length ``s``."""

        idx, u = self._segment_at(s)
        return self.centerline[idx] + (self.centerline[(idx + 1) % len(self.centerline)] - self.centerline[idx]) * u

    def heading_at(self, s: float) -> float:
        """Return the centerline heading angle in radians at wrapped arc length ``s``."""

        tangent = self.tangent_at(s)
        return float(math.atan2(float(tangent[1]), float(tangent[0])))

    def tangent_at(self, s: float) -> Array:
        """Return the unit tangent at wrapped arc length ``s``."""

        idx, u = self._segment_at(s)
        t0 = self.tangents[idx]
        t1 = self.tangents[(idx + 1) % len(self.tangents)]
        return _normalize((t0 * (1.0 - u) + t1 * u)[None, :])[0]

    def localize(self, point: Array) -> tuple[float, float]:
        """Return ``(s, lateral_offset)`` for the nearest centerline segment."""

        p = np.asarray(point, dtype=np.float64)
        candidates = self._nearby_segments(p, self._center_grid, radius_cells=1)
        if not candidates:
            candidates = self._nearby_segments(p, self._center_grid, radius_cells=3)
        if not candidates:
            candidates = range(len(self.centerline))

        idxs = np.fromiter(candidates, dtype=np.int64)
        a = self.centerline[idxs]
        b = self.centerline[(idxs + 1) % len(self.centerline)]
        ab = b - a
        ab_len2 = np.einsum("ij,ij->i", ab, ab)
        u = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(ab_len2, 1e-12), 0.0, 1.0)
        proj = a + ab * u[:, None]
        dist2 = np.einsum("ij,ij->i", p - proj, p - proj)
        best = int(np.argmin(dist2))
        seg_idx = int(idxs[best])
        tangent = self.tangents[seg_idx]
        lateral = float(_cross2(tangent, p - proj[best]))
        s = (float(self.s_values[seg_idx]) + float(u[best] * self._segment_lengths[seg_idx])) % self.length
        return s, lateral

    def on_track(self, point: Array) -> bool:
        """Return whether ``point`` lies inside the track ribbon."""

        _, lateral = self.localize(point)
        return abs(lateral) <= self.half_width

    def raycast_edges(self, origin: Array, angles_world: Iterable[float] | Array, max_dist: float) -> Array:
        """Return distance to the first track edge crossing for each world-space ray angle."""

        p = np.asarray(origin, dtype=np.float64)
        angles = np.asarray(list(angles_world), dtype=np.float64)
        dirs = np.column_stack((np.cos(angles), np.sin(angles)))
        candidates = self._edge_candidates_in_radius(p, float(max_dist))
        if not candidates:
            return np.full(len(angles), float(max_dist), dtype=np.float64)

        idxs = np.fromiter(candidates, dtype=np.int64)
        a = self._edge_a[idxs]
        b = self._edge_b[idxs]
        seg = b - a
        out = np.full(len(angles), float(max_dist), dtype=np.float64)
        qmp = a - p
        for ray_i, direction in enumerate(dirs):
            denom = _cross2(direction, seg)
            valid = np.abs(denom) > 1e-10
            if not np.any(valid):
                continue
            t = _cross2(qmp, seg) / denom
            u = _cross2(qmp, direction) / denom
            hits = valid & (t >= 0.0) & (t <= max_dist) & (u >= 0.0) & (u <= 1.0)
            if np.any(hits):
                out[ray_i] = float(np.min(t[hits]))
        return out

    def _segment_at(self, s: float) -> tuple[int, float]:
        s_wrapped = float(s) % self.length
        idx = int(np.searchsorted(self.s_values, s_wrapped, side="right") - 1)
        idx = max(0, min(idx, len(self.centerline) - 1))
        u = (s_wrapped - float(self.s_values[idx])) / max(float(self._segment_lengths[idx]), 1e-12)
        return idx, float(np.clip(u, 0.0, 1.0))

    def _interp_scalar(self, values: Array, s: float) -> float:
        idx, u = self._segment_at(s)
        return float(values[idx] * (1.0 - u) + values[(idx + 1) % len(values)] * u)

    def _cell(self, point: Array) -> tuple[int, int]:
        cell = np.floor(point / self._grid_cell_size).astype(int)
        return int(cell[0]), int(cell[1])

    def _nearby_segments(
        self, point: Array, grid: dict[tuple[int, int], list[int]], radius_cells: int
    ) -> list[int]:
        cx, cy = self._cell(point)
        seen: set[int] = set()
        result: list[int] = []
        for gx in range(cx - radius_cells, cx + radius_cells + 1):
            for gy in range(cy - radius_cells, cy + radius_cells + 1):
                for idx in grid.get((gx, gy), ()):
                    if idx not in seen:
                        seen.add(idx)
                        result.append(idx)
        return result

    def _edge_candidates_in_radius(self, point: Array, radius: float) -> list[int]:
        lo = np.floor((point - radius) / self._grid_cell_size).astype(int)
        hi = np.floor((point + radius) / self._grid_cell_size).astype(int)
        seen: set[int] = set()
        result: list[int] = []
        for gx in range(int(lo[0]), int(hi[0]) + 1):
            for gy in range(int(lo[1]), int(hi[1]) + 1):
                for idx in self._edge_grid.get((gx, gy), ()):
                    if idx not in seen:
                        seen.add(idx)
                        result.append(idx)
        return result
