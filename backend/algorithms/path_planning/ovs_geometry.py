"""
2D geometry helpers for OVS (orthogonal visibility / guidance-point style) planning.

Self-contained numpy implementation — no external mamp dependency.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Vec2 = np.ndarray  # shape (2,)


def l2norm(a: Vec2, b: Vec2) -> float:
    return float(np.linalg.norm(a - b))


def normalize(v: Vec2) -> Vec2:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(2, dtype=float)
    return v / n


def left_of(p0: Vec2, p1: Vec2, p2: Vec2) -> float:
    """>0 if p2 is left of ray p0->p1, <0 right, ~0 on line."""
    return float((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))


def point_line_dist(p0: Vec2, p1: Vec2, p2: Vec2) -> float:
    """Perpendicular distance from p2 to infinite line through p0,p1."""
    line_len = float(np.linalg.norm(p1 - p0))
    if line_len < 1e-12:
        return float(np.linalg.norm(p2 - p0))
    return abs(left_of(p0, p1, p2)) / line_len


def unit_normal_vector(v: Vec2) -> Tuple[Vec2, Vec2]:
    """Left and right unit normals for vector v (2D)."""
    if float(np.linalg.norm(v)) < 1e-12:
        z = np.zeros(2, dtype=float)
        return z, z
    left = np.array([-v[1], v[0]], dtype=float)
    left = left / np.linalg.norm(left)
    right = -left
    return left, right


def _on_segment(a: Vec2, b: Vec2, c: Vec2, eps: float = 1e-9) -> bool:
    return (
        min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
    )


def _orient(a: Vec2, b: Vec2, c: Vec2) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def segments_intersect(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> bool:
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)

    if (o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and (o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0):
        return True

    eps = 1e-9
    if abs(o1) <= eps and _on_segment(a, b, c):
        return True
    if abs(o2) <= eps and _on_segment(a, b, d):
        return True
    if abs(o3) <= eps and _on_segment(c, d, a):
        return True
    if abs(o4) <= eps and _on_segment(c, d, b):
        return True
    return False


def intersect_polygon_edges(p0: Vec2, p1: Vec2, vertices: Sequence[Vec2]) -> bool:
    n = len(vertices)
    if n < 2:
        return False
    for i in range(n):
        a = np.asarray(vertices[i], dtype=float)
        b = np.asarray(vertices[(i + 1) % n], dtype=float)
        if segments_intersect(p0, p1, a, b):
            return True
    return False


def point_in_polygon(point: Vec2, vertices: Sequence[Vec2]) -> bool:
    """Ray casting; boundary counts as inside."""
    x, y = float(point[0]), float(point[1])
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(vertices[i][0]), float(vertices[i][1])
        xj, yj = float(vertices[j][0]), float(vertices[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi):
            inside = not inside
        j = i
    return inside


def pos_in_polygons(pos: Vec2, obstacles: Iterable[object]) -> bool:
    for ob in obstacles:
        verts = getattr(ob, "vertices_pos", None) or getattr(ob, "vertices_", None)
        if verts is None:
            continue
        poly = [np.asarray(v.point_, dtype=float) if hasattr(v, "point_") else np.asarray(v, dtype=float) for v in verts]
        if len(poly) >= 3 and point_in_polygon(pos, poly):
            return True
    return False


def is_intersect_polys(p0: Tuple[float, float], p1: Tuple[float, float], obstacles: Iterable[object]) -> bool:
    a = np.array([p0[0], p0[1]], dtype=float)
    b = np.array([p1[0], p1[1]], dtype=float)
    for ob in obstacles:
        verts = getattr(ob, "vertices_pos", None) or getattr(ob, "vertices_", None)
        if verts is None:
            continue
        poly = []
        for v in verts:
            if hasattr(v, "point_"):
                poly.append(np.asarray(v.point_, dtype=float))
            else:
                poly.append(np.asarray(v, dtype=float))
        if len(poly) >= 2 and intersect_polygon_edges(a, b, poly):
            return True
    return False


def path_length(path: List[Tuple[float, float]]) -> float:
    if len(path) < 2:
        return 0.0
    s = 0.0
    for i in range(len(path) - 1):
        s += l2norm(np.array(path[i], dtype=float), np.array(path[i + 1], dtype=float))
    return float(s)


def seg_is_intersec(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    return segments_intersect(
        np.asarray(p1, dtype=float),
        np.asarray(p2, dtype=float),
        np.asarray(p3, dtype=float),
        np.asarray(p4, dtype=float),
    )


def smooth_path(
    path: List[Tuple[float, float]],
    inflated_obstacles: Iterable[object],
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Simple corner-cutting shortcut: repeatedly try to connect non-adjacent
    vertices with a straight segment that does not intersect inflated polygons.
    """
    if len(path) < 3:
        return list(path), path_length(path)

    pts = [np.array(p, dtype=float) for p in path]
    changed = True
    while changed and len(pts) >= 3:
        changed = False
        i = 0
        while i < len(pts) - 2:
            j = len(pts) - 1
            while j > i + 1:
                if not is_intersect_polys((pts[i][0], pts[i][1]), (pts[j][0], pts[j][1]), inflated_obstacles):
                    pts = pts[: i + 1] + pts[j:]
                    changed = True
                    break
                j -= 1
            if changed:
                break
            i += 1

    out = [(float(p[0]), float(p[1])) for p in pts]
    return out, path_length(out)


def polygon_signed_area(vertices: Sequence[Vec2]) -> float:
    s = 0.0
    n = len(vertices)
    for i in range(n):
        x1, y1 = float(vertices[i][0]), float(vertices[i][1])
        x2, y2 = float(vertices[(i + 1) % n][0]), float(vertices[(i + 1) % n][1])
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def ensure_ccw(vertices: List[Vec2]) -> List[Vec2]:
    if len(vertices) < 3:
        return vertices
    if polygon_signed_area(vertices) < 0:
        return list(reversed(vertices))
    return vertices


def vertex_convexity_ccw(prev_v: Vec2, cur_v: Vec2, next_v: Vec2) -> bool:
    """Convex vertex for CCW outer boundary (obstacle polygon)."""
    a = cur_v - prev_v
    b = next_v - cur_v
    cross = float(a[0] * b[1] - a[1] * b[0])
    return cross > 1e-9
