"""Heatmap analyzer for UAV safety overlap risks."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

Point2D = Tuple[float, float]
GridKey = Tuple[int, int]


def _distance_to_box(px: float, py: float, west: float, east: float, south: float, north: float) -> float:
    dx = max(west - px, 0.0, px - east)
    dy = max(south - py, 0.0, py - north)
    return float(np.sqrt(dx * dx + dy * dy))


def _distance_to_cylinder(px: float, py: float, cx: float, cy: float, radius: float) -> float:
    return max(0.0, float(np.sqrt((px - cx) ** 2 + (py - cy) ** 2)) - radius)


def _distance_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_len_sq = abx * abx + aby * aby

    if ab_len_sq <= 1e-12:
        return float(np.sqrt((px - ax) ** 2 + (py - ay) ** 2))

    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return float(np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2))


def _point_in_polygon(px: float, py: float, vertices: List[Point2D]) -> bool:
    inside = False
    j = len(vertices) - 1

    for i in range(len(vertices)):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        denominator = (yj - yi) if (yj - yi) != 0 else 1e-9
        intersects = ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / denominator + xi)
        if intersects:
            inside = not inside
        j = i

    return inside


def _distance_to_polygon(px: float, py: float, vertices: List[Point2D]) -> float:
    if len(vertices) < 3:
        return float("inf")

    if _point_in_polygon(px, py, vertices):
        return 0.0

    min_dist = float("inf")
    for i in range(len(vertices)):
        ax, ay = vertices[i]
        bx, by = vertices[(i + 1) % len(vertices)]
        min_dist = min(min_dist, _distance_point_to_segment(px, py, ax, ay, bx, by))

    return float(min_dist)


def _extract_drone_safety_map(drones: List[Any], safe_radius: float) -> Dict[str, float]:
    safety_map: Dict[str, float] = {}

    for drone in drones or []:
        if isinstance(drone, dict):
            drone_id = str(drone.get("id", "")).strip()
            drone_radius = drone.get("safetyDistance")
        else:
            drone_id = str(getattr(drone, "id", "")).strip()
            drone_radius = getattr(drone, "safetyDistance", None)

        if not drone_id:
            continue

        try:
            radius = float(drone_radius)
            safety_map[drone_id] = radius if radius > 0 else float(safe_radius)
        except (TypeError, ValueError):
            safety_map[drone_id] = float(safe_radius)

    return safety_map


def _parse_polygon_vertices(vertices_raw: Any) -> List[Point2D]:
    vertices: List[Point2D] = []
    if not isinstance(vertices_raw, list):
        return vertices

    for vertex in vertices_raw:
        if isinstance(vertex, dict):
            vertices.append((float(vertex.get("x", vertex.get("lng", 0))), float(vertex.get("y", vertex.get("lat", 0)))))
        elif isinstance(vertex, (list, tuple)) and len(vertex) >= 2:
            vertices.append((float(vertex[0]), float(vertex[1])))

    return vertices


def _distance_to_zone(px: float, py: float, zone: dict) -> float:
    vertices = _parse_polygon_vertices(zone.get("vertices", []))
    if len(vertices) >= 3:
        return _distance_to_polygon(px, py, vertices)

    shape = zone.get("shape", "box")
    if shape == "box":
        return _distance_to_box(
            px,
            py,
            float(zone.get("west_lng", 0)),
            float(zone.get("east_lng", 0)),
            float(zone.get("south_lat", 0)),
            float(zone.get("north_lat", 0)),
        )

    if shape == "cylinder":
        return _distance_to_cylinder(
            px,
            py,
            float(zone.get("center_lng", 0)),
            float(zone.get("center_lat", 0)),
            float(zone.get("radius", 50)),
        )

    return float("inf")


def _iter_circle_cells(cx: float, cy: float, radius: float, grid_size: float) -> Iterable[GridKey]:
    if radius <= 0 or grid_size <= 0:
        return

    min_gx = int(np.floor((cx - radius) / grid_size))
    max_gx = int(np.ceil((cx + radius) / grid_size))
    min_gy = int(np.floor((cy - radius) / grid_size))
    max_gy = int(np.ceil((cy + radius) / grid_size))

    radius_sq = radius * radius
    for gx in range(min_gx, max_gx + 1):
        for gy in range(min_gy, max_gy + 1):
            px = (gx + 0.5) * grid_size
            py = (gy + 0.5) * grid_size
            dx = px - cx
            dy = py - cy
            if dx * dx + dy * dy <= radius_sq:
                yield gx, gy


def _extract_xyz(point: dict) -> Tuple[float, float, float, float]:
    location = point.get("location") or {}
    x = float(point.get("x", point.get("lng", location.get("x", 0))))
    y = float(point.get("y", point.get("lat", location.get("y", 0))))
    z = float(point.get("z", point.get("altitude", location.get("z", 50))))
    t = float(point.get("time", 0))
    return x, y, z, t


def _sample_position(route: List[dict], t: float) -> Tuple[float, float, float]:
    if not route:
        return 0.0, 0.0, 50.0

    x0, y0, z0, t0 = _extract_xyz(route[0])
    if t <= t0:
        return x0, y0, z0

    x1, y1, z1, t1 = _extract_xyz(route[-1])
    if t >= t1:
        return x1, y1, z1

    for i in range(len(route) - 1):
        ax, ay, az, at = _extract_xyz(route[i])
        bx, by, bz, bt = _extract_xyz(route[i + 1])
        if at <= t <= bt:
            duration = bt - at
            ratio = 0.0 if duration <= 1e-9 else (t - at) / duration
            return (
                ax + (bx - ax) * ratio,
                ay + (by - ay) * ratio,
                az + (bz - az) * ratio,
            )

    return x1, y1, z1


def _build_time_points(paths: Dict[str, List[dict]], step: float) -> Dict[float, List[dict]]:
    if not paths:
        return {}

    route_ranges: List[Tuple[str, List[dict], float, float]] = []
    global_start = float("inf")
    global_end = 0.0

    for drone_id, route in paths.items():
        if not route:
            continue

        sorted_route = sorted(route, key=lambda p: float((p or {}).get("time", 0)))
        start_t = float((sorted_route[0] or {}).get("time", 0))
        end_t = float((sorted_route[-1] or {}).get("time", 0))

        route_ranges.append((drone_id, sorted_route, start_t, end_t))
        global_start = min(global_start, start_t)
        global_end = max(global_end, end_t)

    if not route_ranges:
        return {}

    if global_end <= global_start:
        global_end = global_start + step

    time_points: Dict[float, List[dict]] = {}
    t = global_start

    while t <= global_end + 1e-9:
        ts = round(t, 6)
        frame_points: List[dict] = []

        for drone_id, route, start_t, end_t in route_ranges:
            if ts < start_t:
                continue
            clamped_t = min(ts, end_t)
            x, y, z = _sample_position(route, clamped_t)
            frame_points.append(
                {
                    "drone_id": drone_id,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )

        if frame_points:
            time_points[ts] = frame_points

        t += step

    return time_points


def generate_heatmap_data(
    paths: Dict[str, List[dict]],
    no_fly_zones: List[dict],
    safe_radius: float = 20.0,
    heatmap_type: str = "hotspot",
    drones: List[Any] | None = None,
    time_step: float = 1.0,
    grid_size: float = 40.0,
) -> List[dict]:
    """Generate grid heatmap points from UAV safety-circle coverage and overlap counts."""
    drone_safety_map = _extract_drone_safety_map(drones or [], safe_radius)

    step = float(time_step) if np.isfinite(time_step) and time_step > 0 else 1.0
    grid = float(grid_size) if np.isfinite(grid_size) and grid_size > 0 else 40.0

    time_points = _build_time_points(paths or {}, step)

    if heatmap_type == "density":
        density_points: List[dict] = []
        for points in time_points.values():
            for p in points:
                density_points.append({"x": round(p["x"], 2), "y": round(p["y"], 2), "weight": 1.0})
        return density_points

    accumulated_cells: Dict[GridKey, float] = {}
    overlap_cells: Dict[GridKey, float] = {}
    zone_risk_cells: Dict[GridKey, float] = {}

    for points in time_points.values():
        frame_coverage: Dict[GridKey, int] = {}

        # 1) Accumulate the raw safety-circle footprint of every UAV on the map grid.
        for p in points:
            drone_radius = float(drone_safety_map.get(p["drone_id"], safe_radius))
            if drone_radius <= 0:
                continue

            for gx, gy in _iter_circle_cells(p["x"], p["y"], drone_radius, grid):
                frame_coverage[(gx, gy)] = frame_coverage.get((gx, gy), 0) + 1

        for cell, coverage_count in frame_coverage.items():
            accumulated_cells[cell] = accumulated_cells.get(cell, 0.0) + float(coverage_count)
            if coverage_count > 1:
                overlap_cells[cell] = overlap_cells.get(cell, 0.0) + float(coverage_count - 1)
                accumulated_cells[cell] += float(coverage_count - 1) * 2.5

        # 2) Add a smaller penalty where the safety circle intrudes into a no-fly zone.
        for p in points:
            drone_radius = float(drone_safety_map.get(p["drone_id"], safe_radius))
            if drone_radius <= 0:
                continue

            for zone in no_fly_zones or []:
                min_z = float(zone.get("min_altitude", zone.get("minAltitude", 0)))
                max_z = float(zone.get("max_altitude", zone.get("maxAltitude", 500)))
                if not (min_z - drone_radius <= p["z"] <= max_z + drone_radius):
                    continue

                dist = _distance_to_zone(p["x"], p["y"], zone)
                if dist >= drone_radius:
                    continue

                overlap_ratio = max(0.0, (drone_radius - dist) / drone_radius)
                if overlap_ratio <= 0:
                    continue

                for gx, gy in _iter_circle_cells(p["x"], p["y"], drone_radius, grid):
                    zone_risk_cells[(gx, gy)] = zone_risk_cells.get((gx, gy), 0.0) + overlap_ratio
                    accumulated_cells[(gx, gy)] = accumulated_cells.get((gx, gy), 0.0) + overlap_ratio * 0.75

    heat_points: List[dict] = []
    for (gx, gy), weight in accumulated_cells.items():
        if weight <= 0:
            continue
        heat_points.append(
            {
                "x": round((gx + 0.5) * grid, 2),
                "y": round((gy + 0.5) * grid, 2),
                "weight": round(weight, 4),
                "coverageCount": round(weight, 4),
                "overlapCount": round(overlap_cells.get((gx, gy), 0.0), 4),
                "zoneRisk": round(zone_risk_cells.get((gx, gy), 0.0), 4),
            }
        )

    return heat_points
