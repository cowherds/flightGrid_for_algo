import numpy as np
from math import ceil
from typing import List

from backend.algorithms.utils.cpp_bridge import get_blocked_flags
from backend.models.domain import GeoPoint, SpatialConstraint


class CityEnvironment:
    def __init__(
        self,
        constraints: List[SpatialConstraint],
        limits=(500, 500, 150),
        line_sample_step: float = 5.0,
    ):
        self.constraints = constraints
        self.limit_x, self.limit_y, self.limit_z = limits
        self.line_sample_step = max(1.0, float(line_sample_step))
        self.has_time_dependent_constraints = any(
            self._constraint_has_dynamic_window(constraint)
            for constraint in constraints
        )
        self.has_time_dependent_blocking_constraints = any(
            constraint.blocks_flight_at(None)
            and self._constraint_has_dynamic_window(constraint)
            for constraint in constraints
        )
        self.has_speed_factor_constraints = any(
            getattr(constraint, "zoneKind", None) == "weather_slow"
            for constraint in constraints
        )
        self._cpp_fastpath_enabled = True
        self._zone_vertices_cache_by_altitude: dict[float, list[dict]] = {}

    @staticmethod
    def _constraint_has_dynamic_window(constraint) -> bool:
        """
        判断约束是否真的“随时间变化”。

        前端常把全天生效区域写成 `0 -> 86400`。
        这种约束对本地单日调度来说应视为静态，不该关闭静态快路径和缓存复用。
        """
        start_time = getattr(constraint, "startActiveTime", None)
        end_time = getattr(constraint, "endActiveTime", None)
        if start_time is None and end_time is None:
            return False

        try:
            normalized_start = None if start_time is None else float(start_time)
        except (TypeError, ValueError):
            normalized_start = start_time
        try:
            normalized_end = None if end_time is None else float(end_time)
        except (TypeError, ValueError):
            normalized_end = end_time

        if normalized_start is not None and normalized_start > 0.0:
            return True
        if normalized_end is not None and normalized_end < 86400.0:
            return True
        return False

    def is_collision(self, point: GeoPoint, current_time=None) -> bool:
        for constraint in self.constraints:
            if (
                constraint.blocks_flight_at(current_time)
                and constraint.contains_point(point, current_time=current_time)
            ):
                return True

        return not (
            0 <= point.x <= self.limit_x
            and 0 <= point.y <= self.limit_y
            and 0 <= point.z <= self.limit_z
        )

    def get_point_speed_factor(self, point: GeoPoint, current_time=None) -> float:
        factor = 1.0
        for constraint in self.constraints:
            if not constraint.contains_point(point, current_time=current_time):
                continue
            factor = min(factor, constraint.get_speed_factor(current_time))
        return max(0.05, min(1.0, float(factor)))

    def estimate_segment_travel_time(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        speed: float,
        *,
        start_time: float | None = None,
        steps: int | None = None,
    ) -> float:
        speed = float(speed or 0.0)
        if speed <= 0:
            return float("inf")

        distance = np.linalg.norm(np.array(start.as_tuple()) - np.array(goal.as_tuple()))
        if distance <= 1e-9:
            return 0.0

        if not self.has_speed_factor_constraints:
            return float(distance / speed)

        if steps is None:
            steps = max(3, int(ceil(distance / self.line_sample_step)) + 1)

        elapsed = 0.0
        previous = start
        for index in range(1, steps):
            ratio = index / float(steps - 1)
            current = GeoPoint(
                x=start.x + ratio * (goal.x - start.x),
                y=start.y + ratio * (goal.y - start.y),
                z=start.z + ratio * (goal.z - start.z),
            )
            segment_distance = np.linalg.norm(np.array(previous.as_tuple()) - np.array(current.as_tuple()))
            midpoint = GeoPoint(
                x=(previous.x + current.x) / 2.0,
                y=(previous.y + current.y) / 2.0,
                z=(previous.z + current.z) / 2.0,
            )
            sample_time = None if start_time is None else start_time + elapsed
            factor = self.get_point_speed_factor(midpoint, current_time=sample_time)
            elapsed += segment_distance / max(speed * factor, 0.05)
            previous = current

        return float(elapsed)

    def estimate_path_travel_time(
        self,
        path: list[GeoPoint] | tuple[GeoPoint, ...],
        speed: float,
        *,
        start_time: float | None = None,
    ) -> float:
        if len(path) < 2:
            return 0.0

        elapsed = 0.0
        for index in range(1, len(path)):
            segment_start_time = None if start_time is None else start_time + elapsed
            elapsed += self.estimate_segment_travel_time(
                path[index - 1],
                path[index],
                speed,
                start_time=segment_start_time,
            )
        return float(elapsed)

    def estimate_arrival_time(self, start: GeoPoint, goal: GeoPoint, current_time, speed):
        if current_time is None:
            return None
        return current_time + self.estimate_segment_travel_time(
            start,
            goal,
            speed,
            start_time=current_time,
        )

    def line_of_sight(self, start: GeoPoint, goal: GeoPoint, steps=None, start_time=None, end_time=None) -> bool:
        if self.is_collision(start, current_time=start_time) or self.is_collision(goal, current_time=end_time):
            return False

        if not self.constraints:
            return True

        cpp_fast = self._line_of_sight_cpp_fast(start, goal, start_time=start_time, end_time=end_time)
        if cpp_fast is not None:
            return cpp_fast

        distance = np.linalg.norm(np.array(start.as_tuple()) - np.array(goal.as_tuple()))
        if steps is None:
            # 采样间距跟随规划分辨率放宽，避免在长航段上做过密检测。
            steps = max(3, int(ceil(distance / self.line_sample_step)) + 1)

        for t in np.linspace(0.0, 1.0, steps):
            sample_time = None
            if start_time is not None and end_time is not None:
                sample_time = start_time + t * (end_time - start_time)
            elif start_time is not None:
                sample_time = start_time
            elif end_time is not None:
                sample_time = end_time

            sample = GeoPoint(
                x=start.x + t * (goal.x - start.x),
                y=start.y + t * (goal.y - start.y),
                z=start.z + t * (goal.z - start.z),
            )
            if self.is_collision(sample, current_time=sample_time):
                return False

        return True

    def _line_of_sight_cpp_fast(self, start: GeoPoint, goal: GeoPoint, start_time=None, end_time=None):
        """
        Fast path using C++ batch segment intersection.

        It is only applied when:
        - constraints are static (no time-dependent activation),
        - segment is horizontal in altitude (constant z),
        because the C++ primitive currently solves 2D footprint intersection.
        """
        if not self._cpp_fastpath_enabled:
            return None
        if self.has_time_dependent_blocking_constraints:
            return None
        if abs(float(start.z) - float(goal.z)) > 1e-6:
            return None

        flight_z = float(start.z)
        zones = self._collect_2d_zones_for_altitude(flight_z)
        if not zones:
            return True

        try:
            flags = get_blocked_flags(
                [[float(start.x), float(start.y), float(goal.x), float(goal.y)]],
                zones,
            )
            if not flags:
                return None
            return not bool(flags[0])
        except Exception:
            self._cpp_fastpath_enabled = False
            return None

    def _collect_2d_zones_for_altitude(self, altitude: float) -> list[dict]:
        cache_key = round(float(altitude), 6)
        cached = self._zone_vertices_cache_by_altitude.get(cache_key)
        if cached is not None:
            return cached

        zones: list[dict] = []
        for constraint in self.constraints:
            if not constraint.blocks_flight_at(None):
                continue
            min_z, max_z = constraint._vertical_limits()
            if not (float(min_z) <= altitude <= float(max_z)):
                continue
            vertices = self._constraint_vertices_2d(constraint)
            if len(vertices) < 3:
                continue
            zones.append(
                {
                    "id": constraint.id,
                    "vertices": [{"x": float(x), "y": float(y)} for x, y in vertices],
                }
            )
        self._zone_vertices_cache_by_altitude[cache_key] = zones
        return zones

    @staticmethod
    def _constraint_vertices_2d(constraint: SpatialConstraint) -> list[tuple[float, float]]:
        if constraint.shape == "polygon" and constraint.polygon:
            return [(float(x), float(y)) for x, y in constraint.polygon]

        if constraint.shape == "box" and constraint.box:
            min_x, min_y, _min_z, max_x, max_y, _max_z = constraint.box
            return [
                (float(min_x), float(min_y)),
                (float(max_x), float(min_y)),
                (float(max_x), float(max_y)),
                (float(min_x), float(max_y)),
            ]

        if constraint.shape == "cylinder" and constraint.cylinder:
            center_x, center_y, radius, _min_z, _max_z = constraint.cylinder
            if radius <= 0:
                return []
            points: list[tuple[float, float]] = []
            sample_count = 16
            for i in range(sample_count):
                theta = (2.0 * np.pi * i) / float(sample_count)
                points.append(
                    (
                        float(center_x + radius * np.cos(theta)),
                        float(center_y + radius * np.sin(theta)),
                    )
                )
            return points

        return []
