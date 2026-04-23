from typing import List, Tuple

import numpy as np

from backend.algorithms.base import BasePathPlanner
from backend.algorithms.registry import AlgorithmRegistry
from backend.models.domain import GeoPoint


@AlgorithmRegistry.register_path_planner("orthogonal_safe_3d")
class OrthogonalSafePlanner(BasePathPlanner):
    """
    正交安全路径规划器。

    仅尝试少量正交折线路径，速度极快，适合大规模场景快速出解。
    """

    def __init__(self, env, grid_resolution=None, margin: float = 20.0):
        super().__init__(env)
        self.margin = max(10.0, float(margin))
        self.cache = {}

    def get_route(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        current_time=None,
        speed=None,
    ) -> Tuple[List[GeoPoint], float]:
        timer_token = self.start_route_timer()
        try:
            self.check_cancelled()
            self.record_route_request()
            key = self._build_cache_key(start, goal, current_time, speed)
            if key in self.cache:
                self.record_cache_hit()
                return self.cache[key]
            self.record_cache_miss()

            estimated_arrival = self._estimate_arrival(start, goal, current_time, speed)
            if self.env.is_collision(start, current_time=current_time) or self.env.is_collision(goal, current_time=estimated_arrival):
                self.cache[key] = ([], float("inf"))
                return self.cache[key]

            if self.env.line_of_sight(start, goal, start_time=current_time, end_time=estimated_arrival):
                distance = self._distance(start, goal)
                self.cache[key] = ([start, goal], distance)
                return self.cache[key]

            cruise_z = self._resolve_cruise_altitude(start, goal)
            climb = GeoPoint(start.x, start.y, cruise_z)
            descend = GeoPoint(goal.x, goal.y, cruise_z)
            x_corner = GeoPoint(goal.x, start.y, cruise_z)
            y_corner = GeoPoint(start.x, goal.y, cruise_z)

            candidates = [
                [start, climb, x_corner, descend, goal],
                [start, climb, y_corner, descend, goal],
                [start, GeoPoint(goal.x, start.y, start.z), goal],
                [start, GeoPoint(start.x, goal.y, start.z), goal],
            ]

            for path in candidates:
                if self._path_clear(path, current_time, speed):
                    path, distance = self._finalize_route_result(
                        path,
                        current_time=current_time,
                        speed=speed,
                    )
                    self.cache[key] = (path, distance)
                    return self.cache[key]

            self.cache[key] = ([], float("inf"))
            return self.cache[key]
        finally:
            self.finish_route_timer(timer_token)

    def _resolve_cruise_altitude(self, start: GeoPoint, goal: GeoPoint) -> float:
        obstacle_top = 0.0
        for constraint in self.env.constraints:
            _, max_z = constraint._vertical_limits()
            obstacle_top = max(obstacle_top, max_z)
        return min(max(start.z, goal.z, obstacle_top + self.margin), float(self.env.limit_z) - self.margin)

    def _estimate_arrival(self, start: GeoPoint, goal: GeoPoint, current_time, speed):
        return self._estimate_arrival_time(
            start,
            goal,
            current_time=current_time,
            speed=speed,
        )

    def _path_clear(self, path: List[GeoPoint], current_time, speed) -> bool:
        segment_start_time = current_time
        for index in range(len(path) - 1):
            segment_end_time = self._estimate_arrival(path[index], path[index + 1], segment_start_time, speed)
            if not self.env.line_of_sight(path[index], path[index + 1], start_time=segment_start_time, end_time=segment_end_time):
                return False
            segment_start_time = segment_end_time
        return True

    def _distance(self, start: GeoPoint, goal: GeoPoint) -> float:
        return float(np.linalg.norm(np.array(start.as_tuple()) - np.array(goal.as_tuple())))

    def _path_distance(self, path: List[GeoPoint]) -> float:
        if len(path) < 2:
            return float("inf")
        return sum(self._distance(path[index], path[index + 1]) for index in range(len(path) - 1))
