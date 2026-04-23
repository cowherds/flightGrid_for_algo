from typing import List, Tuple

import numpy as np

from backend.algorithms.base import BasePathPlanner
from backend.algorithms.registry import AlgorithmRegistry
from backend.config.settings import settings
from backend.models.domain import GeoPoint


@AlgorithmRegistry.register_path_planner("direct_safe_3d")
class DirectSafePlanner(BasePathPlanner):
    """
    快速安全直飞规划器。

    规则很直接：
    1. 先尝试起点到终点直飞。
    2. 若直飞失败，则抬升到安全巡航高度。
    3. 再尝试“起点垂直爬升 -> 高空横移 -> 目标垂直下降”。
    4. 若仍失败，再尝试两种 L 形高空转折。

    它不追求最优，只追求非常快地给出一条可行路径。
    """

    def __init__(self, env, grid_resolution=None, margin: float = 20.0):
        super().__init__(env)
        self.res = grid_resolution or settings.DEFAULT_GRID_RESOLUTION
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

            if self._segment_clear(start, goal, current_time, estimated_arrival):
                distance = self._distance(start, goal)
                self.cache[key] = ([start, goal], distance)
                return self.cache[key]

            safe_altitude = self._resolve_safe_altitude(start, goal)
            candidate_paths = self._build_candidate_paths(start, goal, safe_altitude)
            for path in candidate_paths:
                if self._path_is_clear(path, current_time, speed):
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

    def _resolve_safe_altitude(self, start: GeoPoint, goal: GeoPoint) -> float:
        max_constraint_z = 0.0
        for constraint in self.env.constraints:
            _, max_z = constraint._vertical_limits()
            max_constraint_z = max(max_constraint_z, max_z)

        desired = max(start.z, goal.z, max_constraint_z + self.margin)
        upper_bound = max(self.margin, float(self.env.limit_z) - self.margin)
        return min(desired, upper_bound)

    def _build_candidate_paths(self, start: GeoPoint, goal: GeoPoint, cruise_z: float) -> List[List[GeoPoint]]:
        climb = GeoPoint(start.x, start.y, cruise_z)
        descend = GeoPoint(goal.x, goal.y, cruise_z)
        corner_x_first = GeoPoint(goal.x, start.y, cruise_z)
        corner_y_first = GeoPoint(start.x, goal.y, cruise_z)

        return [
            [start, climb, descend, goal],
            [start, climb, corner_x_first, descend, goal],
            [start, climb, corner_y_first, descend, goal],
        ]

    def _path_is_clear(self, path: List[GeoPoint], current_time, speed) -> bool:
        segment_start_time = current_time
        for index in range(len(path) - 1):
            start = path[index]
            goal = path[index + 1]
            segment_end_time = self._estimate_arrival(start, goal, segment_start_time, speed)
            if not self._segment_clear(start, goal, segment_start_time, segment_end_time):
                return False
            segment_start_time = segment_end_time
        return True

    def _segment_clear(self, start: GeoPoint, goal: GeoPoint, start_time, end_time) -> bool:
        return self.env.line_of_sight(
            start,
            goal,
            start_time=start_time,
            end_time=end_time,
        )

    def _estimate_arrival(self, start: GeoPoint, goal: GeoPoint, current_time, speed):
        return self._estimate_arrival_time(
            start,
            goal,
            current_time=current_time,
            speed=speed,
        )

    def _distance(self, start: GeoPoint, goal: GeoPoint) -> float:
        return float(np.linalg.norm(np.array(start.as_tuple()) - np.array(goal.as_tuple())))

    def _path_distance(self, path: List[GeoPoint]) -> float:
        if len(path) < 2:
            return float("inf")
        return sum(self._distance(path[index], path[index + 1]) for index in range(len(path) - 1))


@AlgorithmRegistry.register_path_planner("direct_safe_conservative_3d")
class DirectSafeConservativePlanner(DirectSafePlanner):
    """
    更保守的直飞规划：抬升安全边距更高，降低擦障风险。
    """

    def __init__(self, env, grid_resolution=None):
        super().__init__(env, grid_resolution=grid_resolution, margin=40.0)
