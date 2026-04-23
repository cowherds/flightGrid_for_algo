import heapq
from typing import List, Tuple

import numpy as np

from backend.algorithms.base import BasePathPlanner
from backend.algorithms.registry import AlgorithmRegistry
from backend.config.settings import settings
from backend.models.domain import GeoPoint


@AlgorithmRegistry.register_path_planner("weighted_a_star_3d")
class WeightedAStarPlanner(BasePathPlanner):
    """
    加权 A* 路径规划器。

    设计目标不是绝对最优，而是用更强的启发式减少扩展节点数，
    在中大规模场景下比普通 A* 更快得到可行路径。
    """

    def __init__(self, env, grid_resolution=None, heuristic_weight: float = 2.2):
        super().__init__(env)
        self.res = grid_resolution or settings.DEFAULT_GRID_RESOLUTION
        self.heuristic_weight = max(1.0, float(heuristic_weight))
        self.max_expansions = 25000
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

            estimated_arrival = self._estimate_arrival_time(
                start,
                goal,
                current_time=current_time,
                speed=speed,
            )

            if self.env.is_collision(start, current_time=current_time) or self.env.is_collision(goal, current_time=estimated_arrival):
                self.cache[key] = ([], float("inf"))
                return self.cache[key]

            if self.env.line_of_sight(
                start,
                goal,
                start_time=current_time,
                end_time=estimated_arrival,
            ):
                distance = np.linalg.norm(np.array(start.as_tuple()) - np.array(goal.as_tuple()))
                self.cache[key] = ([start, goal], distance)
                return self.cache[key]

            start_grid = self._snap_to_grid(start)
            goal_grid = self._snap_to_grid(goal)

            # 用更小的邻接集合减少扩展分支。
            directions = [
                (self.res, 0, 0), (-self.res, 0, 0),
                (0, self.res, 0), (0, -self.res, 0),
                (0, 0, self.res), (0, 0, -self.res),
                (self.res, self.res, 0), (self.res, -self.res, 0),
                (-self.res, self.res, 0), (-self.res, -self.res, 0),
            ]

            open_set = [(0.0, start_grid)]
            came_from = {}
            g_score = {start_grid: 0.0}
            visited = set()
            expansions = 0
            while open_set:
                self.check_cancelled()
                _, current = heapq.heappop(open_set)
                if current in visited:
                    continue
                visited.add(current)
                expansions += 1

                if expansions > self.max_expansions:
                    break

                current_point = GeoPoint(*current)
                current_node_time = None
                if current_time is not None and speed and speed > 0:
                    current_node_time = current_time + g_score[current] / speed
                goal_arrival_time = None
                if current_node_time is not None:
                    goal_arrival_time = self.env.estimate_arrival_time(
                        current_point,
                        goal,
                        current_node_time,
                        speed,
                    )

                if self._strict_line_of_sight(
                    current_point,
                    goal,
                    start_time=current_node_time,
                    end_time=goal_arrival_time,
                ):
                    path = self._reconstruct_path(came_from, current, start, goal)
                    distance = self._path_distance(path)
                    self.cache[key] = (path, distance)
                    return self.cache[key]

                for dx, dy, dz in directions:
                    neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                    neighbor_point = GeoPoint(*neighbor)

                    edge_start_time = None
                    if current_time is not None and speed and speed > 0:
                        edge_start_time = current_time + g_score[current] / speed

                    travel_end_time = None
                    if edge_start_time is not None:
                        travel_end_time = self.env.estimate_arrival_time(
                            current_point,
                            neighbor_point,
                            edge_start_time,
                            speed,
                        )

                    if self.env.is_collision(neighbor_point, current_time=travel_end_time):
                        continue

                    if not self.env.line_of_sight(
                        current_point,
                        neighbor_point,
                        start_time=edge_start_time,
                        end_time=travel_end_time,
                    ):
                        continue

                    tentative_g = g_score[current] + np.linalg.norm([dx, dy, dz])
                    if tentative_g >= g_score.get(neighbor, float("inf")):
                        continue

                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heuristic = np.linalg.norm(np.array(neighbor) - np.array(goal_grid))
                    priority = tentative_g + (self.heuristic_weight * heuristic)
                    heapq.heappush(open_set, (priority, neighbor))

            self.cache[key] = ([], float("inf"))
            return self.cache[key]
        finally:
            self.finish_route_timer(timer_token)

    def _snap_to_grid(self, point: GeoPoint) -> Tuple[float, float, float]:
        return (
            round(point.x / self.res) * self.res,
            round(point.y / self.res) * self.res,
            round(point.z / self.res) * self.res,
        )

    def _reconstruct_path(
        self,
        came_from,
        current: Tuple[float, float, float],
        start: GeoPoint,
        goal: GeoPoint,
    ) -> List[GeoPoint]:
        path = [GeoPoint(*current)]
        while current in came_from:
            current = came_from[current]
            path.append(GeoPoint(*current))
        path.reverse()

        full_path = [start]
        for point in path[1:]:
            full_path.append(point)
        full_path.append(goal)
        return full_path


@AlgorithmRegistry.register_path_planner("weighted_a_star_fast_3d")
class WeightedAStarFastPlanner(WeightedAStarPlanner):
    """更激进启发式，优先速度。"""

    def __init__(self, env, grid_resolution=None):
        super().__init__(env, grid_resolution=grid_resolution, heuristic_weight=3.0)


@AlgorithmRegistry.register_path_planner("weighted_a_star_quality_3d")
class WeightedAStarQualityPlanner(WeightedAStarPlanner):
    """较保守启发式，优先路径质量。"""

    def __init__(self, env, grid_resolution=None):
        super().__init__(env, grid_resolution=grid_resolution, heuristic_weight=1.4)
