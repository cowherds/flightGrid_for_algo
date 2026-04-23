from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

from backend.algorithms.base import BasePathPlanner
from backend.algorithms.registry import AlgorithmRegistry
from backend.models.domain import GeoPoint, SpatialConstraint


class GeometricHeuristicPlanner(BasePathPlanner):
    """
    几何启发式路径规划器。

    核心思路：
    1. 先尝试直飞。
    2. 若直飞受阻，则定位直线路径上最先命中的障碍区域。
    3. 针对该障碍生成少量“翻越 / 侧绕”几何候选点。
    4. 递归地将路径拆分为更小的可行段。

    它不保证全局最优，更适合作为快速近似规划器，尤其适合规整建筑/盒状障碍较多的场景。
    """

    def __init__(
        self,
        env,
        grid_resolution=None,
        margin: float = 20.0,
        max_depth: int = 12,
        sample_step: float | None = None,
        enable_local_visibility: bool = False,
    ):
        super().__init__(env)
        self.margin = max(2.0, float(margin))
        self.max_depth = max(2, int(max_depth))
        default_step = float(sample_step or getattr(env, "line_sample_step", 5.0))
        self.sample_step = max(0.5, default_step)
        self.push_step = max(1.0, min(self.margin, self.sample_step))
        self.enable_local_visibility = bool(enable_local_visibility)
        self.cache: dict[tuple, tuple[list[GeoPoint], float]] = {}

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
            if (
                self.env.is_collision(start, current_time=current_time)
                or self.env.is_collision(goal, current_time=estimated_arrival)
            ):
                self.cache[key] = ([], float("inf"))
                return self.cache[key]

            if self._segment_clear(
                start,
                goal,
                start_time=current_time,
                end_time=estimated_arrival,
            ):
                result = ([start, goal], self._distance_between(start, goal))
                self.cache[key] = result
                return result

            path = self._build_geometric_path(
                start,
                goal,
                current_time=current_time,
                speed=speed,
            )
            if path:
                finalized_path, distance = self._finalize_geometric_path(
                    path,
                    current_time=current_time,
                    speed=speed,
                )
                if self._path_clear(
                    finalized_path,
                    current_time=current_time,
                    speed=speed,
                ):
                    self.cache[key] = (finalized_path, distance)
                    return self.cache[key]

            fallback = self._build_safe_fallback_path(
                start,
                goal,
                current_time=current_time,
                speed=speed,
            )
            if fallback:
                finalized_path, distance = self._finalize_geometric_path(
                    fallback,
                    current_time=current_time,
                    speed=speed,
                )
                if self._path_clear(
                    finalized_path,
                    current_time=current_time,
                    speed=speed,
                ):
                    self.cache[key] = (finalized_path, distance)
                    return self.cache[key]

            self.cache[key] = ([], float("inf"))
            return self.cache[key]
        finally:
            self.finish_route_timer(timer_token)

    def _build_geometric_path(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        *,
        current_time: Optional[float],
        speed: Optional[float],
    ) -> Optional[List[GeoPoint]]:
        path: List[GeoPoint] = [start, goal]
        visited_waypoints: set[tuple[float, float, float]] = set()

        for _ in range(self.max_depth):
            self.check_cancelled()
            segment_start_time = current_time
            inserted = False

            for index in range(len(path) - 1):
                segment_start = path[index]
                segment_goal = path[index + 1]
                segment_end_time = self._estimate_arrival_time(
                    segment_start,
                    segment_goal,
                    current_time=segment_start_time,
                    speed=speed,
                )

                if self._segment_clear(
                    segment_start,
                    segment_goal,
                    start_time=segment_start_time,
                    end_time=segment_end_time,
                ):
                    segment_start_time = segment_end_time
                    continue

                blocker = self._find_first_blocker(
                    segment_start,
                    segment_goal,
                    current_time=segment_start_time,
                    speed=speed,
                )
                if blocker is None:
                    return None

                local_detour = self._build_local_detour_segment(
                    segment_start,
                    segment_goal,
                    blocker=blocker,
                    current_time=segment_start_time,
                    speed=speed,
                )
                if local_detour is not None and len(local_detour) >= 2:
                    path = path[:index] + local_detour + path[index + 2 :]
                    inserted = True
                    break

                waypoint = self._select_best_waypoint(
                    segment_start,
                    segment_goal,
                    blocker=blocker,
                    current_time=segment_start_time,
                    speed=speed,
                    visited_waypoints=visited_waypoints,
                )
                if waypoint is None:
                    return None

                visited_waypoints.add(self._quantize_point(waypoint))
                path.insert(index + 1, waypoint)
                inserted = True
                break

            if not inserted:
                return path

        return path if self._path_clear(path, current_time=current_time, speed=speed) else None

    def _find_first_blocker(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        *,
        current_time: Optional[float],
        speed: Optional[float],
    ) -> Optional[Dict[str, object]]:
        direct_arrival = self._estimate_arrival_time(
            start,
            goal,
            current_time=current_time,
            speed=speed,
        )
        probe_time = self._interpolate_time(current_time, direct_arrival, 0.5)

        best: Optional[Dict[str, object]] = None
        best_enter = float("inf")

        for constraint in self.env.constraints:
            if not (
                constraint.blocks_flight_at(current_time)
                or constraint.blocks_flight_at(probe_time)
                or constraint.blocks_flight_at(direct_arrival)
            ):
                continue

            bounds = self._constraint_bounds(constraint)
            if bounds is None:
                continue

            segment_range = self._segment_bounds_intersection_range(start, goal, bounds)
            if segment_range is None:
                continue

            enter_ratio, exit_ratio = segment_range
            if enter_ratio < best_enter:
                best_enter = enter_ratio
                midpoint_ratio = (enter_ratio + exit_ratio) / 2.0
                best = {
                    "constraint": constraint,
                    "bounds": bounds,
                    "midpoint": self._interpolate_point(start, goal, midpoint_ratio),
                    "current_time": self._interpolate_time(current_time, direct_arrival, midpoint_ratio),
                    "enter_ratio": enter_ratio,
                    "exit_ratio": exit_ratio,
                    "enter_point": self._interpolate_point(start, goal, enter_ratio),
                    "exit_point": self._interpolate_point(start, goal, exit_ratio),
                }

        return best

    def _build_local_detour_segment(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        *,
        blocker: Dict[str, object],
        current_time: Optional[float],
        speed: Optional[float],
    ) -> Optional[List[GeoPoint]]:
        bounds = blocker.get("bounds")
        if bounds is None:
            return None

        enter_ratio = float(blocker.get("enter_ratio", 0.0))
        exit_ratio = float(blocker.get("exit_ratio", 1.0))
        segment_distance = max(1.0, self._distance_between(start, goal))
        pad_ratio = min(0.2, max(0.03, (self.margin * 1.5) / segment_distance))

        anchor_in = self._interpolate_point(start, goal, max(0.0, enter_ratio - pad_ratio))
        anchor_out = self._interpolate_point(start, goal, min(1.0, exit_ratio + pad_ratio))
        current_z = self._clamp(float(anchor_in.z), 0.0, float(self.env.limit_z))
        local_blockers = self._collect_local_blocker_bounds(
            anchor_in,
            anchor_out,
            base_bounds=bounds,
            current_time=current_time,
        )

        direction_x = float(goal.x) - float(start.x)
        direction_y = float(goal.y) - float(start.y)
        norm_xy = math.hypot(direction_x, direction_y)

        candidate_paths: list[List[GeoPoint]] = []

        over_z = max(local_bounds["max_z"] for local_bounds in local_blockers) + self.margin
        if over_z < float(self.env.limit_z):
            candidate_paths.append(
                [
                    start,
                    anchor_in,
                    GeoPoint(anchor_in.x, anchor_in.y, over_z),
                    GeoPoint(anchor_out.x, anchor_out.y, over_z),
                    anchor_out,
                    goal,
                ]
            )

        if norm_xy > 1e-6:
            normal_x = -direction_y / norm_xy
            normal_y = direction_x / norm_xy
            lateral_clearance = (
                0.5 * abs(normal_x) * (bounds["max_x"] - bounds["min_x"])
                + 0.5 * abs(normal_y) * (bounds["max_y"] - bounds["min_y"])
                + self.margin
            )
            for sign in (1.0, -1.0):
                direction = (normal_x * sign, normal_y * sign, 0.0)
                shifted_in = self._push_candidate_to_free_space(
                    GeoPoint(
                        anchor_in.x + direction[0] * lateral_clearance,
                        anchor_in.y + direction[1] * lateral_clearance,
                        current_z,
                    ),
                    direction=direction,
                    current_time=current_time,
                )
                shifted_out = self._push_candidate_to_free_space(
                    GeoPoint(
                        anchor_out.x + direction[0] * lateral_clearance,
                        anchor_out.y + direction[1] * lateral_clearance,
                        current_z,
                    ),
                    direction=direction,
                    current_time=current_time,
                )
                if shifted_in is None or shifted_out is None:
                    continue
                candidate_paths.append(
                    [
                        start,
                        anchor_in,
                        shifted_in,
                        shifted_out,
                        anchor_out,
                        goal,
                    ]
                )

        best_path: Optional[List[GeoPoint]] = None
        best_distance = float("inf")
        for candidate in candidate_paths:
            if not self._path_clear(candidate, current_time=current_time, speed=speed):
                continue
            candidate_distance = self._path_distance(candidate)
            if candidate_distance < best_distance:
                best_distance = candidate_distance
                best_path = candidate

        if self.enable_local_visibility and len(local_blockers) > 1:
            visibility_path = self._build_local_visibility_path(
                anchor_in,
                anchor_out,
                blocker_bounds_list=local_blockers,
                current_time=current_time,
            )
            if visibility_path is not None:
                visibility_candidate = [start] + visibility_path + [goal]
                if self._path_clear(visibility_candidate, current_time=current_time, speed=speed):
                    visibility_distance = self._path_distance(visibility_candidate)
                    if visibility_distance < best_distance:
                        best_distance = visibility_distance
                        best_path = visibility_candidate

        if best_path is None:
            return None

        simplified_path, _distance = self._finalize_route_result(
            best_path,
            current_time=current_time,
            speed=speed,
        )
        return simplified_path

    def _finalize_geometric_path(
        self,
        path: List[GeoPoint],
        *,
        current_time: Optional[float],
        speed: Optional[float],
    ) -> Tuple[List[GeoPoint], float]:
        compact_path = self._dedupe_path(path)
        if (
            len(compact_path) <= 5
            and not getattr(self.env, "has_time_dependent_blocking_constraints", False)
            and not getattr(self.env, "has_speed_factor_constraints", False)
        ):
            return compact_path, self._path_distance(compact_path)
        return self._finalize_route_result(
            compact_path,
            current_time=current_time,
            speed=speed,
        )

    def _dedupe_path(self, path: List[GeoPoint]) -> List[GeoPoint]:
        if not path:
            return []
        compact = [path[0]]
        for point in path[1:]:
            if self._same_point(compact[-1], point):
                continue
            compact.append(point)
        return compact

    def _build_local_visibility_path(
        self,
        anchor_in: GeoPoint,
        anchor_out: GeoPoint,
        *,
        blocker_bounds_list: List[Dict[str, float]],
        current_time: Optional[float],
    ) -> Optional[List[GeoPoint]]:
        current_z = self._clamp(float(anchor_in.z), 0.0, float(self.env.limit_z))
        nodes: list[GeoPoint] = [anchor_in, anchor_out]
        seen = {self._quantize_point(anchor_in), self._quantize_point(anchor_out)}

        for bounds in blocker_bounds_list:
            if not (float(bounds["min_z"]) <= current_z <= float(bounds["max_z"])):
                continue

            corner_points = [
                GeoPoint(bounds["min_x"] - self.margin, bounds["min_y"] - self.margin, current_z),
                GeoPoint(bounds["min_x"] - self.margin, bounds["max_y"] + self.margin, current_z),
                GeoPoint(bounds["max_x"] + self.margin, bounds["min_y"] - self.margin, current_z),
                GeoPoint(bounds["max_x"] + self.margin, bounds["max_y"] + self.margin, current_z),
            ]
            for point in corner_points:
                candidate = self._clamp_point(point)
                key = self._quantize_point(candidate)
                if key in seen or self.env.is_collision(candidate, current_time=current_time):
                    continue
                seen.add(key)
                nodes.append(candidate)

        if len(nodes) <= 2:
            return None

        adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(nodes))]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if not self._segment_clear(
                    nodes[i],
                    nodes[j],
                    start_time=current_time,
                    end_time=current_time,
                ):
                    continue
                weight = self._distance_between(nodes[i], nodes[j])
                adjacency[i].append((j, weight))
                adjacency[j].append((i, weight))

        return self._shortest_visible_path(nodes, adjacency)

    def _shortest_visible_path(
        self,
        nodes: List[GeoPoint],
        adjacency: List[List[Tuple[int, float]]],
    ) -> Optional[List[GeoPoint]]:
        target = 1
        distances = [float("inf")] * len(nodes)
        previous = [-1] * len(nodes)
        distances[0] = 0.0
        heap: list[tuple[float, int]] = [(0.0, 0)]

        while heap:
            current_distance, node_index = heapq.heappop(heap)
            if current_distance > distances[node_index]:
                continue
            if node_index == target:
                break

            for neighbor_index, edge_weight in adjacency[node_index]:
                new_distance = current_distance + edge_weight
                if new_distance >= distances[neighbor_index]:
                    continue
                distances[neighbor_index] = new_distance
                previous[neighbor_index] = node_index
                heapq.heappush(heap, (new_distance, neighbor_index))

        if not math.isfinite(distances[target]):
            return None

        indices = [target]
        cursor = target
        while previous[cursor] != -1:
            cursor = previous[cursor]
            indices.append(cursor)
        indices.reverse()
        return [nodes[index] for index in indices]

    def _collect_local_blocker_bounds(
        self,
        anchor_in: GeoPoint,
        anchor_out: GeoPoint,
        *,
        base_bounds: Dict[str, float],
        current_time: Optional[float],
    ) -> List[Dict[str, float]]:
        min_x = min(float(anchor_in.x), float(anchor_out.x), float(base_bounds["min_x"])) - self.margin * 2.0
        max_x = max(float(anchor_in.x), float(anchor_out.x), float(base_bounds["max_x"])) + self.margin * 2.0
        min_y = min(float(anchor_in.y), float(anchor_out.y), float(base_bounds["min_y"])) - self.margin * 2.0
        max_y = max(float(anchor_in.y), float(anchor_out.y), float(base_bounds["max_y"])) + self.margin * 2.0

        bounds_list = [base_bounds]
        seen = {
            (
                round(float(base_bounds["min_x"]), 4),
                round(float(base_bounds["min_y"]), 4),
                round(float(base_bounds["max_x"]), 4),
                round(float(base_bounds["max_y"]), 4),
                round(float(base_bounds["min_z"]), 4),
                round(float(base_bounds["max_z"]), 4),
            )
        }

        for constraint in self.env.constraints:
            if not constraint.blocks_flight_at(current_time):
                continue
            bounds = self._constraint_bounds(constraint)
            if bounds is None:
                continue
            if bounds["max_x"] < min_x or bounds["min_x"] > max_x:
                continue
            if bounds["max_y"] < min_y or bounds["min_y"] > max_y:
                continue

            key = (
                round(float(bounds["min_x"]), 4),
                round(float(bounds["min_y"]), 4),
                round(float(bounds["max_x"]), 4),
                round(float(bounds["max_y"]), 4),
                round(float(bounds["min_z"]), 4),
                round(float(bounds["max_z"]), 4),
            )
            if key in seen:
                continue
            seen.add(key)
            bounds_list.append(bounds)

        return bounds_list

    def _select_best_waypoint(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        *,
        blocker: Dict[str, object],
        current_time: Optional[float],
        speed: Optional[float],
        visited_waypoints: set[tuple[float, float, float]],
    ) -> Optional[GeoPoint]:
        best_exact: tuple[float, GeoPoint] | None = None
        best_progressive: tuple[float, GeoPoint] | None = None

        for waypoint in self._build_candidate_waypoints(blocker, start=start, goal=goal):
            waypoint_key = self._quantize_point(waypoint)
            if waypoint_key in visited_waypoints:
                continue
            if self._same_point(start, waypoint) or self._same_point(goal, waypoint):
                continue

            first_end_time = self._estimate_arrival_time(
                start,
                waypoint,
                current_time=current_time,
                speed=speed,
            )
            if not self._segment_clear(
                start,
                waypoint,
                start_time=current_time,
                end_time=first_end_time,
            ):
                continue

            second_end_time = self._estimate_arrival_time(
                waypoint,
                goal,
                current_time=first_end_time,
                speed=speed,
            )
            total_distance = self._distance_between(start, waypoint) + self._distance_between(waypoint, goal)

            if self._segment_clear(
                waypoint,
                goal,
                start_time=first_end_time,
                end_time=second_end_time,
            ):
                if best_exact is None or total_distance < best_exact[0]:
                    best_exact = (total_distance, waypoint)
                continue

            if best_progressive is None or total_distance < best_progressive[0]:
                best_progressive = (total_distance, waypoint)

        if best_exact is not None:
            return best_exact[1]
        if best_progressive is not None:
            return best_progressive[1]
        return None

    def _build_candidate_waypoints(
        self,
        blocker: Dict[str, object],
        *,
        start: GeoPoint,
        goal: GeoPoint,
    ) -> List[GeoPoint]:
        constraint = blocker["constraint"]
        midpoint = blocker["midpoint"]
        current_time = blocker.get("current_time")
        bounds = blocker.get("bounds") or self._constraint_bounds(constraint)
        if bounds is None:
            return []

        candidates: list[tuple[float, GeoPoint]] = []
        current_z = self._clamp(float(midpoint.z), 0.0, float(self.env.limit_z))

        over_z = bounds["max_z"] + self.margin
        if over_z < self.env.limit_z:
            over = self._push_candidate_to_free_space(
                GeoPoint(midpoint.x, midpoint.y, over_z),
                direction=(0.0, 0.0, 1.0),
                current_time=current_time,
            )
            if over is not None:
                detour = self._distance_between(midpoint, over)
                candidates.append((detour, over))

        dir_x = float(goal.x) - float(start.x)
        dir_y = float(goal.y) - float(start.y)
        norm_xy = math.hypot(dir_x, dir_y)

        if norm_xy <= 1e-6:
            normal_specs = [
                (GeoPoint(midpoint.x - self.margin, midpoint.y, current_z), (-1.0, 0.0, 0.0)),
                (GeoPoint(midpoint.x + self.margin, midpoint.y, current_z), (1.0, 0.0, 0.0)),
            ]
        else:
            normal_x = -dir_y / norm_xy
            normal_y = dir_x / norm_xy
            lateral_span = max(
                bounds["max_x"] - bounds["min_x"],
                bounds["max_y"] - bounds["min_y"],
            ) / 2.0 + self.margin
            normal_specs = [
                (
                    GeoPoint(
                        midpoint.x + normal_x * lateral_span,
                        midpoint.y + normal_y * lateral_span,
                        current_z,
                    ),
                    (normal_x, normal_y, 0.0),
                ),
                (
                    GeoPoint(
                        midpoint.x - normal_x * lateral_span,
                        midpoint.y - normal_y * lateral_span,
                        current_z,
                    ),
                    (-normal_x, -normal_y, 0.0),
                ),
            ]

        for raw_point, direction in normal_specs:
            candidate = self._push_candidate_to_free_space(
                raw_point,
                direction=direction,
                current_time=current_time,
            )
            if candidate is not None:
                detour = self._distance_between(midpoint, candidate)
                candidates.append((detour, candidate))

        unique: list[GeoPoint] = []
        seen: set[tuple[float, float, float]] = set()
        for _detour, candidate in sorted(candidates, key=lambda item: item[0]):
            key = self._quantize_point(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def _build_safe_fallback_path(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        *,
        current_time: Optional[float],
        speed: Optional[float],
    ) -> Optional[List[GeoPoint]]:
        cruise_z = self._resolve_safe_altitude(start, goal)
        climb = GeoPoint(start.x, start.y, cruise_z)
        descend = GeoPoint(goal.x, goal.y, cruise_z)
        x_corner = GeoPoint(goal.x, start.y, cruise_z)
        y_corner = GeoPoint(start.x, goal.y, cruise_z)

        candidate_paths = [
            [start, climb, descend, goal],
            [start, climb, x_corner, descend, goal],
            [start, climb, y_corner, descend, goal],
        ]

        for path in candidate_paths:
            if self._path_clear(path, current_time=current_time, speed=speed):
                return path
        return None

    def _resolve_safe_altitude(self, start: GeoPoint, goal: GeoPoint) -> float:
        obstacle_top = 0.0
        for constraint in self.env.constraints:
            min_z, max_z = constraint._vertical_limits()
            if max_z <= min_z:
                continue
            obstacle_top = max(obstacle_top, float(max_z))
        upper_bound = max(1.0, float(self.env.limit_z) - 1.0)
        desired = max(float(start.z), float(goal.z), obstacle_top + self.margin)
        return self._clamp(desired, 0.0, upper_bound)

    def _path_clear(
        self,
        path: List[GeoPoint],
        *,
        current_time: Optional[float],
        speed: Optional[float],
    ) -> bool:
        segment_start_time = current_time
        for index in range(len(path) - 1):
            segment_start = path[index]
            segment_goal = path[index + 1]
            segment_end_time = self._estimate_arrival_time(
                segment_start,
                segment_goal,
                current_time=segment_start_time,
                speed=speed,
            )
            if not self._segment_clear(
                segment_start,
                segment_goal,
                start_time=segment_start_time,
                end_time=segment_end_time,
            ):
                return False
            segment_start_time = segment_end_time
        return True

    def _segment_clear(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        *,
        start_time: Optional[float],
        end_time: Optional[float],
    ) -> bool:
        return self._strict_line_of_sight(
            start,
            goal,
            start_time=start_time,
            end_time=end_time,
            sample_step=self.sample_step,
        )

    def _estimate_path_end_time(
        self,
        path: List[GeoPoint],
        *,
        current_time: Optional[float],
        speed: Optional[float],
    ) -> Optional[float]:
        speed_value = float(speed or 0.0)
        if current_time is None or speed_value <= 0.0:
            return None
        elapsed = self.env.estimate_path_travel_time(
            path,
            speed_value,
            start_time=current_time,
        )
        if not math.isfinite(float(elapsed)):
            return None
        return current_time + float(elapsed)

    def _push_candidate_to_free_space(
        self,
        point: GeoPoint,
        *,
        direction: tuple[float, float, float],
        current_time: Optional[float],
    ) -> Optional[GeoPoint]:
        candidate = self._clamp_point(point)
        max_iterations = max(4, int(math.ceil((self.margin * 3.0) / self.push_step)))

        for _ in range(max_iterations):
            if not self.env.is_collision(candidate, current_time=current_time):
                return candidate
            moved = GeoPoint(
                candidate.x + direction[0] * self.push_step,
                candidate.y + direction[1] * self.push_step,
                candidate.z + direction[2] * self.push_step,
            )
            next_candidate = self._clamp_point(moved)
            if self._same_point(candidate, next_candidate):
                break
            candidate = next_candidate
        return None

    def _segment_bounds_intersection_range(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        bounds: Dict[str, float],
    ) -> Optional[Tuple[float, float]]:
        direction = (
            float(goal.x) - float(start.x),
            float(goal.y) - float(start.y),
            float(goal.z) - float(start.z),
        )
        start_values = (float(start.x), float(start.y), float(start.z))
        min_values = (bounds["min_x"], bounds["min_y"], bounds["min_z"])
        max_values = (bounds["max_x"], bounds["max_y"], bounds["max_z"])

        t_enter = 0.0
        t_exit = 1.0

        for axis in range(3):
            delta = direction[axis]
            origin = start_values[axis]
            axis_min = min_values[axis]
            axis_max = max_values[axis]

            if abs(delta) < 1e-9:
                if origin < axis_min or origin > axis_max:
                    return None
                continue

            t1 = (axis_min - origin) / delta
            t2 = (axis_max - origin) / delta
            if t1 > t2:
                t1, t2 = t2, t1

            t_enter = max(t_enter, t1)
            t_exit = min(t_exit, t2)
            if t_enter > t_exit:
                return None

        if t_exit < 0.0 or t_enter > 1.0:
            return None

        return max(0.0, t_enter), min(1.0, t_exit)

    def _constraint_bounds(self, constraint: SpatialConstraint) -> Optional[Dict[str, float]]:
        if constraint.shape == "box" and constraint.box:
            min_x, min_y, min_z, max_x, max_y, max_z = constraint.box
        elif constraint.shape == "polygon" and constraint.polygon:
            xs = [float(x) for x, _y in constraint.polygon]
            ys = [float(y) for _x, y in constraint.polygon]
            if not xs or not ys:
                return None
            min_z, max_z = constraint._vertical_limits()
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
        elif constraint.shape == "cylinder" and constraint.cylinder:
            center_x, center_y, radius, min_z, max_z = constraint.cylinder
            min_x = float(center_x) - float(radius)
            max_x = float(center_x) + float(radius)
            min_y = float(center_y) - float(radius)
            max_y = float(center_y) + float(radius)
        else:
            return None

        min_x = float(min_x)
        max_x = float(max_x)
        min_y = float(min_y)
        max_y = float(max_y)
        min_z = float(min_z)
        max_z = float(max_z)
        return {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "min_z": min_z,
            "max_z": max_z,
            "mid_x": (min_x + max_x) / 2.0,
            "mid_y": (min_y + max_y) / 2.0,
            "mid_z": (min_z + max_z) / 2.0,
        }

    @staticmethod
    def _interpolate_point(start: GeoPoint, goal: GeoPoint, ratio: float) -> GeoPoint:
        return GeoPoint(
            x=start.x + (goal.x - start.x) * ratio,
            y=start.y + (goal.y - start.y) * ratio,
            z=start.z + (goal.z - start.z) * ratio,
        )

    @staticmethod
    def _interpolate_time(
        start_time: Optional[float],
        end_time: Optional[float],
        ratio: float,
    ) -> Optional[float]:
        if start_time is None or end_time is None:
            return start_time
        return float(start_time + (end_time - start_time) * ratio)

    def _clamp_point(self, point: GeoPoint) -> GeoPoint:
        upper_z = max(0.0, float(self.env.limit_z) - 1e-3)
        return GeoPoint(
            x=self._clamp(float(point.x), 0.0, float(self.env.limit_x)),
            y=self._clamp(float(point.y), 0.0, float(self.env.limit_y)),
            z=self._clamp(float(point.z), 0.0, upper_z),
        )

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, float(value)))

    @staticmethod
    def _quantize_point(point: GeoPoint) -> tuple[float, float, float]:
        return (round(float(point.x), 3), round(float(point.y), 3), round(float(point.z), 3))

    @staticmethod
    def _same_point(left: GeoPoint, right: GeoPoint, *, tolerance: float = 1e-6) -> bool:
        return (
            math.isclose(float(left.x), float(right.x), abs_tol=tolerance)
            and math.isclose(float(left.y), float(right.y), abs_tol=tolerance)
            and math.isclose(float(left.z), float(right.z), abs_tol=tolerance)
        )


@AlgorithmRegistry.register_path_planner("geometric_heuristic_3d")
@AlgorithmRegistry.register_path_planner("geometric_heuristic_speed_3d")
class GeometricHeuristicSpeedPlanner(GeometricHeuristicPlanner):
    """速度优先版几何规划器。"""

    def __init__(self, env, grid_resolution=None, margin: float = 20.0, max_depth: int = 12, sample_step: float | None = None):
        super().__init__(
            env,
            grid_resolution=grid_resolution,
            margin=margin,
            max_depth=max_depth,
            sample_step=sample_step,
            enable_local_visibility=False,
        )


@AlgorithmRegistry.register_path_planner("geometric_heuristic_quality_3d")
class GeometricHeuristicQualityPlanner(GeometricHeuristicPlanner):
    """质量优先版几何规划器。"""

    def __init__(self, env, grid_resolution=None, margin: float = 20.0, max_depth: int = 14, sample_step: float | None = None):
        super().__init__(
            env,
            grid_resolution=grid_resolution,
            margin=margin,
            max_depth=max_depth,
            sample_step=sample_step,
            enable_local_visibility=True,
        )
