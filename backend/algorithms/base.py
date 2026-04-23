"""
算法基类模块

功能说明：
- 定义路径规划算法的基类 (BasePathPlanner)
- 定义调度算法的基类 (BaseScheduler)
- 提供约束检查方法（电池、飞行时间等）
- 所有具体算法都继承这些基类

调用关系：
base.py ← insertion.py, nearest_neighbor.py, a_star.py, dijkstra.py
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Any
from time import perf_counter
from backend.models.domain import GeoPoint, Task, Drone, PathPoint
from backend.algorithms.utils.environment import CityEnvironment
from backend.routing.types import EdgeQuery
import numpy as np
import math


# ============================================================================
# 路径规划算法基类
# ============================================================================

class BasePathPlanner(ABC):
    """
    路径规划算法基类

    功能说明：
    - 定义路径规划算法的接口
    - 所有路径规划算法都应继承此类
    - 负责在给定环境中规划从起点到终点的路径

    属性：
    - env: 城市环境对象，包含地图、障碍物等信息

    子类实现：
    - AStarPlanner: A* 算法实现
    - DijkstraPlanner: Dijkstra 算法实现
    """

    def __init__(self, env: CityEnvironment):
        """
        初始化路径规划器

        参数：
        - env: CityEnvironment 对象，包含地图和约束信息
        """
        self.env = env
        self.cancel_check = None
        self.stats = {
            "routeRequests": 0,
            "cacheHits": 0,
            "cacheMisses": 0,
        }
        self._phase_stats = {
            "scheduler": {
                "routeRequests": 0,
                "cacheHits": 0,
                "cacheMisses": 0,
                "routeTime": 0.0,
            },
            "final": {
                "routeRequests": 0,
                "cacheHits": 0,
                "cacheMisses": 0,
                "routeTime": 0.0,
            },
            "other": {
                "routeRequests": 0,
                "cacheHits": 0,
                "cacheMisses": 0,
                "routeTime": 0.0,
            },
        }
        self._measurement_phase = "other"

    def _build_cache_key(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        current_time: Optional[float],
        speed: Optional[float],
    ) -> Tuple:
        if getattr(self.env, "has_time_dependent_constraints", False):
            return (start.as_tuple(), goal.as_tuple(), current_time, speed)
        return (start.as_tuple(), goal.as_tuple(), None, None)

    def check_cancelled(self) -> None:
        if callable(self.cancel_check) and self.cancel_check():
            raise RuntimeError("DISPATCH_CANCELLED")

    def record_route_request(self) -> None:
        self.stats["routeRequests"] += 1
        self._phase_stats[self._measurement_phase]["routeRequests"] += 1

    def record_cache_hit(self) -> None:
        self.stats["cacheHits"] += 1
        self._phase_stats[self._measurement_phase]["cacheHits"] += 1

    def record_cache_miss(self) -> None:
        self.stats["cacheMisses"] += 1
        self._phase_stats[self._measurement_phase]["cacheMisses"] += 1

    def set_measurement_phase(self, phase: str | None) -> None:
        normalized = str(phase or "other").strip().lower()
        self._measurement_phase = normalized if normalized in self._phase_stats else "other"

    def start_route_timer(self) -> Tuple[str, float]:
        return self._measurement_phase, perf_counter()

    def finish_route_timer(self, token: Tuple[str, float]) -> None:
        phase, started_at = token
        normalized = phase if phase in self._phase_stats else "other"
        self._phase_stats[normalized]["routeTime"] += perf_counter() - started_at

    def get_phase_stats(self) -> Dict[str, Dict[str, float | int]]:
        return {
            phase: dict(values)
            for phase, values in self._phase_stats.items()
        }

    def get_stats(self) -> Dict[str, int]:
        scheduler = self._phase_stats["scheduler"]
        final = self._phase_stats["final"]
        return {
            **self.stats,
            "schedulerRouteRequests": int(scheduler["routeRequests"]),
            "schedulerCacheHits": int(scheduler["cacheHits"]),
            "schedulerCacheMisses": int(scheduler["cacheMisses"]),
            "finalRouteRequests": int(final["routeRequests"]),
            "finalCacheHits": int(final["cacheHits"]),
            "finalCacheMisses": int(final["cacheMisses"]),
        }

    def reset_cache(self) -> None:
        cache = getattr(self, "cache", None)
        if hasattr(cache, "clear"):
            cache.clear()

    @abstractmethod
    def get_route(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        current_time: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> Tuple[List[GeoPoint], float]:
        """
        规划从起点到终点的路径

        参数：
        - start: 起点坐标
        - goal: 终点坐标

        返回值：
        - (路径点列表, 路径距离)
        - 如果无法到达，返回 ([], float('inf'))

        实现要求：
        - 必须避开所有障碍物和禁飞区
        - 返回的路径应该是最优或接近最优的
        """
        pass

    def _distance_between(self, start: GeoPoint, goal: GeoPoint) -> float:
        return float(np.linalg.norm(np.array(start.as_tuple()) - np.array(goal.as_tuple())))

    def _segment_travel_time(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        *,
        speed: Optional[float],
        current_time: Optional[float] = None,
    ) -> float:
        speed_value = float(speed or 0.0)
        if speed_value <= 0.0:
            return float("inf")
        return float(
            self.env.estimate_segment_travel_time(
                start,
                goal,
                speed_value,
                start_time=current_time,
            )
        )

    def _estimate_arrival_time(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        *,
        current_time: Optional[float],
        speed: Optional[float],
    ) -> Optional[float]:
        if current_time is None:
            return None
        travel_time = self._segment_travel_time(
            start,
            goal,
            speed=speed,
            current_time=current_time,
        )
        if not math.isfinite(travel_time):
            return None
        return current_time + travel_time

    def _path_distance(self, path: List[GeoPoint]) -> float:
        if len(path) < 2:
            return float("inf")
        return sum(
            self._distance_between(path[index], path[index + 1])
            for index in range(len(path) - 1)
        )

    def _strict_line_of_sight(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        *,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        sample_step: float = 1.0,
    ) -> bool:
        distance = self._distance_between(start, goal)
        steps = max(3, int(math.ceil(distance / max(0.25, float(sample_step)))) + 1)
        return self.env.line_of_sight(
            start,
            goal,
            steps=steps,
            start_time=start_time,
            end_time=end_time,
        )

    def _simplify_path(
        self,
        path: List[GeoPoint],
        *,
        current_time: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> List[GeoPoint]:
        if len(path) <= 2:
            return list(path)

        cumulative_distances = [0.0]
        cumulative_times = [current_time]
        for index in range(1, len(path)):
            cumulative_distances.append(
                cumulative_distances[-1] + self._distance_between(path[index - 1], path[index])
            )
            if current_time is not None and speed and speed > 0:
                previous_time = cumulative_times[-1]
                cumulative_times.append(
                    self._estimate_arrival_time(
                        path[index - 1],
                        path[index],
                        current_time=previous_time,
                        speed=speed,
                    )
                )
            else:
                cumulative_times.append(None)

        simplified = [path[0]]
        anchor_index = 0

        while anchor_index < len(path) - 1:
            farthest_visible = anchor_index + 1
            for candidate_index in range(len(path) - 1, anchor_index, -1):
                start_time = None
                end_time = None
                if current_time is not None and speed and speed > 0:
                    start_time = cumulative_times[anchor_index]
                    end_time = cumulative_times[candidate_index]

                if self.env.line_of_sight(
                    path[anchor_index],
                    path[candidate_index],
                    start_time=start_time,
                    end_time=end_time,
                ):
                    farthest_visible = candidate_index
                    break

            simplified.append(path[farthest_visible])
            anchor_index = farthest_visible

        return simplified

    def _finalize_route_result(
        self,
        path: List[GeoPoint],
        *,
        current_time: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> Tuple[List[GeoPoint], float]:
        if len(path) < 2:
            return path, float("inf")

        simplified_path = self._simplify_path(
            path,
            current_time=current_time,
            speed=speed,
        )
        return simplified_path, self._path_distance(simplified_path)


# ============================================================================
# 调度算法基类
# ============================================================================

class BaseScheduler(ABC):
    """
    调度算法基类

    功能说明：
    - 定义调度算法的接口
    - 所有调度算法都应继承此类
    - 负责将任务分配给无人机

    属性：
    - planner: 路径规划器实例
    - depots: 站点字典 {站点ID: 坐标}
    - depot_mgr: 站点管理器，用于管理无人机排队

    子类实现：
    - InsertionHeuristicScheduler: 插入启发式调度
    - NearestNeighborScheduler: 最近邻调度
    - CapacityFirstScheduler: 容量优先调度
    - DistanceFirstScheduler: 距离优先调度
    """

    def __init__(self, planner: BasePathPlanner, depots: Dict[str, GeoPoint], depot_mgr):
        """
        初始化调度器

        参数：
        - planner: 路径规划器实例
        - depots: 站点字典 {站点ID: 坐标}
        - depot_mgr: 站点管理器
        """
        self.planner = planner
        self.depots = depots  # 站点字典
        self.depot_mgr = depot_mgr  # 站点管理器
        self._route_profile_cache: Dict[Tuple, Dict[str, float | bool | str | None]] = {}
        self._exact_route_profile_cache: Dict[Tuple, Dict[str, float | bool | str | None]] = {}
        self._route_progress_cache: Dict[Tuple, Dict[str, Any]] = {}
        self._exact_route_progress_cache: Dict[Tuple, Dict[str, Any]] = {}
        self._edge_distance_cache: Dict[Tuple, Tuple[bool, float]] = {}
        self._exact_edge_distance_cache: Dict[Tuple, Tuple[bool, float]] = {}
        self.max_exact_insertions = 4
        self.max_candidate_tasks = 12
        self.max_exact_rerank_candidates = 2
        self.exact_rerank_relative_gap = 0.0
        self.exact_rerank_absolute_gap = 0.0
        self.repair_candidate_drones = 6
        self.repair_expand_candidate_drones = 12
        self.repair_global_chunk_size = 24
        self.feasible_first_mode = False
        self.coarse_schedule_ignore_blocking_zones = False
        self.trace_enabled = True
        self._planning_trace: List[Dict[str, Any]] = []
        self._trace_max_items = 60000
        self._failed_trace_sample_limit = 2
        self._failed_trace_samples: Dict[Tuple[str, str], int] = {}
        self.route_oracle = None
        self._runtime_stats: Dict[str, float | int] = {}
        self.reset_runtime_stats()

    def reset_planning_trace(self) -> None:
        self._planning_trace.clear()
        self._failed_trace_samples.clear()

    def get_planning_trace(self) -> List[Dict[str, Any]]:
        return list(self._planning_trace)

    def reset_runtime_stats(self) -> None:
        self._runtime_stats = {
            "candidateExactRerankTime": 0.0,
            "candidateExactRerankEvaluations": 0,
            "candidateExactGateTime": 0.0,
            "candidateExactGateEvaluations": 0,
            "localRepairTime": 0.0,
            "localRepairTasks": 0,
            "localRepairAssigned": 0,
            "localRepairUnresolved": 0,
            "globalRepairTime": 0.0,
            "globalRepairTasks": 0,
        }

    def get_runtime_stats(self) -> Dict[str, float | int]:
        return dict(self._runtime_stats)

    def add_runtime_stat(self, key: str, delta: float | int) -> None:
        current = self._runtime_stats.get(key, 0)
        self._runtime_stats[key] = current + delta

    def prefers_feasible_first(self) -> bool:
        return bool(
            getattr(self, "feasible_first_mode", False)
            or getattr(self, "coarse_schedule_ignore_blocking_zones", False)
        )

    def should_run_progressive_repair(self) -> bool:
        return not self.prefers_feasible_first()

    def _build_route_world_points(self, drone: Drone, route: List[Task]) -> List[Dict[str, float]]:
        points = [
            {
                "x": float(drone.currentLocation.x),
                "y": float(drone.currentLocation.y),
                "z": float(drone.currentLocation.z),
            }
        ]
        for task in route:
            points.append({
                "x": float(task.location.x),
                "y": float(task.location.y),
                "z": float(task.location.z),
            })
        return points

    def record_planning_trace(
        self,
        *,
        phase: str,
        drone: Drone,
        route: List[Task],
        valid: bool,
        reason: str = "",
        cost: float | None = None,
    ) -> None:
        if not self.trace_enabled:
            return
        if not valid:
            failure_key = (str(phase or ""), str(drone.id or ""))
            sampled = self._failed_trace_samples.get(failure_key, 0)
            if sampled >= self._failed_trace_sample_limit:
                return
            self._failed_trace_samples[failure_key] = sampled + 1
        if len(self._planning_trace) >= self._trace_max_items:
            return

        self._planning_trace.append({
            "index": len(self._planning_trace),
            "phase": phase,
            "droneId": drone.id,
            "taskIds": [task.id for task in route],
            "valid": bool(valid),
            "reason": reason or "",
            "cost": None if cost is None or not np.isfinite(cost) else float(cost),
            "points": self._build_route_world_points(drone, route),
        })

    def _build_route_profile_key(self, drone: Drone, route: List[Task]) -> Tuple:
        """
        构建路线画像缓存键。

        这里缓存的是“某架无人机在当前状态下执行某条任务序列”的整体评估结果，
        用于复用距离、时间、航程、电池等约束检查，避免对同一候选路线重复规划。
        """
        return (
            drone.id,
            drone.currentLocation.as_tuple(),
            drone.currentTime,
            drone.currentLoad,
            drone.remainingRange,
            drone.battery,
            drone.speed,
            drone.capacity,
            drone.returnToDepotRequired,
            tuple(task.id for task in route),
        )

    def _build_edge_query_key(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        *,
        current_time: float | None,
        speed: float | None,
        exact: bool = False,
    ) -> Tuple:
        planner = None
        if self.route_oracle is not None:
            planner = self.route_oracle.exact_planner if exact else self.route_oracle.estimate_planner
        if planner is None:
            planner = self.planner

        planner_env = getattr(planner, "env", None)
        if planner_env is not None and not getattr(planner_env, "has_time_dependent_constraints", False):
            current_time = None
            speed = None

        return (
            start.as_tuple(),
            goal.as_tuple(),
            None if current_time is None else round(float(current_time), 6),
            None if speed is None else round(float(speed), 6),
            id(planner),
        )

    def _segment_travel_time(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        *,
        speed: float | None,
        current_time: float | None = None,
    ) -> float:
        speed_value = float(speed or 0.0)
        if speed_value <= 0.0:
            return float("inf")
        return float(
            self.planner.env.estimate_segment_travel_time(
                start,
                goal,
                speed_value,
                start_time=current_time,
            )
        )

    def _query_edge_distance(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        *,
        current_time: float | None,
        speed: float | None,
        exact: bool = False,
    ) -> Tuple[bool, float]:
        cache = self._exact_edge_distance_cache if exact else self._edge_distance_cache
        cache_key = self._build_edge_query_key(
            start,
            goal,
            current_time=current_time,
            speed=speed,
            exact=exact,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        if exact:
            if self.route_oracle is not None:
                exact_result = self.route_oracle.solve_edge_exact(
                    EdgeQuery(
                        start=start,
                        goal=goal,
                        start_time=current_time,
                        speed=speed,
                    )
                )
                result = (bool(exact_result.feasible), float(exact_result.distance))
            else:
                _path, distance = self.planner.get_route(
                    start,
                    goal,
                    current_time=current_time,
                    speed=speed,
                )
                result = (distance != float("inf"), float(distance))
        else:
            if self.route_oracle is not None:
                estimate = self.route_oracle.estimate_edge(
                    EdgeQuery(
                        start=start,
                        goal=goal,
                        start_time=current_time,
                        speed=speed,
                    )
                )
                result = (bool(estimate.feasible), float(estimate.distance))
            else:
                _path, distance = self.planner.get_route(
                    start,
                    goal,
                    current_time=current_time,
                    speed=speed,
                )
                result = (distance != float("inf"), float(distance))

        cache[cache_key] = result
        return result

    @staticmethod
    def _same_point(left: GeoPoint, right: GeoPoint, *, tolerance: float = 1e-6) -> bool:
        return (
            math.isclose(float(left.x), float(right.x), abs_tol=tolerance)
            and math.isclose(float(left.y), float(right.y), abs_tol=tolerance)
            and math.isclose(float(left.z), float(right.z), abs_tol=tolerance)
        )

    def _require_planned_route(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        *,
        current_time: float | None,
        speed: float | None,
        label: str,
    ) -> Tuple[List[GeoPoint], float]:
        if self.route_oracle is not None:
            exact_result = self.route_oracle.solve_edge_exact(
                EdgeQuery(
                    start=start,
                    goal=goal,
                    start_time=current_time,
                    speed=speed,
                )
            )
            path = list(exact_result.path)
            distance = float(exact_result.distance)
        else:
            path, distance = self.planner.get_route(
                start,
                goal,
                current_time=current_time,
                speed=speed,
            )

        if (
            not path
            or len(path) < 2
            or not math.isfinite(float(distance))
            or float(distance) < 0.0
        ):
            raise RuntimeError(f"{label} 无法生成可行路径")

        if not self._same_point(path[0], start) or not self._same_point(path[-1], goal):
            raise RuntimeError(f"{label} 返回了不完整的路径端点")

        return path, float(distance)

    def _select_return_depot(
        self,
        drone: Drone,
        current_loc: GeoPoint,
        *,
        current_time: float | None,
        speed: float | None,
        exact: bool = False,
    ) -> Tuple[str, GeoPoint, float] | None:
        if not self.depots:
            return None

        best_choice: Tuple[str, GeoPoint, float] | None = None
        best_score: Tuple[float, int, str] | None = None

        for depot_id, depot_loc in self.depots.items():
            if self._same_point(current_loc, depot_loc):
                feasible = True
                distance = 0.0
            else:
                feasible, distance = self._query_edge_distance(
                    current_loc,
                    depot_loc,
                    current_time=current_time,
                    speed=speed,
                    exact=exact,
                )

            if not feasible or not math.isfinite(float(distance)):
                continue

            score = (
                float(distance),
                0 if depot_id == drone.depotId else 1,
                depot_id,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_choice = (depot_id, depot_loc, float(distance))

        return best_choice

    def _compute_route_profile_internal(
        self,
        drone: Drone,
        route: List[Task],
        *,
        exact: bool,
    ) -> Dict[str, float | bool | str | None]:
        """
        计算路线画像。

        画像一次性产出该路线的核心评估指标：
        - 是否可行
        - 总距离
        - 总飞行时间/完成时间
        - 所需电量
        - 是否违反时间窗、航程、返航等约束
        """
        profile: Dict[str, float | bool | str | None] = {
            "valid": True,
            "reason": "",
            "total_distance": 0.0,
            "total_flight_time": 0.0,
            "completion_time": drone.currentTime,
            "remaining_range": drone.remainingRange,
            "remaining_battery": drone.battery,
            "energy_needed": 0.0,
            "initial_load": drone.currentLoad,
            "peak_load": drone.currentLoad,
            "final_load": drone.currentLoad,
            "outbound_delivery_load": 0.0,
            "return_depot_id": None,
            "return_distance": 0.0,
        }

        load_profile = self._analyze_route_load_profile(drone, route)
        profile.update(load_profile)
        if not load_profile["valid"]:
            profile["valid"] = False
            profile["reason"] = load_profile["reason"]
            return profile

        dependency_valid, dependency_reason = self._validate_route_dependency_order(route)
        if not dependency_valid:
            profile["valid"] = False
            profile["reason"] = dependency_reason
            return profile

        progress = self._get_route_progress(drone, route, exact=exact)
        profile["total_distance"] = float(progress["total_distance"])
        profile["total_flight_time"] = float(progress["total_flight_time"])
        if not progress["valid"]:
            profile["valid"] = False
            profile["reason"] = str(progress["reason"] or "")
            return profile

        current_time = float(progress["current_time"])
        current_range = float(progress["current_range"])
        current_loc = progress["current_loc"]

        if drone.returnToDepotRequired:
            return_choice = self._select_return_depot(
                drone,
                current_loc,
                current_time=current_time,
                speed=drone.speed,
                exact=exact,
            )
            if return_choice is None:
                profile["valid"] = False
                profile["reason"] = "无法返回任何站点"
                return profile

            return_depot_id, _return_depot_loc, return_dist = return_choice
            return_time = self._segment_travel_time(
                current_loc,
                _return_depot_loc,
                speed=drone.speed,
                current_time=current_time,
            )
            current_time += return_time
            current_range -= return_dist
            profile["total_distance"] += return_dist
            profile["total_flight_time"] += return_time
            profile["return_depot_id"] = return_depot_id
            profile["return_distance"] = return_dist

        if current_range < 0:
            profile["valid"] = False
            profile["reason"] = f"无人机 {drone.id} 航程不足"
            return profile

        energy_per_meter = drone.energyPerMeter or 0.0
        energy_needed = profile["total_distance"] * energy_per_meter
        remaining_battery = None if drone.battery is None else drone.battery - energy_needed

        profile["energy_needed"] = energy_needed
        profile["remaining_battery"] = remaining_battery
        profile["remaining_range"] = current_range
        profile["completion_time"] = current_time
        profile["final_load"] = load_profile["final_load"]

        if drone.battery is not None and remaining_battery is not None and remaining_battery < 0:
            profile["valid"] = False
            profile["reason"] = f"无人机 {drone.id} 电池不足"
            return profile

        return profile

    def _compute_route_profile(self, drone: Drone, route: List[Task]) -> Dict[str, float | bool | str | None]:
        return self._compute_route_profile_internal(drone, route, exact=False)

    def _compute_exact_route_profile(self, drone: Drone, route: List[Task]) -> Dict[str, float | bool | str | None]:
        return self._compute_route_profile_internal(drone, route, exact=True)

    def get_route_profile(self, drone: Drone, route: List[Task]) -> Dict[str, float | bool | str | None]:
        """获取并缓存路线画像。"""
        cache_key = self._build_route_profile_key(drone, route)
        cached = self._route_profile_cache.get(cache_key)
        if cached is not None:
            return cached

        profile = self._compute_route_profile(drone, route)
        self._route_profile_cache[cache_key] = profile
        return profile

    def get_exact_route_profile(self, drone: Drone, route: List[Task]) -> Dict[str, float | bool | str | None]:
        cache_key = self._build_route_profile_key(drone, route)
        cached = self._exact_route_profile_cache.get(cache_key)
        if cached is not None:
            return cached

        profile = self._compute_exact_route_profile(drone, route)
        self._exact_route_profile_cache[cache_key] = profile
        return profile

    def reset_route_profile_cache(self) -> None:
        """在每轮新调度开始前清空路线画像缓存。"""
        self._route_profile_cache.clear()
        self._exact_route_profile_cache.clear()
        self._route_progress_cache.clear()
        self._exact_route_progress_cache.clear()
        self._edge_distance_cache.clear()
        self._exact_edge_distance_cache.clear()

    def _make_initial_route_progress(self, drone: Drone) -> Dict[str, Any]:
        return {
            "valid": True,
            "reason": "",
            "current_time": float(drone.currentTime),
            "current_range": float(drone.remainingRange),
            "current_loc": drone.currentLocation,
            "total_distance": 0.0,
            "total_flight_time": 0.0,
        }

    @staticmethod
    def _clone_route_progress(progress: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "valid": bool(progress["valid"]),
            "reason": str(progress.get("reason") or ""),
            "current_time": float(progress["current_time"]),
            "current_range": float(progress["current_range"]),
            "current_loc": progress["current_loc"],
            "total_distance": float(progress["total_distance"]),
            "total_flight_time": float(progress["total_flight_time"]),
        }

    def _extend_route_progress(
        self,
        drone: Drone,
        base_progress: Dict[str, Any],
        tasks: List[Task],
        *,
        exact: bool,
    ) -> Dict[str, Any]:
        progress = self._clone_route_progress(base_progress)
        if not progress["valid"] or not tasks:
            return progress

        current_time = float(progress["current_time"])
        current_range = float(progress["current_range"])
        current_loc = progress["current_loc"]
        total_distance = float(progress["total_distance"])
        total_flight_time = float(progress["total_flight_time"])

        for task in tasks:
            self.planner.check_cancelled()

            edge_ok, dist = self._query_edge_distance(
                current_loc,
                task.location,
                current_time=current_time,
                speed=drone.speed,
                exact=exact,
            )
            if not edge_ok or dist == float("inf"):
                progress["valid"] = False
                progress["reason"] = f"无法到达任务 {task.id}"
                break

            flight_time = self._segment_travel_time(
                current_loc,
                task.location,
                speed=drone.speed,
                current_time=current_time,
            )
            arrival_time = current_time + flight_time
            if arrival_time > task.timeWindow[1]:
                progress["valid"] = False
                progress["reason"] = f"任务 {task.id} 超出时间窗"
                break

            wait_time = max(0, task.timeWindow[0] - arrival_time)
            current_time = arrival_time + wait_time + task.serviceDuration
            current_range -= (dist + wait_time * 2)
            current_loc = task.location
            total_distance += float(dist)
            total_flight_time += float(flight_time + wait_time + task.serviceDuration)

        progress["current_time"] = current_time
        progress["current_range"] = current_range
        progress["current_loc"] = current_loc
        progress["total_distance"] = total_distance
        progress["total_flight_time"] = total_flight_time
        return progress

    def _get_route_progress(self, drone: Drone, route: List[Task], *, exact: bool) -> Dict[str, Any]:
        cache = self._exact_route_progress_cache if exact else self._route_progress_cache
        cache_key = self._build_route_profile_key(drone, route)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        if not route:
            progress = self._make_initial_route_progress(drone)
        else:
            prefix_progress = self._get_route_progress(drone, route[:-1], exact=exact)
            progress = self._extend_route_progress(
                drone,
                prefix_progress,
                [route[-1]],
                exact=exact,
            )

        cache[cache_key] = progress
        return progress

    def _euclidean_distance(self, start: GeoPoint, goal: GeoPoint) -> float:
        """快速几何距离估计，用于候选粗筛。"""
        return math.dist(start.as_tuple(), goal.as_tuple())

    @staticmethod
    def _task_metadata(task: Task) -> Dict[str, Any]:
        metadata = getattr(task, "metadata", None)
        return metadata if isinstance(metadata, dict) else {}

    def _task_preferred_drone_id(self, task: Task) -> str | None:
        preferred_drone_id = self._task_metadata(task).get("preferredDroneId")
        if preferred_drone_id is None:
            return None
        preferred_drone_id = str(preferred_drone_id).strip()
        return preferred_drone_id or None

    def _task_distance_to_preferred_depot(
        self,
        task: Task,
        drone: Drone,
    ) -> float:
        metadata = self._task_metadata(task)
        preferred_drone_id = self._task_preferred_drone_id(task)
        raw_distance = metadata.get("distanceToDepot")
        if raw_distance is not None and preferred_drone_id == drone.id:
            try:
                numeric_distance = float(raw_distance)
                if math.isfinite(numeric_distance):
                    return max(0.0, numeric_distance)
            except (TypeError, ValueError):
                pass

        depot_loc = self.depots.get(drone.depotId)
        if depot_loc is None:
            return self._euclidean_distance(drone.currentLocation, task.location)
        return self._euclidean_distance(depot_loc, task.location)

    def _score_task_for_drone(
        self,
        drone: Drone,
        anchor: GeoPoint,
        task: Task,
    ) -> float:
        anchor_distance = self._euclidean_distance(anchor, task.location)
        depot_distance = self._task_distance_to_preferred_depot(task, drone)
        preferred_drone_id = self._task_preferred_drone_id(task)

        # 兼顾当前航段和“回站点附近捡点”的能力，避免路线一旦离开站点就把近站点任务整体遗忘。
        score = min(anchor_distance, depot_distance * 1.15) + anchor_distance * 0.08
        if preferred_drone_id == drone.id:
            score *= 0.18
        elif preferred_drone_id is not None:
            score *= 3.5
        return float(score)

    def should_run_exact_rerank(self, estimated_increases: List[float]) -> bool:
        if len(estimated_increases) <= 1:
            return False

        ordered = [float(value) for value in estimated_increases if np.isfinite(float(value))]
        if len(ordered) <= 1:
            return False

        best = ordered[0]
        runner_up = ordered[1]
        gap = runner_up - best
        threshold = max(
            float(getattr(self, "exact_rerank_absolute_gap", 30.0)),
            abs(best) * float(getattr(self, "exact_rerank_relative_gap", 0.05)),
        )
        return gap <= threshold

    @staticmethod
    def should_skip_empty_route_exact_rerank(
        routes: Dict[str, List[Task]],
        top_candidates: List[Tuple[Any, ...]],
        *,
        drone_id_index: int,
        insert_index_index: int,
    ) -> bool:
        if len(top_candidates) <= 1:
            return False

        drone_ids = {str(candidate[drone_id_index]) for candidate in top_candidates}
        if len(drone_ids) <= 1:
            return False

        if {int(candidate[insert_index_index]) for candidate in top_candidates} != {0}:
            return False

        return all(len(routes.get(drone_id, [])) == 0 for drone_id in drone_ids)

    def _select_first_exact_feasible_insert(
        self,
        current_drones: List[Drone],
        routes: Dict[str, List[Task]],
        candidate_heap: List[Tuple[float, str, int, str, Task]],
    ) -> Tuple[Task, str, int] | None:
        if not candidate_heap:
            return None

        drone_map = {drone.id: drone for drone in current_drones}
        ordered_candidates = sorted(candidate_heap)
        started_at = perf_counter()
        exact_evaluations = 0

        try:
            for rank, (_estimated_increase, drone_id, insert_index, _task_id, task) in enumerate(ordered_candidates, start=1):
                drone = drone_map.get(drone_id)
                if drone is None:
                    continue

                current_route = routes.get(drone_id, [])
                test_route = current_route[:insert_index] + [task] + current_route[insert_index:]
                exact_evaluations += 1
                _profile, valid, exact_cost = self.evaluate_route_candidate_exact(drone, test_route)
                if not valid:
                    continue

                self.record_planning_trace(
                    phase="candidate_exact_gate",
                    drone=drone,
                    route=test_route,
                    valid=True,
                    reason=f"rank{rank} exact_gate_insert",
                    cost=exact_cost,
                )
                return (task, drone_id, insert_index)
        finally:
            self.add_runtime_stat("candidateExactGateTime", perf_counter() - started_at)
            self.add_runtime_stat("candidateExactGateEvaluations", exact_evaluations)

        return None

    def _select_first_exact_feasible_append_task(
        self,
        drone: Drone,
        current_route: List[Task],
        candidate_heap: List[Tuple[float, float, str, Task]],
    ) -> Task | None:
        if not candidate_heap:
            return None

        ordered_candidates = sorted(candidate_heap)
        started_at = perf_counter()
        exact_evaluations = 0

        try:
            for rank, (_estimated_increase, _anchor_dist, _task_id, task) in enumerate(ordered_candidates, start=1):
                test_route = current_route + [task]
                exact_evaluations += 1
                _profile, valid, exact_cost = self.evaluate_route_candidate_exact(drone, test_route)
                if not valid:
                    continue

                self.record_planning_trace(
                    phase="candidate_exact_gate",
                    drone=drone,
                    route=test_route,
                    valid=True,
                    reason=f"rank{rank} exact_gate_append",
                    cost=exact_cost,
                )
                return task
        finally:
            self.add_runtime_stat("candidateExactGateTime", perf_counter() - started_at)
            self.add_runtime_stat("candidateExactGateEvaluations", exact_evaluations)

        return None

    @staticmethod
    def _normalize_task_type(task: Task) -> str:
        task_type = str(getattr(task, "type", "") or "delivery").strip().lower()
        if task_type in {"delivery", "pickup", "inspection"}:
            return task_type
        return "delivery"

    @staticmethod
    def _task_weight(task: Task) -> float:
        return max(0.0, float(getattr(task, "weight", 0.0) or 0.0))

    def _apply_task_load_change(self, current_load: float, task: Task) -> float:
        task_type = self._normalize_task_type(task)
        task_weight = self._task_weight(task)
        if task_type == "delivery":
            return current_load - task_weight
        if task_type == "pickup":
            return current_load + task_weight
        return current_load

    def _analyze_route_load_profile(
        self,
        drone: Drone,
        route: List[Task],
    ) -> Dict[str, float | bool | str | None]:
        """
        分析路线的载荷演化。

        默认语义：
        - delivery: 起飞前预装，服务后卸载
        - pickup: 服务后装载
        - inspection: 不改变载荷
        """
        capacity = float(drone.capacity or 0.0)
        initial_carried_load = float(drone.currentLoad or 0.0)
        outbound_delivery_load = sum(
            self._task_weight(task)
            for task in route
            if self._normalize_task_type(task) == "delivery"
        )
        initial_load = initial_carried_load + outbound_delivery_load
        current_load = initial_load
        peak_load = initial_load

        if initial_load > capacity + 1e-9:
            return {
                "valid": False,
                "reason": (
                    f"无人机 {drone.id} 起飞载荷超限，需要 {initial_load:.2f}kg，"
                    f"容量 {capacity:.2f}kg"
                ),
                "initial_load": initial_load,
                "peak_load": initial_load,
                "final_load": initial_load,
                "outbound_delivery_load": outbound_delivery_load,
            }

        for task in route:
            task_type = self._normalize_task_type(task)
            task_weight = self._task_weight(task)

            if task_type == "delivery" and current_load + 1e-9 < task_weight:
                return {
                    "valid": False,
                    "reason": f"任务 {task.id} 缺少可配送载荷",
                    "initial_load": initial_load,
                    "peak_load": peak_load,
                    "final_load": current_load,
                    "outbound_delivery_load": outbound_delivery_load,
                }

            current_load = self._apply_task_load_change(current_load, task)
            peak_load = max(peak_load, current_load)

            if current_load > capacity + 1e-9:
                return {
                    "valid": False,
                    "reason": (
                        f"任务 {task.id} 后载荷超限，需要 {current_load:.2f}kg，"
                        f"容量 {capacity:.2f}kg"
                    ),
                    "initial_load": initial_load,
                    "peak_load": peak_load,
                    "final_load": current_load,
                    "outbound_delivery_load": outbound_delivery_load,
                }

        return {
            "valid": True,
            "reason": "",
            "initial_load": initial_load,
            "peak_load": peak_load,
            "final_load": current_load,
            "outbound_delivery_load": outbound_delivery_load,
        }

    @staticmethod
    def _validate_route_dependency_order(route: List[Task]) -> Tuple[bool, str]:
        """
        校验路线中的依赖顺序。

        当前调度器主要做单轮静态分配，还没有完整的跨无人机依赖时序模型。
        这里先保证同一路线内部不会把有前置依赖的任务排到依赖之前，
        这样像巡线订单这种串联目标就不会被倒序执行。
        """
        visited: set[str] = set()
        for task in route:
            dependencies = [dependency_id for dependency_id in (task.dependencies or []) if dependency_id]
            missing = [dependency_id for dependency_id in dependencies if dependency_id not in visited]
            if missing:
                return False, f"任务 {task.id} 依赖未满足: {', '.join(missing)}"
            visited.add(task.id)
        return True, ""

    def rank_insertion_positions(
        self,
        drone: Drone,
        current_route: List[Task],
        inserted_tasks: List[Task],
        limit: Optional[int] = None,
    ) -> List[int]:
        """
        使用局部几何增量对插入位置做粗筛。

        这里不调用真实 planner，只估计“把一段任务插入当前位置”会带来多大的局部路径增量。
        后续再对排名靠前的位置做精确路径规划。
        """
        if limit is None:
            limit = self.max_exact_insertions
        if not inserted_tasks:
            return [0]

        start_loc = inserted_tasks[0].location
        end_loc = inserted_tasks[-1].location
        internal_cost = 0.0
        for index in range(len(inserted_tasks) - 1):
            internal_cost += self._euclidean_distance(
                inserted_tasks[index].location,
                inserted_tasks[index + 1].location,
            )

        ranked: List[Tuple[float, int]] = []
        for i in range(len(current_route) + 1):
            prev_loc = drone.currentLocation if i == 0 else current_route[i - 1].location
            next_loc = None if i == len(current_route) else current_route[i].location

            added = self._euclidean_distance(prev_loc, start_loc) + internal_cost
            removed = 0.0
            if next_loc is not None:
                added += self._euclidean_distance(end_loc, next_loc)
                removed = self._euclidean_distance(prev_loc, next_loc)

            ranked.append((added - removed, i))

        ranked.sort(key=lambda item: item[0])
        return [idx for _, idx in ranked[: max(1, min(limit, len(ranked)))]]

    def rank_candidate_tasks(
        self,
        current_location: GeoPoint,
        tasks: List[Task],
        limit: int = 8,
        *,
        drone: Drone | None = None,
    ) -> List[Task]:
        """
        对候选任务做几何粗筛，优先精查更近的任务。
        """
        if len(tasks) <= limit:
            return list(tasks)

        if drone is None:
            return sorted(
                tasks,
                key=lambda task: self._euclidean_distance(current_location, task.location),
            )[:limit]

        return sorted(
            tasks,
            key=lambda task: self._score_task_for_drone(drone, current_location, task),
        )[:limit]

    def evaluate_route_with_completion_cost(self, drone: Drone, route: List[Task]) -> Tuple[bool, float]:
        """使用完成时间作为路线成本。"""
        _profile, valid, cost = self.evaluate_route_profile_with_completion_cost(drone, route)
        return valid, cost

    def evaluate_route_profile_with_completion_cost(
        self,
        drone: Drone,
        route: List[Task],
    ) -> Tuple[Dict[str, float | bool | str | None], bool, float]:
        """返回路线画像及其完成时间成本。"""
        profile = self.get_route_profile(drone, route)
        valid = bool(profile["valid"])
        cost = float(profile["completion_time"]) if profile["valid"] else float("inf")
        self.record_planning_trace(
            phase="candidate_evaluation",
            drone=drone,
            route=route,
            valid=valid,
            reason=str(profile.get("reason") or ""),
            cost=cost,
        )
        return profile, valid, cost

    def evaluate_route_candidate(
        self,
        drone: Drone,
        route: List[Task],
    ) -> Tuple[Dict[str, float | bool | str | None], bool, float]:
        """
        一次性完成候选路线画像、成本和约束检查，避免重复读取同一 profile。
        """
        profile, valid, cost = self.evaluate_route_profile_with_completion_cost(drone, route)
        if not valid:
            return profile, False, cost

        battery_ok, battery_msg = self.check_battery_constraint_from_profile(drone, profile)
        if not battery_ok:
            profile["valid"] = False
            profile["reason"] = battery_msg
            return profile, False, float("inf")

        time_ok, time_msg = self.check_flight_time_constraint_from_profile(drone, profile)
        if not time_ok:
            profile["valid"] = False
            profile["reason"] = time_msg
            return profile, False, float("inf")

        return profile, True, cost

    def evaluate_route_candidate_exact(
        self,
        drone: Drone,
        route: List[Task],
    ) -> Tuple[Dict[str, float | bool | str | None], bool, float]:
        profile = self.get_exact_route_profile(drone, route)
        valid = bool(profile["valid"])
        cost = float(profile["completion_time"]) if valid else float("inf")
        if not valid:
            return profile, False, cost

        battery_ok, battery_msg = self.check_battery_constraint_from_profile(drone, profile)
        if not battery_ok:
            profile["valid"] = False
            profile["reason"] = battery_msg
            return profile, False, float("inf")

        time_ok, time_msg = self.check_flight_time_constraint_from_profile(drone, profile)
        if not time_ok:
            profile["valid"] = False
            profile["reason"] = time_msg
            return profile, False, float("inf")

        return profile, True, cost

    def evaluate_route_with_distance_cost(self, drone: Drone, route: List[Task]) -> Tuple[bool, float]:
        """使用总距离作为路线成本。"""
        profile = self.get_route_profile(drone, route)
        return bool(profile["valid"]), (
            float(profile["total_distance"]) if profile["valid"] else float("inf")
        )

    @abstractmethod
    def plan(self, drones: List[Drone], tasks: List[Task]) -> Dict[str, List[Task]]:
        """
        执行调度

        参数：
        - drones: 无人机列表
        - tasks: 任务列表

        返回值：
        - 调度结果字典 {无人机ID: [任务列表]}

        实现要求：
        - 必须满足所有约束条件
        - 应该最大化无人机利用率
        - 应该最小化总飞行距离或时间
        """
        pass

    @abstractmethod
    def generate_path_points(self, drone: Drone, route: List[Task]) -> List[PathPoint]:
        """
        为无人机生成路径点

        参数：
        - drone: 无人机对象
        - route: 任务列表（已排序）

        返回值：
        - 路径点列表

        实现要求：
        - 路径点应该包含所有任务位置
        - 应该包含返回站点的路径
        - 每个路径点应该有正确的时间戳
        """
        pass

    # ========================================================================
    # 约束检查方法
    # ========================================================================

    def check_battery_constraint(self, drone: Drone, route: List[Task]) -> Tuple[bool, str]:
        """
        检查电池约束

        功能说明：
        - 检查无人机电池是否足以完成整个路线
        - 考虑飞行距离和能耗

        参数：
        - drone: 无人机对象
        - route: 任务列表

        返回值：
        - (是否满足约束, 错误信息)
        - 满足约束时返回 (True, "")
        - 不满足时返回 (False, 错误描述)

        计算逻辑：
        1. 计算从当前位置到每个任务的距离
        2. 计算返回站点的距离
        3. 计算总能耗 = 总距离 × 每米能耗
        4. 检查剩余电量是否足够
        """
        if drone.battery is None or drone.battery <= 0:
            return False, f"无人机 {drone.id} 电池为空"

        profile = self.get_route_profile(drone, route)
        return self.check_battery_constraint_from_profile(drone, profile)

    def check_flight_time_constraint(self, drone: Drone, route: List[Task]) -> Tuple[bool, str]:
        """
        检查飞行时间约束

        功能说明：
        - 检查无人机飞行时间是否超过最大限制
        - 考虑飞行时间和服务时间

        参数：
        - drone: 无人机对象
        - route: 任务列表

        返回值：
        - (是否满足约束, 错误信息)

        计算逻辑：
        1. 计算飞行时间 = 距离 / 速度
        2. 计算服务时间 = 所有任务的服务时间之和
        3. 总时间 = 飞行时间 + 服务时间
        4. 检查总时间是否超过最大飞行时间
        """
        if drone.maxFlightTime is None or drone.maxFlightTime <= 0:
            return True, ""

        profile = self.get_route_profile(drone, route)
        return self.check_flight_time_constraint_from_profile(drone, profile)

    def check_battery_constraint_from_profile(
        self,
        drone: Drone,
        profile: Dict[str, float | bool | str | None],
    ) -> Tuple[bool, str]:
        if drone.battery is None or drone.battery <= 0:
            return False, f"无人机 {drone.id} 电池为空"

        if not profile["valid"] and "电池" not in str(profile["reason"]):
            return False, str(profile["reason"])

        energy_needed = float(profile["energy_needed"])
        remaining_battery = profile["remaining_battery"]
        if remaining_battery is not None and float(remaining_battery) < 0:
            return False, f"无人机 {drone.id} 电池不足，需要 {energy_needed:.2f}%，剩余 {drone.battery:.2f}%"

        return True, ""

    def check_flight_time_constraint_from_profile(
        self,
        drone: Drone,
        profile: Dict[str, float | bool | str | None],
    ) -> Tuple[bool, str]:
        if drone.maxFlightTime is None or drone.maxFlightTime <= 0:
            return True, ""

        if not profile["valid"]:
            return False, str(profile["reason"])

        total_time = float(profile["completion_time"]) - float(drone.currentTime)
        if total_time > drone.maxFlightTime:
            return False, f"无人机 {drone.id} 飞行时间超限，需要 {total_time:.2f}s，限制 {drone.maxFlightTime:.2f}s"

        return True, ""

    def check_capacity_constraint(self, drone: Drone, route: List[Task]) -> Tuple[bool, str]:
        """
        检查容量约束

        功能说明：
        - 检查无人机载重是否超过最大容量

        参数：
        - drone: 无人机对象
        - route: 任务列表

        返回值：
        - (是否满足约束, 错误信息)

        计算逻辑：
        1. 配送任务在起飞前预装，构成初始载荷
        2. 揽收任务在服务后装载，形成过程峰值载荷
        3. 巡检任务不改变载荷
        """
        load_profile = self._analyze_route_load_profile(drone, route)
        if not load_profile["valid"]:
            return False, str(load_profile["reason"])

        return True, ""

    def check_range_constraint(self, drone: Drone, route: List[Task]) -> Tuple[bool, str]:
        """
        检查航程约束

        功能说明：
        - 检查无人机航程是否足以完成路线

        参数：
        - drone: 无人机对象
        - route: 任务列表

        返回值：
        - (是否满足约束, 错误信息)

        计算逻辑：
        1. 计算路线总距离
        2. 检查总距离是否超过无人机最大航程
        """
        profile = self.get_route_profile(drone, route)
        if not profile["valid"]:
            return False, str(profile["reason"])

        total_distance = float(profile["total_distance"])
        if total_distance > drone.maxRange:
            return False, f"无人机 {drone.id} 航程不足，需要 {total_distance:.2f}m，最大 {drone.maxRange:.2f}m"

        return True, ""

    def check_all_constraints(self, drone: Drone, route: List[Task]) -> Tuple[bool, List[str]]:
        """
        检查所有约束

        功能说明：
        - 一次性检查所有约束条件

        参数：
        - drone: 无人机对象
        - route: 任务列表

        返回值：
        - (是否满足所有约束, 错误信息列表)

        检查项：
        1. 电池约束
        2. 飞行时间约束
        3. 容量约束
        4. 航程约束
        """
        errors = []

        # 检查电池
        battery_ok, battery_msg = self.check_battery_constraint(drone, route)
        if not battery_ok:
            errors.append(battery_msg)

        # 检查飞行时间
        time_ok, time_msg = self.check_flight_time_constraint(drone, route)
        if not time_ok:
            errors.append(time_msg)

        # 检查容量
        capacity_ok, capacity_msg = self.check_capacity_constraint(drone, route)
        if not capacity_ok:
            errors.append(capacity_msg)

        # 检查航程
        range_ok, range_msg = self.check_range_constraint(drone, route)
        if not range_ok:
            errors.append(range_msg)

        return len(errors) == 0, errors
