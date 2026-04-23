from __future__ import annotations

import heapq
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from backend.algorithms.registry import AlgorithmRegistry
from backend.algorithms.scheduling.optimization.insertion import InsertionScheduler
from backend.algorithms.utils.environment_graph import (
    EnvironmentGraph,
    EnvironmentGraphBuilder,
    RouteGeometry,
)
from backend.algorithms.utils.time_expanded_network import TimeExpandedNetwork
from backend.config.config_manager import config_manager
from backend.models.domain import Drone, GeoPoint, PathPoint, Task

logger = logging.getLogger(__name__)


def _distance(a: GeoPoint, b: GeoPoint) -> float:
    return float(np.linalg.norm(np.array(a.as_tuple()) - np.array(b.as_tuple())))


@dataclass(frozen=True)
class ScheduleComponent:
    component_id: str
    tasks: tuple[Task, ...]
    group_ids: tuple[str, ...]
    task_count: int
    order_count: int
    total_weight: float
    total_service_duration: float
    priority: int
    internal_distance: float
    start_point: GeoPoint
    end_point: GeoPoint
    centroid: GeoPoint
    earliest_start: float
    latest_end: float
    penalty: float

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.id for task in self.tasks)


@dataclass(frozen=True)
class InsertionCandidate:
    drone_id: str
    position: int
    score: float
    new_cost: float


@AlgorithmRegistry.register_scheduler("milp_alns")
class MilpAlnsScheduler(InsertionScheduler):
    """
    MILP + ALNS scheduler with graph preprocessing and time-expanded evaluation.

    Dependency-linked tasks are grouped into contiguous components for
    assignment and local-search stability, while route evaluation is driven by
    explicit task-level x/y/T/E variables and graph-based timing.
    """

    def __init__(self, planner, depots, depot_mgr):
        super().__init__(planner, depots, depot_mgr)

        cfg = config_manager.get_scheduling_config("milp_alns") or {}
        collision_cfg = config_manager.get_collision_detection_config() or {}

        self.milp_time_limit = float(cfg.get("milp_time_limit", 5.0))
        self.alns_iterations = int(cfg.get("alns_iterations", 60))
        self.destroy_fraction = float(cfg.get("destroy_fraction", 0.25))
        self.max_no_improve = int(cfg.get("max_no_improve", max(10, self.alns_iterations // 3)))
        self.accept_worse_probability = float(cfg.get("accept_worse_probability", 0.05))
        self.repair_candidate_limit = int(cfg.get("repair_candidate_limit", 8))
        self.unassigned_penalty_base = float(cfg.get("unassigned_penalty", 1_000_000.0))

        self.graph_clearance = float(cfg.get("graph_clearance", 20.0))
        self.max_graph_neighbors = int(cfg.get("max_graph_neighbors", 12))
        self.time_step = float(cfg.get("time_step", collision_cfg.get("time_step", 30.0)))
        self.horizon_slack = float(cfg.get("horizon_slack", 600.0))
        self.safe_radius = float(cfg.get("safe_radius", collision_cfg.get("safe_radius", 20.0)))
        self.conflict_penalty = float(cfg.get("conflict_penalty", 5_000.0))
        self.max_layers = cfg.get("max_layers")
        self.max_layers = None if self.max_layers in (None, "", 0) else int(self.max_layers)
        self.prune_threshold = float(cfg.get("prune_threshold", 0.0))
        self.big_m_time = float(cfg.get("big_m_time", 1_000_000.0))
        self.big_m_energy = float(cfg.get("big_m_energy", 1_000_000.0))
        self.distance_weight = float(cfg.get("distance_weight", 1.0))
        self.completion_weight = float(cfg.get("completion_weight", 1.0))
        self.energy_weight = float(cfg.get("energy_weight", 0.05))
        self.service_energy_rate = float(cfg.get("service_energy_rate", 0.01))
        self.hover_energy_rate = float(cfg.get("hover_energy_rate", 0.02))

        seed = cfg.get("seed")
        self.random = random.Random(seed)
        self.max_orders_per_drone = -1

        self._graph: Optional[EnvironmentGraph] = None
        self._time_network: Optional[TimeExpandedNetwork] = None
        self._components: list[ScheduleComponent] = []
        self._component_by_id: Dict[str, ScheduleComponent] = {}
        self._task_to_component: Dict[str, str] = {}
        self._task_by_id: Dict[str, Task] = {}
        self._drone_by_id: Dict[str, Drone] = {}
        self._component_task_cache: Dict[Tuple[str, ...], Tuple[Task, ...]] = {}
        self._component_route_duration_cache: Dict[Tuple, float] = {}
        self._component_route_distance_cache: Dict[Tuple, float] = {}
        self._arc_metric_cache: Dict[Tuple, Dict[str, float]] = {}

    def plan(self, current_drones: List[Drone], unassigned_tasks: List[Task]) -> Dict[str, List[Task]]:
        self.reset_route_profile_cache()
        self._reset_component_caches()
        self._prepare_scene(current_drones, unassigned_tasks)

        if not current_drones or not unassigned_tasks:
            return {drone.id: [] for drone in current_drones}

        if not self._components:
            return {drone.id: [] for drone in current_drones}

        assigned_routes, unassigned_components = self._solve_with_milp(current_drones, self._components)
        current_routes = self._clone_routes(assigned_routes)
        current_unassigned = list(unassigned_components)

        current_routes, current_unassigned = self._repair_solution(
            current_drones,
            current_routes,
            current_unassigned,
        )

        best_routes = self._clone_routes(current_routes)
        best_unassigned = list(current_unassigned)
        best_cost = self._solution_cost(current_drones, best_routes, best_unassigned)
        current_cost = best_cost
        no_improve = 0

        for _ in range(max(1, self.alns_iterations)):
            self.planner.check_cancelled()

            candidate_routes = self._clone_routes(current_routes)
            candidate_unassigned = list(current_unassigned)
            assigned_components = self._all_assigned_components(candidate_routes)

            if assigned_components:
                destroy_count = max(
                    1,
                    min(
                        len(assigned_components),
                        int(math.ceil(len(assigned_components) * max(0.05, min(self.destroy_fraction, 1.0)))),
                    ),
                )
                if self.random.random() < 0.5:
                    removed = self._destroy_random(candidate_routes, destroy_count)
                else:
                    removed = self._destroy_worst(current_drones, candidate_routes, destroy_count)
                candidate_unassigned.extend(removed)

            candidate_routes, candidate_unassigned = self._repair_solution(
                current_drones,
                candidate_routes,
                candidate_unassigned,
            )
            self._local_swap_search(current_drones, candidate_routes)

            candidate_cost = self._solution_cost(current_drones, candidate_routes, candidate_unassigned)
            if candidate_cost + 1e-6 < current_cost:
                current_routes = candidate_routes
                current_unassigned = candidate_unassigned
                current_cost = candidate_cost
                no_improve = 0
            else:
                no_improve += 1
                if self.random.random() < self.accept_worse_probability:
                    current_routes = candidate_routes
                    current_unassigned = candidate_unassigned
                    current_cost = candidate_cost

            if candidate_cost + 1e-6 < best_cost:
                best_routes = self._clone_routes(candidate_routes)
                best_unassigned = list(candidate_unassigned)
                best_cost = candidate_cost
                no_improve = 0
            elif no_improve >= self.max_no_improve:
                current_routes = self._clone_routes(best_routes)
                current_unassigned = list(best_unassigned)
                current_cost = best_cost
                no_improve = 0

        logger.info(
            "MILP+ALNS finished: components=%s unassigned=%s cost=%.2f",
            len(self._components),
            len(best_unassigned),
            best_cost,
        )

        return {
            drone.id: self._flatten_components(best_routes.get(drone.id, []))
            for drone in current_drones
        }

    def generate_path_points(self, drone: Drone, route: List[Task]) -> List[PathPoint]:
        pts: List[PathPoint] = []
        current_time = float(drone.currentTime or 0.0)
        current_range = float(drone.remainingRange if drone.remainingRange > 0 else drone.maxRange)
        current_loc = drone.currentLocation

        if current_time == 0.0:
            actual_takeoff_time = self.depot_mgr.get_available_time(drone.depotId, current_time)
            if actual_takeoff_time > current_time:
                pts.append(
                    PathPoint(
                        location=current_loc,
                        time=current_time,
                        action="wait",
                        taskId=None,
                        remainingRange=current_range,
                    )
                )
                current_time = actual_takeoff_time
            pts.append(
                PathPoint(
                    location=current_loc,
                    time=current_time,
                    action="takeoff",
                    taskId=None,
                    remainingRange=current_range,
                )
            )
        else:
            pts.append(
                PathPoint(
                    location=current_loc,
                    time=current_time,
                    action="fly",
                    taskId=None,
                    remainingRange=current_range,
                )
            )

        for task in route:
            self.planner.check_cancelled()
            geometry = self._route_geometry(current_loc, task.location, current_time=current_time, speed=drone.speed)
            if geometry is None:
                return self._fallback_path_points(drone, route)

            segment_points = list(geometry.points)
            for index in range(len(segment_points) - 1):
                seg_start = segment_points[index]
                seg_end = segment_points[index + 1]
                seg_distance = _distance(seg_start, seg_end)
                if seg_distance <= 1e-9:
                    continue
                current_time += self._segment_travel_time(
                    seg_start,
                    seg_end,
                    speed=drone.speed,
                    current_time=current_time,
                )
                current_range -= seg_distance
                current_loc = seg_end
                pts.append(
                    PathPoint(
                        location=seg_end,
                        time=current_time,
                        action="fly",
                        taskId=None,
                        remainingRange=current_range,
                    )
                )

            wait_time = max(0.0, float(task.timeWindow[0]) - current_time)
            if wait_time > 0:
                pts.append(
                    PathPoint(
                        location=current_loc,
                        time=current_time,
                        action="wait",
                        taskId=task.id,
                        remainingRange=current_range,
                    )
                )
                current_time += wait_time

            pts.append(
                PathPoint(
                    location=current_loc,
                    time=current_time,
                    action="serve",
                    taskId=task.id,
                    remainingRange=current_range,
                )
            )
            current_time += float(task.serviceDuration)

        return_depot_id = None
        depot_loc = None
        if drone.returnToDepotRequired:
            return_choice = self._select_return_depot(
                drone,
                current_loc,
                current_time=current_time,
                speed=drone.speed,
                exact=True,
            )
            if return_choice is None:
                return self._fallback_path_points(drone, route)
            return_depot_id, depot_loc, _return_dist = return_choice

        if drone.returnToDepotRequired and current_loc.as_tuple() != depot_loc.as_tuple():
            geometry = self._route_geometry(current_loc, depot_loc, current_time=current_time, speed=drone.speed)
            if geometry is None:
                return self._fallback_path_points(drone, route)

            segment_points = list(geometry.points)
            for index in range(len(segment_points) - 1):
                seg_start = segment_points[index]
                seg_end = segment_points[index + 1]
                seg_distance = _distance(seg_start, seg_end)
                if seg_distance <= 1e-9:
                    continue
                current_time += self._segment_travel_time(
                    seg_start,
                    seg_end,
                    speed=drone.speed,
                    current_time=current_time,
                )
                current_range -= seg_distance
                current_loc = seg_end
                pts.append(
                    PathPoint(
                        location=seg_end,
                        time=current_time,
                        action="fly",
                        taskId=None,
                        remainingRange=current_range,
                    )
                )

            actual_land_time = self.depot_mgr.get_available_time(return_depot_id, current_time)
            if actual_land_time > current_time:
                pts.append(
                    PathPoint(
                        location=depot_loc,
                        time=current_time,
                        action="wait",
                        taskId="HOVER_FOR_LANDING",
                        remainingRange=current_range,
                    )
                )
                current_time = actual_land_time

            pts.append(
                PathPoint(
                    location=depot_loc,
                    time=current_time,
                    action="land",
                    taskId=None,
                    remainingRange=current_range,
                )
            )

        return pts

    def _reset_component_caches(self) -> None:
        self._component_task_cache.clear()
        self._component_route_duration_cache.clear()
        self._component_route_distance_cache.clear()
        self._arc_metric_cache.clear()

    def _prepare_scene(self, drones: Sequence[Drone], tasks: Sequence[Task]) -> None:
        self._reset_component_caches()
        self._drone_by_id = {drone.id: drone for drone in drones}
        self._task_by_id = {task.id: task for task in tasks}
        self._graph = EnvironmentGraphBuilder(
            self.planner.env,
            tasks,
            self.depots,
            graph_clearance=self.graph_clearance,
            max_neighbors=self.max_graph_neighbors,
        ).build()
        self._time_network = TimeExpandedNetwork.from_scene(
            drones=drones,
            tasks=tasks,
            time_step=self.time_step,
            horizon_slack=self.horizon_slack,
            safe_radius=self.safe_radius,
            conflict_penalty=self.conflict_penalty,
            max_layers=self.max_layers,
            prune_threshold=self.prune_threshold,
        )
        self._components = self._build_task_components(tasks)
        self._component_by_id = {component.component_id: component for component in self._components}
        self._task_to_component = {
            task.id: component.component_id
            for component in self._components
            for task in component.tasks
        }

    def _compute_route_profile(self, drone: Drone, route: List[Task]) -> Dict[str, float | bool | str | None]:
        if self._graph is None:
            return super()._compute_route_profile(drone, route)

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
        }

        load_profile = self._analyze_route_load_profile(drone, route)
        profile.update(load_profile)
        if not load_profile["valid"]:
            profile["valid"] = False
            profile["reason"] = load_profile["reason"]
            return profile

        current_time = float(drone.currentTime or 0.0)
        current_load = float(load_profile["initial_load"] or 0.0)
        current_range = float(drone.remainingRange if drone.remainingRange > 0 else drone.maxRange)
        current_loc = drone.currentLocation
        energy_needed = 0.0

        for task in route:
            self.planner.check_cancelled()

            geometry = self._route_geometry(current_loc, task.location, current_time=current_time, speed=drone.speed)
            if geometry is None:
                profile["valid"] = False
                profile["reason"] = f"unable to reach task {task.id}"
                return profile

            travel_time = geometry.travel_time_at(float(drone.speed or 0.0), start_time=current_time)
            if not math.isfinite(travel_time):
                profile["valid"] = False
                profile["reason"] = f"invalid speed for task {task.id}"
                return profile

            arrival_time = current_time + travel_time
            if arrival_time > float(task.timeWindow[1]) + 1e-6:
                profile["valid"] = False
                profile["reason"] = f"task {task.id} missed time window"
                return profile

            wait_time = max(0.0, float(task.timeWindow[0]) - arrival_time)
            current_time = arrival_time + wait_time + float(task.serviceDuration)
            current_load = self._apply_task_load_change(current_load, task)
            current_range -= geometry.distance
            energy_needed += (
                geometry.distance * float(drone.energyPerMeter or 0.0)
                + wait_time * self.hover_energy_rate
                + float(task.serviceDuration) * self.service_energy_rate
            )

            profile["total_distance"] = float(profile["total_distance"]) + geometry.distance
            profile["total_flight_time"] = float(profile["total_flight_time"]) + travel_time + wait_time + float(task.serviceDuration)
            current_loc = task.location

        if drone.returnToDepotRequired:
            return_choice = self._select_return_depot(
                drone,
                current_loc,
                current_time=current_time,
                speed=drone.speed,
                exact=False,
            )
            if return_choice is None:
                profile["valid"] = False
                profile["reason"] = f"unable to return drone {drone.id} to any depot"
                return profile

            _return_depot_id, depot_loc, _return_dist = return_choice
            geometry = self._route_geometry(current_loc, depot_loc, current_time=current_time, speed=drone.speed)
            if geometry is None:
                profile["valid"] = False
                profile["reason"] = f"unable to return drone {drone.id} to any depot"
                return profile

            travel_time = geometry.travel_time_at(float(drone.speed or 0.0), start_time=current_time)
            if not math.isfinite(travel_time):
                profile["valid"] = False
                profile["reason"] = f"invalid speed for drone {drone.id}"
                return profile

            current_time += travel_time
            current_range -= geometry.distance
            energy_needed += geometry.distance * float(drone.energyPerMeter or 0.0)
            profile["total_distance"] = float(profile["total_distance"]) + geometry.distance
            profile["total_flight_time"] = float(profile["total_flight_time"]) + travel_time

        profile["completion_time"] = current_time
        profile["remaining_range"] = current_range
        profile["energy_needed"] = energy_needed
        profile["final_load"] = current_load
        remaining_battery = None if drone.battery is None else float(drone.battery) - energy_needed
        profile["remaining_battery"] = remaining_battery

        if current_range < -1e-6:
            profile["valid"] = False
            profile["reason"] = f"drone {drone.id} range exhausted"
            return profile

        if remaining_battery is not None and remaining_battery < -1e-6:
            profile["valid"] = False
            profile["reason"] = f"drone {drone.id} battery exhausted"
            return profile

        if drone.maxFlightTime is not None and drone.maxFlightTime > 0:
            if current_time - float(drone.currentTime or 0.0) > float(drone.maxFlightTime) + 1e-6:
                profile["valid"] = False
                profile["reason"] = f"drone {drone.id} flight time exceeded"
                return profile

        return profile

    def _route_geometry(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        current_time: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> Optional[RouteGeometry]:
        if self._graph is not None:
            geometry = self._graph.shortest_route(start, goal, current_time=current_time, speed=speed)
            if geometry is not None:
                return geometry

        try:
            path, distance = self.planner.get_route(start, goal, current_time=current_time, speed=speed)
        except Exception:
            return None

        if not path or distance == float("inf"):
            return None

        return RouteGeometry(
            node_ids=tuple(f"planner::{index}" for index in range(len(path))),
            points=tuple(path),
            distance=float(distance),
            env=self.planner.env,
        )

    def _fallback_path_points(self, drone: Drone, route: Sequence[Task]) -> List[PathPoint]:
        if self._graph is None:
            return []

        pts: List[PathPoint] = []
        current_time = float(drone.currentTime or 0.0)
        current_range = float(drone.remainingRange if drone.remainingRange > 0 else drone.maxRange)
        current_loc = drone.currentLocation

        if current_time == 0.0:
            pts.append(PathPoint(location=current_loc, time=current_time, action="takeoff", taskId=None, remainingRange=current_range))
        else:
            pts.append(PathPoint(location=current_loc, time=current_time, action="fly", taskId=None, remainingRange=current_range))

        for task in route:
            geometry = self._route_geometry(current_loc, task.location, current_time=current_time, speed=drone.speed)
            if geometry is None:
                break
            for index in range(1, len(geometry.points)):
                seg_start = geometry.points[index - 1]
                seg_end = geometry.points[index]
                seg_distance = _distance(seg_start, seg_end)
                if seg_distance <= 1e-9:
                    continue
                current_time += self._segment_travel_time(
                    seg_start,
                    seg_end,
                    speed=drone.speed,
                    current_time=current_time,
                )
                current_range -= seg_distance
                current_loc = seg_end
                pts.append(PathPoint(location=seg_end, time=current_time, action="fly", taskId=None, remainingRange=current_range))

            wait_time = max(0.0, float(task.timeWindow[0]) - current_time)
            if wait_time > 0:
                pts.append(PathPoint(location=current_loc, time=current_time, action="wait", taskId=task.id, remainingRange=current_range))
                current_time += wait_time
            pts.append(PathPoint(location=current_loc, time=current_time, action="serve", taskId=task.id, remainingRange=current_range))
            current_time += float(task.serviceDuration)

        depot_loc = None
        if drone.returnToDepotRequired:
            return_choice = self._select_return_depot(
                drone,
                current_loc,
                current_time=current_time,
                speed=drone.speed,
                exact=True,
            )
            if return_choice is not None:
                _return_depot_id, depot_loc, _return_dist = return_choice

        if drone.returnToDepotRequired and depot_loc is not None and current_loc.as_tuple() != depot_loc.as_tuple():
            geometry = self._route_geometry(current_loc, depot_loc, current_time=current_time, speed=drone.speed)
            if geometry is not None:
                for index in range(1, len(geometry.points)):
                    seg_start = geometry.points[index - 1]
                    seg_end = geometry.points[index]
                    seg_distance = _distance(seg_start, seg_end)
                    if seg_distance <= 1e-9:
                        continue
                    current_time += self._segment_travel_time(
                        seg_start,
                        seg_end,
                        speed=drone.speed,
                        current_time=current_time,
                    )
                    current_range -= seg_distance
                    current_loc = seg_end
                    pts.append(PathPoint(location=seg_end, time=current_time, action="fly", taskId=None, remainingRange=current_range))
            pts.append(PathPoint(location=depot_loc, time=current_time, action="land", taskId=None, remainingRange=current_range))

        return pts

    def _build_task_components(self, tasks: Sequence[Task]) -> List[ScheduleComponent]:
        task_map = {task.id: task for task in tasks}
        adjacency: Dict[str, set[str]] = {task.id: set() for task in tasks}

        grouped: Dict[str, List[Task]] = defaultdict(list)
        for task in tasks:
            grouped[task.groupId or task.id].append(task)

        for group_tasks in grouped.values():
            ordered = sorted(group_tasks, key=self._task_sort_key)
            for left, right in zip(ordered, ordered[1:]):
                adjacency[left.id].add(right.id)
                adjacency[right.id].add(left.id)

        for task in tasks:
            for dependency_id in task.dependencies or []:
                if dependency_id in adjacency:
                    adjacency[task.id].add(dependency_id)
                    adjacency[dependency_id].add(task.id)

        components: List[List[Task]] = []
        visited: set[str] = set()
        for task in sorted(tasks, key=self._task_sort_key):
            if task.id in visited:
                continue
            stack = [task.id]
            visited.add(task.id)
            component: List[Task] = []
            while stack:
                node = stack.pop()
                component.append(task_map[node])
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            components.append(component)

        schedule_components: List[ScheduleComponent] = []
        for index, component in enumerate(
            sorted(
                components,
                key=lambda item: (
                    -max(task.priority for task in item),
                    min(task.timeWindow[1] for task in item),
                    min(task.id for task in item),
                ),
            )
        ):
            ordered = self._topologically_order_component(component)
            schedule_components.append(self._make_component(index, ordered))

        return schedule_components

    def _topologically_order_component(self, tasks: Sequence[Task]) -> List[Task]:
        task_map = {task.id: task for task in tasks}
        successors: Dict[str, set[str]] = {task.id: set() for task in tasks}
        indegree: Dict[str, int] = {task.id: 0 for task in tasks}

        grouped: Dict[str, List[Task]] = defaultdict(list)
        for task in tasks:
            grouped[task.groupId or task.id].append(task)

        for group_tasks in grouped.values():
            ordered = sorted(group_tasks, key=self._task_sort_key)
            for left, right in zip(ordered, ordered[1:]):
                if right.id not in successors[left.id]:
                    successors[left.id].add(right.id)
                    indegree[right.id] += 1

        component_ids = set(task_map.keys())
        for task in tasks:
            for dependency_id in task.dependencies or []:
                if dependency_id in component_ids and task.id not in successors[dependency_id]:
                    successors[dependency_id].add(task.id)
                    indegree[task.id] += 1

        queue: List[Tuple[Tuple, str]] = []
        for task_id, count in indegree.items():
            if count == 0:
                heapq.heappush(queue, (self._task_priority_key(task_map[task_id]), task_id))

        ordered: List[Task] = []
        while queue:
            _, task_id = heapq.heappop(queue)
            ordered.append(task_map[task_id])
            for successor in successors[task_id]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    heapq.heappush(queue, (self._task_priority_key(task_map[successor]), successor))

        if len(ordered) != len(tasks):
            remaining = [task_map[task_id] for task_id, count in indegree.items() if count > 0]
            ordered.extend(sorted(remaining, key=self._task_priority_key))

        return ordered

    def _make_component(self, index: int, ordered_tasks: Sequence[Task]) -> ScheduleComponent:
        task_count = len(ordered_tasks)
        group_ids = tuple(sorted({task.groupId or task.id for task in ordered_tasks}))
        total_weight = sum(task.weight for task in ordered_tasks)
        total_service_duration = sum(task.serviceDuration for task in ordered_tasks)
        priority = max((task.priority for task in ordered_tasks), default=0)
        start_point = ordered_tasks[0].location
        end_point = ordered_tasks[-1].location
        centroid = GeoPoint(
            x=sum(task.location.x for task in ordered_tasks) / task_count,
            y=sum(task.location.y for task in ordered_tasks) / task_count,
            z=sum(task.location.z for task in ordered_tasks) / task_count,
        )
        internal_distance = 0.0
        for position in range(task_count - 1):
            geometry = self._route_geometry(ordered_tasks[position].location, ordered_tasks[position + 1].location)
            internal_distance += geometry.distance if geometry is not None else _distance(
                ordered_tasks[position].location,
                ordered_tasks[position + 1].location,
            )
        earliest_start = min(task.timeWindow[0] for task in ordered_tasks)
        latest_end = min(task.timeWindow[1] for task in ordered_tasks)
        order_count = len(group_ids)
        penalty = self.unassigned_penalty_base * (
            1.0 + 0.15 * order_count + 0.05 * priority + 0.01 * task_count
        )

        return ScheduleComponent(
            component_id=f"component_{index:03d}",
            tasks=tuple(ordered_tasks),
            group_ids=group_ids,
            task_count=task_count,
            order_count=order_count,
            total_weight=total_weight,
            total_service_duration=total_service_duration,
            priority=priority,
            internal_distance=internal_distance,
            start_point=start_point,
            end_point=end_point,
            centroid=centroid,
            earliest_start=earliest_start,
            latest_end=latest_end,
            penalty=penalty,
        )

    @staticmethod
    def _task_sort_key(task: Task) -> Tuple:
        return (
            task.groupId or task.id,
            task.sequence,
            -task.priority,
            task.timeWindow[0],
            task.timeWindow[1],
            task.id,
        )

    @staticmethod
    def _task_priority_key(task: Task) -> Tuple:
        return (
            task.sequence,
            -task.priority,
            task.timeWindow[0],
            task.timeWindow[1],
            task.id,
        )

    def _solve_with_milp(
        self,
        drones: Sequence[Drone],
        components: Sequence[ScheduleComponent],
    ) -> Tuple[Dict[str, List[ScheduleComponent]], List[ScheduleComponent]]:
        if not components:
            return {drone.id: [] for drone in drones}, []

        exact_component_limit = int(config_manager.get_scheduling_config("milp_alns").get("milp_exact_component_limit", 6))
        exact_task_limit = int(config_manager.get_scheduling_config("milp_alns").get("milp_exact_task_limit", 12))
        total_task_count = sum(component.task_count for component in components)
        if len(components) > exact_component_limit or total_task_count > exact_task_limit:
            logger.info(
                "Skipping exact MILP due to instance size: components=%s tasks=%s limits=(%s,%s)",
                len(components),
                total_task_count,
                exact_component_limit,
                exact_task_limit,
            )
            return self._greedy_assignment(drones, components)

        try:
            model = pyo.ConcreteModel()
            drone_ids = [drone.id for drone in drones]
            component_ids = [component.component_id for component in components]
            task_ids = [task.id for component in components for task in component.tasks]

            model.K = pyo.Set(initialize=drone_ids, ordered=True)
            model.C = pyo.Set(initialize=component_ids, ordered=True)
            model.M = pyo.Set(initialize=task_ids, ordered=True)

            task_lookup = self._task_by_id
            component_lookup = {component.component_id: component for component in components}
            task_to_component = {
                task.id: component.component_id
                for component in components
                for task in component.tasks
            }

            node_info: Dict[tuple[str, str], Dict[str, object]] = {}
            arc_info: Dict[tuple[str, str, str], Dict[str, float]] = {}
            node_keys: list[tuple[str, str]] = []
            arc_keys: list[tuple[str, str, str]] = []
            start_nodes: Dict[str, str] = {}
            end_nodes: Dict[str, str] = {}

            for drone in drones:
                start_node = f"start::{drone.id}"
                end_node = f"end::{drone.id}"
                start_nodes[drone.id] = start_node
                end_nodes[drone.id] = end_node
                node_keys.extend([(drone.id, start_node), (drone.id, end_node)])
                node_info[(drone.id, start_node)] = {
                    "kind": "start",
                    "location": drone.currentLocation,
                    "task_id": None,
                }
                node_info[(drone.id, end_node)] = {
                    "kind": "end",
                    "location": self._drone_end_location(drone),
                    "task_id": None,
                }

                for task in task_lookup.values():
                    node_id = f"task::{task.id}"
                    node_keys.append((drone.id, node_id))
                    node_info[(drone.id, node_id)] = {
                        "kind": "task",
                        "location": task.location,
                        "task_id": task.id,
                    }

                allowed_nodes = [start_node] + [f"task::{task.id}" for task in task_lookup.values()] + [end_node]
                for from_node in allowed_nodes:
                    for to_node in allowed_nodes:
                        if from_node == to_node or from_node == end_node or to_node == start_node:
                            continue

                        if from_node.startswith("task::") and to_node.startswith("task::"):
                            from_task = from_node.split("::", 1)[1]
                            to_task = to_node.split("::", 1)[1]
                            if task_to_component.get(from_task) == task_to_component.get(to_task):
                                component = component_lookup[task_to_component[from_task]]
                                ordered_ids = list(component.task_ids)
                                index = ordered_ids.index(from_task)
                                if not (index + 1 < len(ordered_ids) and ordered_ids[index + 1] == to_task):
                                    continue

                        arc_keys.append((drone.id, from_node, to_node))
                        arc_info[(drone.id, from_node, to_node)] = self._arc_metrics(
                            drone,
                            node_info[(drone.id, from_node)]["location"],
                            node_info[(drone.id, to_node)]["location"],
                        )

            model.NODE = pyo.Set(initialize=node_keys, dimen=2)
            model.ARC = pyo.Set(initialize=arc_keys, dimen=3)
            model.z = pyo.Var(model.K, model.C, domain=pyo.Binary)
            model.unassigned = pyo.Var(model.C, domain=pyo.Binary)
            model.y = pyo.Var(model.K, model.M, domain=pyo.Binary)
            model.x = pyo.Var(model.ARC, domain=pyo.Binary)
            model.T = pyo.Var(model.NODE, domain=pyo.NonNegativeReals, bounds=(0.0, self.big_m_time))
            model.E = pyo.Var(model.NODE, domain=pyo.NonNegativeReals, bounds=(0.0, self.big_m_energy))
            model.constraints = pyo.ConstraintList()

            for component in components:
                model.constraints.add(
                    sum(model.z[k, component.component_id] for k in model.K) + model.unassigned[component.component_id] == 1
                )
                for task in component.tasks:
                    for k in model.K:
                        model.constraints.add(model.y[k, task.id] == model.z[k, component.component_id])

            for drone in drones:
                max_orders = self._max_orders_limit()
                if max_orders > 0:
                    model.constraints.add(
                        sum(component.order_count * model.z[drone.id, component.component_id] for component in components)
                        <= max_orders
                    )

            for drone in drones:
                start_node = start_nodes[drone.id]
                end_node = end_nodes[drone.id]
                model.constraints.add(
                    sum(
                        model.x[drone.id, start_node, to_node]
                        for _drone_id, from_node, to_node in model.ARC
                        if _drone_id == drone.id and from_node == start_node
                    )
                    == 1
                )
                model.constraints.add(
                    sum(
                        model.x[drone.id, from_node, end_node]
                        for _drone_id, from_node, to_node in model.ARC
                        if _drone_id == drone.id and to_node == end_node
                    )
                    == 1
                )
                model.constraints.add(model.T[drone.id, start_node] == float(drone.currentTime or 0.0))
                model.constraints.add(model.E[drone.id, start_node] == 0.0)
                if drone.maxFlightTime is not None and drone.maxFlightTime > 0:
                    model.constraints.add(model.T[drone.id, end_node] <= float(drone.currentTime or 0.0) + float(drone.maxFlightTime))
                if drone.battery is not None and drone.battery > 0:
                    model.constraints.add(model.E[drone.id, end_node] <= float(drone.battery))

            for drone in drones:
                for task in task_lookup.values():
                    task_node = f"task::{task.id}"
                    incoming = [
                        model.x[drone.id, from_node, task_node]
                        for _drone_id, from_node, to_node in model.ARC
                        if _drone_id == drone.id and to_node == task_node
                    ]
                    outgoing = [
                        model.x[drone.id, task_node, to_node]
                        for _drone_id, from_node, to_node in model.ARC
                        if _drone_id == drone.id and from_node == task_node
                    ]
                    model.constraints.add(sum(incoming) == model.y[drone.id, task.id])
                    model.constraints.add(sum(outgoing) == model.y[drone.id, task.id])

                    task_window_start, task_window_end = float(task.timeWindow[0]), float(task.timeWindow[1])
                    model.constraints.add(model.T[drone.id, task_node] >= task_window_start * model.y[drone.id, task.id])
                    model.constraints.add(
                        model.T[drone.id, task_node] <= task_window_end + self.big_m_time * (1 - model.y[drone.id, task.id])
                    )

                for component in components:
                    ordered_ids = list(component.task_ids)
                    for left, right in zip(ordered_ids, ordered_ids[1:]):
                        model.constraints.add(
                            model.x[drone.id, f"task::{left}", f"task::{right}"] == model.z[drone.id, component.component_id]
                        )

            for arc in model.ARC:
                drone_id, from_node, to_node = arc
                metrics = arc_info[arc]
                from_info = node_info[(drone_id, from_node)]
                from_kind = from_info["kind"]
                from_task_id = from_info["task_id"]
                service_duration = float(task_lookup[from_task_id].serviceDuration) if from_kind == "task" and from_task_id in task_lookup else 0.0
                service_energy = service_duration * self.service_energy_rate
                model.constraints.add(
                    model.T[drone_id, to_node]
                    >= model.T[drone_id, from_node]
                    + service_duration
                    + metrics["travel_time"]
                    - self.big_m_time * (1 - model.x[drone_id, from_node, to_node])
                )
                model.constraints.add(
                    model.E[drone_id, to_node]
                    >= model.E[drone_id, from_node]
                    + metrics["travel_energy"]
                    + service_energy
                    - self.big_m_energy * (1 - model.x[drone_id, from_node, to_node])
                )

            model.objective = pyo.Objective(
                expr=(
                    self.distance_weight * sum(arc_info[arc]["travel_distance"] * model.x[arc] for arc in model.ARC)
                    + self.completion_weight * sum(model.T[drone.id, end_nodes[drone.id]] for drone in drones)
                    + self.energy_weight * sum(model.E[drone.id, end_nodes[drone.id]] for drone in drones)
                    + sum(component.penalty * model.unassigned[component.component_id] for component in components)
                ),
                sense=pyo.minimize,
            )

            solver = pyo.SolverFactory("highs")
            if solver is None or not solver.available(False):
                raise RuntimeError("HiGHS solver is unavailable")
            solve_kwargs = {"tee": False}
            if self.milp_time_limit > 0:
                solve_kwargs["timelimit"] = float(self.milp_time_limit)

            results = solver.solve(model, **solve_kwargs)
            termination = results.solver.termination_condition
            if termination not in {
                TerminationCondition.optimal,
                TerminationCondition.feasible,
                TerminationCondition.locallyOptimal,
                TerminationCondition.maxTimeLimit,
            }:
                raise RuntimeError(f"unexpected MILP termination: {termination}")

            assigned_routes = {drone.id: [] for drone in drones}
            unassigned_components: List[ScheduleComponent] = []

            for component in components:
                assigned_drone = None
                for drone in drones:
                    if pyo.value(model.z[drone.id, component.component_id]) > 0.5:
                        assigned_drone = drone.id
                        break
                if assigned_drone is None:
                    unassigned_components.append(component)
                    continue

                task_sequence = self._extract_route_sequence_from_solution(
                    model,
                    assigned_drone,
                    tasks=list(component.tasks),
                    start_node=start_nodes[assigned_drone],
                    end_node=end_nodes[assigned_drone],
                )
                if not task_sequence:
                    task_sequence = list(component.tasks)
                assigned_routes[assigned_drone].extend(self._sequence_to_components(task_sequence))

            for drone in drones:
                assigned_routes[drone.id] = self._normalize_route_components(assigned_routes[drone.id])
            unassigned_components.extend(self._unassign_infeasible_routes(drones, assigned_routes))

            logger.info(
                "MILP solved: assigned_components=%s unassigned=%s",
                sum(len(route) for route in assigned_routes.values()),
                len(unassigned_components),
            )
            return assigned_routes, unassigned_components

        except Exception as exc:
            logger.warning("MILP stage failed, using greedy fallback: %s", exc)
            return self._greedy_assignment(drones, components)

    def _arc_metrics(
        self,
        drone: Drone,
        from_location: GeoPoint,
        to_location: GeoPoint,
    ) -> Dict[str, float]:
        cache_key = (
            drone.id,
            from_location.as_tuple(),
            to_location.as_tuple(),
            float(drone.speed or 0.0),
            float(drone.energyPerMeter or 0.0),
        )
        cached = self._arc_metric_cache.get(cache_key)
        if cached is not None:
            self.add_runtime_stat("arcMetricCacheHits", 1)
            return dict(cached)

        self.add_runtime_stat("arcMetricCacheMisses", 1)
        geometry = self._route_geometry(from_location, to_location, current_time=None, speed=drone.speed)
        if geometry is None:
            metrics = {"travel_distance": float("inf"), "travel_time": float("inf"), "travel_energy": float("inf")}
            self._arc_metric_cache[cache_key] = metrics
            return dict(metrics)

        travel_time = geometry.travel_time_at(float(drone.speed or 0.0), start_time=None)
        if not math.isfinite(travel_time):
            metrics = {"travel_distance": float("inf"), "travel_time": float("inf"), "travel_energy": float("inf")}
            self._arc_metric_cache[cache_key] = metrics
            return dict(metrics)

        metrics = {
            "travel_distance": geometry.distance,
            "travel_time": travel_time,
            "travel_energy": geometry.distance * float(drone.energyPerMeter or 0.0),
        }
        self._arc_metric_cache[cache_key] = metrics
        return dict(metrics)

    def _drone_end_location(self, drone: Drone) -> GeoPoint:
        if drone.returnToDepotRequired:
            return_choice = self._select_return_depot(
                drone,
                drone.currentLocation,
                current_time=float(drone.currentTime or 0.0),
                speed=drone.speed,
                exact=False,
            )
            if return_choice is not None:
                _return_depot_id, depot_loc, _return_dist = return_choice
                return depot_loc
        return drone.currentLocation

    def _extract_route_sequence_from_solution(
        self,
        model: pyo.ConcreteModel,
        drone_id: str,
        tasks: Sequence[Task],
        start_node: str,
        end_node: str,
    ) -> List[Task]:
        selected_successor: Dict[str, str] = {}
        for _drone_id, from_node, to_node in model.ARC:
            if _drone_id != drone_id:
                continue
            try:
                value = pyo.value(model.x[_drone_id, from_node, to_node])
            except Exception:
                value = None
            if value is not None and value > 0.5:
                selected_successor[from_node] = to_node

        sequence: List[Task] = []
        current = start_node
        visited: set[str] = set()
        while current != end_node:
            nxt = selected_successor.get(current)
            if nxt is None or nxt in visited:
                sequence = []
                break
            visited.add(nxt)
            if nxt.startswith("task::"):
                task_id = nxt.split("::", 1)[1]
                task = self._task_by_id.get(task_id)
                if task is not None:
                    sequence.append(task)
            current = nxt

        if sequence:
            return sequence

        assigned_tasks = []
        for task in tasks:
            try:
                value = pyo.value(model.y[drone_id, task.id])
            except Exception:
                value = None
            if value is not None and value > 0.5:
                assigned_tasks.append(task)

        return sorted(
            assigned_tasks,
            key=lambda task: (
                float(pyo.value(model.T[drone_id, f"task::{task.id}"]) or 0.0),
                task.sequence,
                task.id,
            ),
        )

    def _greedy_assignment(
        self,
        drones: Sequence[Drone],
        components: Sequence[ScheduleComponent],
    ) -> Tuple[Dict[str, List[ScheduleComponent]], List[ScheduleComponent]]:
        assigned = {drone.id: [] for drone in drones}
        unassigned: List[ScheduleComponent] = []

        for component in sorted(components, key=self._component_sort_key):
            best_choice: Optional[Tuple[float, str, int]] = None
            for drone in drones:
                route = assigned[drone.id]
                if self._max_orders_limit() > 0 and self._route_order_count(route) + component.order_count > self._max_orders_limit():
                    continue
                candidate = self._find_best_insertion(drone, route, component, use_penalty=False)
                if candidate is None:
                    continue
                if best_choice is None or candidate.score < best_choice[0]:
                    best_choice = (candidate.score, drone.id, candidate.position)

            if best_choice is None:
                unassigned.append(component)
            else:
                assigned[best_choice[1]].insert(best_choice[2], component)

        return assigned, unassigned

    def _repair_solution(
        self,
        drones: Sequence[Drone],
        routes: Dict[str, List[ScheduleComponent]],
        unassigned_components: List[ScheduleComponent],
    ) -> Tuple[Dict[str, List[ScheduleComponent]], List[ScheduleComponent]]:
        remaining = sorted(
            {component.component_id: component for component in unassigned_components}.values(),
            key=self._component_sort_key,
        )

        while remaining:
            self.planner.check_cancelled()
            best_candidate: Optional[Tuple[ScheduleComponent, InsertionCandidate]] = None
            candidates_to_check = remaining
            if self.repair_candidate_limit > 0 and len(remaining) > self.repair_candidate_limit:
                candidates_to_check = remaining[: self.repair_candidate_limit]

            for component in candidates_to_check:
                candidate = self._find_best_insertion_any_drone(drones, routes, component, use_penalty=True)
                if candidate is None:
                    continue
                if best_candidate is None or candidate.score < best_candidate[1].score:
                    best_candidate = (component, candidate)

            if best_candidate is None or best_candidate[1].score >= self.prune_threshold:
                break

            component, candidate = best_candidate
            routes[candidate.drone_id].insert(candidate.position, component)
            remaining.remove(component)

        return routes, remaining

    def _find_best_insertion_any_drone(
        self,
        drones: Sequence[Drone],
        routes: Dict[str, List[ScheduleComponent]],
        component: ScheduleComponent,
        use_penalty: bool,
    ) -> Optional[InsertionCandidate]:
        best: Optional[InsertionCandidate] = None
        for drone in drones:
            candidate = self._find_best_insertion(drone, routes[drone.id], component, use_penalty=use_penalty)
            if candidate is None:
                continue
            if best is None or candidate.score < best.score:
                best = candidate
        return best

    def _find_best_insertion(
        self,
        drone: Drone,
        route: List[ScheduleComponent],
        component: ScheduleComponent,
        use_penalty: bool,
    ) -> Optional[InsertionCandidate]:
        max_orders = self._max_orders_limit()
        if max_orders > 0 and self._route_order_count(route) + component.order_count > max_orders:
            return None

        current_cost = self._route_duration_cost(drone, route)
        if math.isinf(current_cost):
            return None

        best: Optional[InsertionCandidate] = None
        for position in range(len(route) + 1):
            candidate_route = route[:position] + [component] + route[position:]
            if max_orders > 0 and self._route_order_count(candidate_route) > max_orders:
                continue

            candidate_cost = self._route_duration_cost(drone, candidate_route)
            if math.isinf(candidate_cost):
                continue

            score = candidate_cost - current_cost
            if use_penalty:
                score -= component.penalty

            candidate = InsertionCandidate(
                drone_id=drone.id,
                position=position,
                score=score,
                new_cost=candidate_cost,
            )
            if best is None or candidate.score < best.score:
                best = candidate

        return best

    def _destroy_random(
        self,
        routes: Dict[str, List[ScheduleComponent]],
        destroy_count: int,
    ) -> List[ScheduleComponent]:
        pool = [(drone_id, component) for drone_id, route in routes.items() for component in route]
        if not pool:
            return []

        removed: List[ScheduleComponent] = []
        for drone_id, component in self.random.sample(pool, min(destroy_count, len(pool))):
            if component in routes[drone_id]:
                routes[drone_id].remove(component)
                removed.append(component)
        return removed

    def _destroy_worst(
        self,
        drones: Sequence[Drone],
        routes: Dict[str, List[ScheduleComponent]],
        destroy_count: int,
    ) -> List[ScheduleComponent]:
        scored: List[Tuple[float, str, ScheduleComponent]] = []
        drone_map = {drone.id: drone for drone in drones}

        for drone_id, route in routes.items():
            drone = drone_map[drone_id]
            base_cost = self._route_duration_cost(drone, route)
            for component in route:
                reduced_route = [item for item in route if item != component]
                reduced_cost = self._route_duration_cost(drone, reduced_route)
                scored.append((base_cost - reduced_cost, drone_id, component))

        scored.sort(key=lambda item: item[0], reverse=True)
        removed: List[ScheduleComponent] = []
        for _, drone_id, component in scored[: min(destroy_count, len(scored))]:
            if component in routes[drone_id]:
                routes[drone_id].remove(component)
                removed.append(component)
        return removed

    def _local_swap_search(self, drones: Sequence[Drone], routes: Dict[str, List[ScheduleComponent]]) -> None:
        improved = True
        while improved:
            improved = False
            for first_index in range(len(drones)):
                first = drones[first_index]
                first_route = routes[first.id]
                if not first_route:
                    continue

                first_candidates = [component for _, component in self._rank_component_removal_gains(first, first_route)[:2]]
                for second_index in range(first_index + 1, len(drones)):
                    second = drones[second_index]
                    second_route = routes[second.id]
                    if not second_route:
                        continue

                    second_candidates = [component for _, component in self._rank_component_removal_gains(second, second_route)[:2]]
                    current_cost = self._route_duration_cost(first, first_route) + self._route_duration_cost(second, second_route)

                    found_swap = None
                    for component_a in first_candidates:
                        route_a_minus = [component for component in first_route if component != component_a]
                        for component_b in second_candidates:
                            route_b_minus = [component for component in second_route if component != component_b]

                            insert_a = self._find_best_insertion(first, route_a_minus, component_b, use_penalty=False)
                            insert_b = self._find_best_insertion(second, route_b_minus, component_a, use_penalty=False)
                            if insert_a is None or insert_b is None:
                                continue

                            new_first = route_a_minus[: insert_a.position] + [component_b] + route_a_minus[insert_a.position :]
                            new_second = route_b_minus[: insert_b.position] + [component_a] + route_b_minus[insert_b.position :]
                            total_cost = self._route_duration_cost(first, new_first) + self._route_duration_cost(second, new_second)
                            if total_cost + 1e-6 < current_cost:
                                found_swap = (component_a, component_b, insert_a, insert_b)
                                break
                        if found_swap is not None:
                            break

                    if found_swap is not None:
                        component_a, component_b, insert_a, insert_b = found_swap
                        new_first_route = [component for component in first_route if component != component_a]
                        new_second_route = [component for component in second_route if component != component_b]
                        new_first_route.insert(insert_a.position, component_b)
                        new_second_route.insert(insert_b.position, component_a)
                        routes[first.id] = new_first_route
                        routes[second.id] = new_second_route
                        improved = True
                        break

                if improved:
                    break

    def _rank_component_removal_gains(
        self,
        drone: Drone,
        route: List[ScheduleComponent],
    ) -> List[Tuple[float, ScheduleComponent]]:
        if not route:
            return []

        base_cost = self._route_duration_cost(drone, route)
        gains: List[Tuple[float, ScheduleComponent]] = []
        for component in route:
            reduced = [item for item in route if item != component]
            reduced_cost = self._route_duration_cost(drone, reduced)
            gains.append((base_cost - reduced_cost, component))

        gains.sort(key=lambda item: item[0], reverse=True)
        return gains

    def _solution_cost(
        self,
        drones: Sequence[Drone],
        routes: Dict[str, List[ScheduleComponent]],
        unassigned_components: Sequence[ScheduleComponent],
    ) -> float:
        total = 0.0
        drone_map = {drone.id: drone for drone in drones}

        for drone_id, route in routes.items():
            total += self._route_duration_cost(drone_map[drone_id], route)
            total += 0.1 * self._route_distance_cost(drone_map[drone_id], route)

        if self._time_network is not None and self._graph is not None:
            flat_routes = {drone_id: self._flatten_components(route) for drone_id, route in routes.items()}
            penalty, _warnings, _conflicts = self._time_network.evaluate_routes(flat_routes, drones, self._graph)
            total += penalty

        total += sum(component.penalty for component in unassigned_components)
        return total

    def _route_duration_cost(self, drone: Drone, route: List[ScheduleComponent]) -> float:
        cache_key = self._build_component_route_cost_key(drone, route)
        cached = self._component_route_duration_cache.get(cache_key)
        if cached is not None:
            self.add_runtime_stat("componentRouteDurationCacheHits", 1)
            return cached

        self.add_runtime_stat("componentRouteDurationCacheMisses", 1)
        tasks = self._flatten_components(route)
        valid, completion_cost = self.evaluate_route_with_completion_cost(drone, tasks)
        cost = (
            float("inf")
            if not valid
            else max(0.0, completion_cost - float(drone.currentTime or 0.0))
        )
        self._component_route_duration_cache[cache_key] = cost
        return cost

    def _route_distance_cost(self, drone: Drone, route: List[ScheduleComponent]) -> float:
        cache_key = self._build_component_route_cost_key(drone, route)
        cached = self._component_route_distance_cache.get(cache_key)
        if cached is not None:
            self.add_runtime_stat("componentRouteDistanceCacheHits", 1)
            return cached

        self.add_runtime_stat("componentRouteDistanceCacheMisses", 1)
        tasks = self._flatten_components(route)
        valid, distance_cost = self.evaluate_route_with_distance_cost(drone, tasks)
        cost = float("inf") if not valid else distance_cost
        self._component_route_distance_cache[cache_key] = cost
        return cost

    def _route_order_count(self, route: Sequence[ScheduleComponent]) -> int:
        return sum(component.order_count for component in route)

    def _unassign_infeasible_routes(
        self,
        drones: Sequence[Drone],
        routes: Dict[str, List[ScheduleComponent]],
    ) -> List[ScheduleComponent]:
        rejected: List[ScheduleComponent] = []
        for drone in drones:
            route = routes.get(drone.id, [])
            if not route:
                continue
            if math.isinf(self._route_duration_cost(drone, route)):
                rejected.extend(route)
                routes[drone.id] = []
        return rejected

    def _flatten_components(self, route: Sequence[ScheduleComponent]) -> List[Task]:
        component_ids = tuple(component.component_id for component in route)
        cached = self._component_task_cache.get(component_ids)
        if cached is not None:
            return list(cached)

        tasks: List[Task] = []
        for component in route:
            tasks.extend(component.tasks)
        cached_tasks = tuple(tasks)
        self._component_task_cache[component_ids] = cached_tasks
        return list(cached_tasks)

    def _build_component_route_cost_key(
        self,
        drone: Drone,
        route: Sequence[ScheduleComponent],
    ) -> Tuple:
        return (
            drone.id,
            drone.currentLocation.as_tuple(),
            float(drone.currentTime or 0.0),
            float(drone.currentLoad or 0.0),
            float(drone.remainingRange if drone.remainingRange is not None else drone.maxRange or 0.0),
            None if drone.battery is None else float(drone.battery),
            float(drone.speed or 0.0),
            float(drone.capacity or 0.0),
            bool(drone.returnToDepotRequired),
            tuple(component.component_id for component in route),
        )

    def _sequence_to_components(self, tasks: Sequence[Task]) -> List[ScheduleComponent]:
        components: List[ScheduleComponent] = []
        current_component_id: Optional[str] = None
        current_tasks: List[Task] = []

        for task in tasks:
            component_id = self._task_to_component.get(task.id)
            if component_id is None:
                continue
            if current_component_id is None or component_id != current_component_id:
                if current_tasks and current_component_id is not None:
                    components.append(self._component_by_id[current_component_id])
                current_component_id = component_id
                current_tasks = [task]
            else:
                current_tasks.append(task)

        if current_tasks and current_component_id is not None:
            components.append(self._component_by_id[current_component_id])

        return components

    def _normalize_route_components(self, route: List[ScheduleComponent]) -> List[ScheduleComponent]:
        if not route:
            return []
        normalized: List[ScheduleComponent] = []
        seen: set[str] = set()
        for component in route:
            if component.component_id in seen:
                continue
            seen.add(component.component_id)
            normalized.append(component)
        return normalized

    def _all_assigned_components(self, routes: Dict[str, List[ScheduleComponent]]) -> List[ScheduleComponent]:
        components: List[ScheduleComponent] = []
        for route in routes.values():
            components.extend(route)
        return components

    def _clone_routes(self, routes: Dict[str, List[ScheduleComponent]]) -> Dict[str, List[ScheduleComponent]]:
        return {drone_id: list(route) for drone_id, route in routes.items()}

    def _max_orders_limit(self) -> int:
        value = getattr(self, "max_orders_per_drone", -1)
        if value is None:
            return -1
        try:
            return int(value)
        except Exception:
            return -1

    def _component_sort_key(self, component: ScheduleComponent) -> Tuple:
        return (
            -component.priority,
            component.latest_end,
            -component.task_count,
            component.earliest_start,
            component.component_id,
        )


    def _solve_with_milp(
        self,
        drones: Sequence[Drone],
        components: Sequence[ScheduleComponent],
    ) -> Tuple[Dict[str, List[ScheduleComponent]], List[ScheduleComponent]]:
        if not components:
            return {drone.id: [] for drone in drones}, []

        exact_component_limit = int(config_manager.get_scheduling_config("milp_alns").get("milp_exact_component_limit", 6))
        exact_task_limit = int(config_manager.get_scheduling_config("milp_alns").get("milp_exact_task_limit", 12))
        total_task_count = sum(component.task_count for component in components)
        if len(components) > exact_component_limit or total_task_count > exact_task_limit:
            logger.info(
                "Skipping exact MILP due to instance size: components=%s tasks=%s limits=(%s,%s)",
                len(components),
                total_task_count,
                exact_component_limit,
                exact_task_limit,
            )
            return self._greedy_assignment(drones, components)

        try:
            model = pyo.ConcreteModel()
            drone_ids = [drone.id for drone in drones]
            component_ids = [component.component_id for component in components]
            task_ids = [task.id for component in components for task in component.tasks]

            model.K = pyo.Set(initialize=drone_ids, ordered=True)
            model.C = pyo.Set(initialize=component_ids, ordered=True)
            model.M = pyo.Set(initialize=task_ids, ordered=True)

            task_lookup = self._task_by_id
            component_lookup = {component.component_id: component for component in components}
            task_to_component = {
                task.id: component.component_id
                for component in components
                for task in component.tasks
            }

            node_info: Dict[tuple[str, str], Dict[str, object]] = {}
            arc_info: Dict[tuple[str, str, str], Dict[str, float]] = {}
            node_keys: list[tuple[str, str]] = []
            arc_keys: list[tuple[str, str, str]] = []
            start_nodes: Dict[str, str] = {}
            end_nodes: Dict[str, str] = {}

            for drone in drones:
                start_node = f"start::{drone.id}"
                end_node = f"end::{drone.id}"
                start_nodes[drone.id] = start_node
                end_nodes[drone.id] = end_node
                node_keys.extend([(drone.id, start_node), (drone.id, end_node)])
                node_info[(drone.id, start_node)] = {"kind": "start", "location": drone.currentLocation, "task_id": None}
                node_info[(drone.id, end_node)] = {"kind": "end", "location": self._drone_end_location(drone), "task_id": None}

                for task in task_lookup.values():
                    node_id = f"task::{task.id}"
                    node_keys.append((drone.id, node_id))
                    node_info[(drone.id, node_id)] = {"kind": "task", "location": task.location, "task_id": task.id}

                allowed_nodes = [start_node] + [f"task::{task.id}" for task in task_lookup.values()] + [end_node]
                for from_node in allowed_nodes:
                    for to_node in allowed_nodes:
                        if from_node == to_node or from_node == end_node or to_node == start_node:
                            continue

                        if from_node.startswith("task::") and to_node.startswith("task::"):
                            from_task = from_node.split("::", 1)[1]
                            to_task = to_node.split("::", 1)[1]
                            if task_to_component.get(from_task) == task_to_component.get(to_task):
                                component = component_lookup[task_to_component[from_task]]
                                ordered_ids = list(component.task_ids)
                                index = ordered_ids.index(from_task)
                                if not (index + 1 < len(ordered_ids) and ordered_ids[index + 1] == to_task):
                                    continue

                        arc_keys.append((drone.id, from_node, to_node))
                        arc_info[(drone.id, from_node, to_node)] = self._arc_metrics(
                            drone,
                            node_info[(drone.id, from_node)]["location"],
                            node_info[(drone.id, to_node)]["location"],
                        )

            model.NODE = pyo.Set(initialize=node_keys, dimen=2)
            model.ARC = pyo.Set(initialize=arc_keys, dimen=3)
            model.z = pyo.Var(model.K, model.C, domain=pyo.Binary)
            model.unassigned = pyo.Var(model.C, domain=pyo.Binary)
            model.y = pyo.Var(model.K, model.M, domain=pyo.Binary)
            model.x = pyo.Var(model.ARC, domain=pyo.Binary)
            model.T = pyo.Var(model.NODE, domain=pyo.NonNegativeReals, bounds=(0.0, self.big_m_time))
            model.E = pyo.Var(model.NODE, domain=pyo.NonNegativeReals, bounds=(0.0, self.big_m_energy))
            model.constraints = pyo.ConstraintList()

            for component in components:
                model.constraints.add(
                    sum(model.z[k, component.component_id] for k in model.K) + model.unassigned[component.component_id] == 1
                )
                for task in component.tasks:
                    for k in model.K:
                        model.constraints.add(model.y[k, task.id] == model.z[k, component.component_id])

            for drone in drones:
                max_orders = self._max_orders_limit()
                if max_orders > 0:
                    model.constraints.add(
                        sum(component.order_count * model.z[drone.id, component.component_id] for component in components)
                        <= max_orders
                    )

            for drone in drones:
                start_node = start_nodes[drone.id]
                end_node = end_nodes[drone.id]
                model.constraints.add(
                    sum(
                        model.x[drone.id, start_node, to_node]
                        for _drone_id, from_node, to_node in model.ARC
                        if _drone_id == drone.id and from_node == start_node
                    )
                    == 1
                )
                model.constraints.add(
                    sum(
                        model.x[drone.id, from_node, end_node]
                        for _drone_id, from_node, to_node in model.ARC
                        if _drone_id == drone.id and to_node == end_node
                    )
                    == 1
                )
                model.constraints.add(model.T[drone.id, start_node] == float(drone.currentTime or 0.0))
                model.constraints.add(model.E[drone.id, start_node] == 0.0)
                if drone.maxFlightTime is not None and drone.maxFlightTime > 0:
                    model.constraints.add(model.T[drone.id, end_node] <= float(drone.currentTime or 0.0) + float(drone.maxFlightTime))
                if drone.battery is not None and drone.battery > 0:
                    model.constraints.add(model.E[drone.id, end_node] <= float(drone.battery))

            for drone in drones:
                for task in task_lookup.values():
                    task_node = f"task::{task.id}"
                    incoming = [
                        model.x[drone.id, from_node, task_node]
                        for _drone_id, from_node, to_node in model.ARC
                        if _drone_id == drone.id and to_node == task_node
                    ]
                    outgoing = [
                        model.x[drone.id, task_node, to_node]
                        for _drone_id, from_node, to_node in model.ARC
                        if _drone_id == drone.id and from_node == task_node
                    ]
                    model.constraints.add(sum(incoming) == model.y[drone.id, task.id])
                    model.constraints.add(sum(outgoing) == model.y[drone.id, task.id])

                    task_window_start, task_window_end = float(task.timeWindow[0]), float(task.timeWindow[1])
                    model.constraints.add(model.T[drone.id, task_node] >= task_window_start * model.y[drone.id, task.id])
                    model.constraints.add(
                        model.T[drone.id, task_node] <= task_window_end + self.big_m_time * (1 - model.y[drone.id, task.id])
                    )

                for component in components:
                    ordered_ids = list(component.task_ids)
                    for left, right in zip(ordered_ids, ordered_ids[1:]):
                        model.constraints.add(
                            model.x[drone.id, f"task::{left}", f"task::{right}"] == model.z[drone.id, component.component_id]
                        )

            for arc in model.ARC:
                drone_id, from_node, to_node = arc
                metrics = arc_info[arc]
                from_info = node_info[(drone_id, from_node)]
                from_kind = from_info["kind"]
                from_task_id = from_info["task_id"]
                service_duration = float(task_lookup[from_task_id].serviceDuration) if from_kind == "task" and from_task_id in task_lookup else 0.0
                service_energy = service_duration * self.service_energy_rate
                model.constraints.add(
                    model.T[drone_id, to_node]
                    >= model.T[drone_id, from_node]
                    + service_duration
                    + metrics["travel_time"]
                    - self.big_m_time * (1 - model.x[drone_id, from_node, to_node])
                )
                model.constraints.add(
                    model.E[drone_id, to_node]
                    >= model.E[drone_id, from_node]
                    + metrics["travel_energy"]
                    + service_energy
                    - self.big_m_energy * (1 - model.x[drone_id, from_node, to_node])
                )

            model.objective = pyo.Objective(
                expr=(
                    self.distance_weight * sum(arc_info[arc]["travel_distance"] * model.x[arc] for arc in model.ARC)
                    + self.completion_weight * sum(model.T[drone.id, end_nodes[drone.id]] for drone in drones)
                    + self.energy_weight * sum(model.E[drone.id, end_nodes[drone.id]] for drone in drones)
                    + sum(component.penalty * model.unassigned[component.component_id] for component in components)
                ),
                sense=pyo.minimize,
            )

            solver = pyo.SolverFactory("highs")
            if solver is None or not solver.available(False):
                raise RuntimeError("HiGHS solver is unavailable")
            solve_kwargs = {"tee": False}
            if self.milp_time_limit > 0:
                solve_kwargs["timelimit"] = float(self.milp_time_limit)

            results = solver.solve(model, **solve_kwargs)
            termination = results.solver.termination_condition
            if termination not in {
                TerminationCondition.optimal,
                TerminationCondition.feasible,
                TerminationCondition.locallyOptimal,
                TerminationCondition.maxTimeLimit,
            }:
                raise RuntimeError(f"unexpected MILP termination: {termination}")

            assigned_routes = {drone.id: [] for drone in drones}
            unassigned_components: List[ScheduleComponent] = []

            for component in components:
                assigned_drone = None
                for drone in drones:
                    if pyo.value(model.z[drone.id, component.component_id]) > 0.5:
                        assigned_drone = drone.id
                        break
                if assigned_drone is None:
                    unassigned_components.append(component)
                    continue

                task_sequence = self._extract_route_sequence_from_solution(
                    model,
                    assigned_drone,
                    tasks=list(component.tasks),
                    start_node=start_nodes[assigned_drone],
                    end_node=end_nodes[assigned_drone],
                )
                if not task_sequence:
                    task_sequence = list(component.tasks)
                assigned_routes[assigned_drone].extend(self._sequence_to_components(task_sequence))

            for drone in drones:
                assigned_routes[drone.id] = self._normalize_route_components(assigned_routes[drone.id])
            unassigned_components.extend(self._unassign_infeasible_routes(drones, assigned_routes))

            logger.info(
                "MILP solved: assigned_components=%s unassigned=%s",
                sum(len(route) for route in assigned_routes.values()),
                len(unassigned_components),
            )
            return assigned_routes, unassigned_components

        except Exception as exc:
            logger.warning("MILP stage failed, using greedy fallback: %s", exc)
            return self._greedy_assignment(drones, components)

    def _arc_metrics(
        self,
        drone: Drone,
        from_location: GeoPoint,
        to_location: GeoPoint,
    ) -> Dict[str, float]:
        cache_key = (
            drone.id,
            from_location.as_tuple(),
            to_location.as_tuple(),
            float(drone.speed or 0.0),
            float(drone.energyPerMeter or 0.0),
        )
        cached = self._arc_metric_cache.get(cache_key)
        if cached is not None:
            self.add_runtime_stat("arcMetricCacheHits", 1)
            return dict(cached)

        self.add_runtime_stat("arcMetricCacheMisses", 1)
        geometry = self._route_geometry(from_location, to_location, current_time=None, speed=drone.speed)
        if geometry is None:
            metrics = {"travel_distance": float("inf"), "travel_time": float("inf"), "travel_energy": float("inf")}
            self._arc_metric_cache[cache_key] = metrics
            return dict(metrics)

        travel_time = geometry.travel_time_at(float(drone.speed or 0.0), start_time=None)
        if not math.isfinite(travel_time):
            metrics = {"travel_distance": float("inf"), "travel_time": float("inf"), "travel_energy": float("inf")}
            self._arc_metric_cache[cache_key] = metrics
            return dict(metrics)

        metrics = {
            "travel_distance": geometry.distance,
            "travel_time": travel_time,
            "travel_energy": geometry.distance * float(drone.energyPerMeter or 0.0),
        }
        self._arc_metric_cache[cache_key] = metrics
        return dict(metrics)
