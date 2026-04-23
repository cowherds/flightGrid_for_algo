"""
插入启发式调度器。

这一版重点优化了调度阶段的调用逻辑：
1. 候选插入位置先做几何粗筛。
2. 只有少量靠前位置才做精确路径规划。
3. 如果粗筛位置都不可行，再回退扫描剩余位置，保证可行性。
"""

import numpy as np
from typing import Dict, List, Tuple
import heapq
from time import perf_counter

from backend.algorithms.base import BaseScheduler
from backend.algorithms.registry import AlgorithmRegistry
from backend.models.domain import Drone, GeoPoint, PathPoint, Task


@AlgorithmRegistry.register_scheduler("insertion_heuristic")
class InsertionScheduler(BaseScheduler):
    def plan(self, current_drones: List[Drone], unassigned_tasks: List[Task]) -> Dict[str, List[Task]]:
        self.reset_route_profile_cache()
        self.reset_runtime_stats()
        routes = {drone.id: [] for drone in current_drones}
        self._assign_tasks_into_routes(current_drones, routes, unassigned_tasks)
        return routes

    def _seed_routes_with_local_tasks(
        self,
        current_drones: List[Drone],
        routes: Dict[str, List[Task]],
        tasks_pool: List[Task],
    ) -> None:
        if not tasks_pool:
            return

        route_costs = {drone.id: 0.0 for drone in current_drones}
        for drone in current_drones:
            if not tasks_pool or routes.get(drone.id):
                continue

            candidate_tasks = self.rank_candidate_tasks(
                drone.currentLocation,
                tasks_pool,
                limit=max(1, self.max_candidate_tasks),
                drone=drone,
            )
            if not candidate_tasks:
                continue

            preferred_task_ids = {drone.id: {task.id for task in candidate_tasks}}
            best_insert = self._find_best_insert(
                [drone],
                candidate_tasks,
                routes,
                route_costs,
                preferred_task_ids,
                preferred_only=False,
            )
            if best_insert is None:
                continue

            task, drone_id, insert_index = best_insert
            routes[drone_id].insert(insert_index, task)
            tasks_pool.remove(task)
            route_costs.pop(drone_id, None)

    def _assign_tasks_into_routes(
        self,
        current_drones: List[Drone],
        routes: Dict[str, List[Task]],
        unassigned_tasks: List[Task],
    ) -> Dict[str, List[Task]]:
        tasks_pool = sorted(unassigned_tasks.copy(), key=lambda task: task.priority, reverse=True)
        self._seed_routes_with_local_tasks(current_drones, routes, tasks_pool)

        while tasks_pool:
            self.planner.check_cancelled()
            route_costs = {
                drone.id: (self._evaluate_route(drone, routes[drone.id])[1] if routes[drone.id] else 0.0)
                for drone in current_drones
            }
            preferred_task_ids = self._build_preferred_task_sets(current_drones, routes, tasks_pool)

            best_insert = self._find_best_insert(
                current_drones,
                tasks_pool,
                routes,
                route_costs,
                preferred_task_ids,
                preferred_only=True,
            )
            if best_insert is None:
                best_insert = self._find_best_insert(
                    current_drones,
                    tasks_pool,
                    routes,
                    route_costs,
                    preferred_task_ids,
                    preferred_only=False,
                )

            if best_insert is None:
                break

            task, drone_id, insert_index = best_insert
            routes[drone_id].insert(insert_index, task)
            tasks_pool.remove(task)

        return routes

    def _repair_tasks_into_existing_routes(
        self,
        current_drones: List[Drone],
        routes: Dict[str, List[Task]],
        tasks: List[Task],
    ) -> Dict[str, List[Task]]:
        self.reset_route_profile_cache()
        return self._assign_tasks_into_routes(current_drones, routes, tasks)

    def _rank_candidate_drones_for_repair(
        self,
        current_drones: List[Drone],
        routes: Dict[str, List[Task]],
        task: Task,
        *,
        limit: int,
    ) -> List[Drone]:
        scored: List[Tuple[float, Drone]] = []
        for drone in current_drones:
            current_route = routes.get(drone.id, [])
            anchor = current_route[-1].location if current_route else drone.currentLocation
            score = self._score_task_for_drone(drone, anchor, task)
            score += max(0, len(current_route) - 4) * 3.0
            scored.append((score, drone))

        scored.sort(key=lambda item: item[0])
        return [drone for _, drone in scored[: max(1, min(limit, len(scored)))]]

    def _find_best_insert_for_single_task(
        self,
        candidate_drones: List[Drone],
        task: Task,
        routes: Dict[str, List[Task]],
        route_costs: Dict[str, float],
    ) -> Tuple[Task, str, int] | None:
        if not candidate_drones:
            return None

        best_insert = None
        min_cost_increase = float("inf")
        candidate_heap: List[Tuple[float, str, int, str, Task]] = []

        for drone in candidate_drones:
            current_route = routes[drone.id]
            old_cost = route_costs.get(drone.id)
            if old_cost is None:
                old_cost = self._evaluate_route(drone, current_route)[1] if current_route else 0.0
                route_costs[drone.id] = old_cost

            ranked_positions = self.rank_insertion_positions(drone, current_route, [task])
            remaining_positions = [
                idx for idx in range(len(current_route) + 1)
                if idx not in ranked_positions
            ]

            for position_group in (ranked_positions, remaining_positions):
                if not position_group:
                    continue

                found_valid_for_drone = False
                for insert_index in position_group:
                    test_route = (
                        current_route[:insert_index]
                        + [task]
                        + current_route[insert_index:]
                    )
                    _profile, valid, cost = self.evaluate_route_candidate(drone, test_route)
                    if not valid:
                        continue

                    cost_increase = cost - old_cost
                    if cost_increase < min_cost_increase:
                        min_cost_increase = cost_increase
                        best_insert = (task, drone.id, insert_index)
                    heapq.heappush(candidate_heap, (cost_increase, drone.id, insert_index, task.id, task))
                    found_valid_for_drone = True

                if found_valid_for_drone:
                    break

        if self.prefers_feasible_first():
            reranked = self._select_first_exact_feasible_insert(
                candidate_drones,
                routes,
                candidate_heap,
            )
            return reranked or best_insert

        reranked = self._rerank_candidates_exact(candidate_drones, routes, candidate_heap)
        return reranked or best_insert

    def _repair_tasks_locally(
        self,
        current_drones: List[Drone],
        routes: Dict[str, List[Task]],
        tasks: List[Task],
        *,
        candidate_drone_limit: int,
        expand_candidate_limit: int,
    ) -> Tuple[Dict[str, List[Task]], List[Task]]:
        if not current_drones or not tasks:
            return routes, []

        started_at = perf_counter()
        route_costs: Dict[str, float] = {}
        unresolved: List[Task] = []
        assigned_count = 0
        ordered_tasks = sorted(tasks, key=lambda task: (task.priority, task.weight), reverse=True)

        try:
            for task in ordered_tasks:
                primary_drones = self._rank_candidate_drones_for_repair(
                    current_drones,
                    routes,
                    task,
                    limit=candidate_drone_limit,
                )
                best_insert = self._find_best_insert_for_single_task(
                    primary_drones,
                    task,
                    routes,
                    route_costs,
                )

                expanded_limit = max(candidate_drone_limit, expand_candidate_limit)
                if best_insert is None and expanded_limit > len(primary_drones):
                    expanded_drones = self._rank_candidate_drones_for_repair(
                        current_drones,
                        routes,
                        task,
                        limit=expanded_limit,
                    )
                    best_insert = self._find_best_insert_for_single_task(
                        expanded_drones,
                        task,
                        routes,
                        route_costs,
                    )

                if best_insert is None:
                    unresolved.append(task)
                    continue

                _task, drone_id, insert_index = best_insert
                routes[drone_id].insert(insert_index, task)
                route_costs.pop(drone_id, None)
                assigned_count += 1
        finally:
            self.add_runtime_stat("localRepairTime", perf_counter() - started_at)
            self.add_runtime_stat("localRepairTasks", len(ordered_tasks))
            self.add_runtime_stat("localRepairAssigned", assigned_count)
            self.add_runtime_stat("localRepairUnresolved", len(unresolved))

        return routes, unresolved

    def _repair_tasks_globally_in_chunks(
        self,
        current_drones: List[Drone],
        routes: Dict[str, List[Task]],
        tasks: List[Task],
        *,
        chunk_size: int,
    ) -> Dict[str, List[Task]]:
        if not tasks:
            return routes

        started_at = perf_counter()
        try:
            ordered_tasks = sorted(tasks, key=lambda task: (task.priority, task.weight), reverse=True)
            effective_chunk_size = max(1, chunk_size)
            for start in range(0, len(ordered_tasks), effective_chunk_size):
                routes = self._repair_tasks_into_existing_routes(
                    current_drones,
                    routes,
                    ordered_tasks[start:start + effective_chunk_size],
                )
            return routes
        finally:
            self.add_runtime_stat("globalRepairTime", perf_counter() - started_at)
            self.add_runtime_stat("globalRepairTasks", len(tasks))

    def _repair_tasks_progressive(
        self,
        current_drones: List[Drone],
        routes: Dict[str, List[Task]],
        tasks: List[Task],
    ) -> Dict[str, List[Task]]:
        if not tasks:
            return routes

        if not self.should_run_progressive_repair():
            return routes

        routes, unresolved = self._repair_tasks_locally(
            current_drones,
            routes,
            tasks,
            candidate_drone_limit=max(1, int(getattr(self, "repair_candidate_drones", 6))),
            expand_candidate_limit=max(1, int(getattr(self, "repair_expand_candidate_drones", 12))),
        )
        if not unresolved:
            return routes

        broadened_limit = min(
            len(current_drones),
            max(
                int(getattr(self, "repair_expand_candidate_drones", 12)) + 4,
                int(getattr(self, "repair_candidate_drones", 6)) * 3,
            ),
        )
        if broadened_limit > int(getattr(self, "repair_expand_candidate_drones", 12)):
            routes, unresolved = self._repair_tasks_locally(
                current_drones,
                routes,
                unresolved,
                candidate_drone_limit=broadened_limit,
                expand_candidate_limit=broadened_limit,
            )
            if not unresolved:
                return routes

        return self._repair_tasks_globally_in_chunks(
            current_drones,
            routes,
            unresolved,
            chunk_size=max(1, int(getattr(self, "repair_global_chunk_size", 24))),
        )

    def _merge_ordered_tasks_for_drone(
        self,
        drone: Drone,
        base_route: List[Task],
        ordered_tasks: List[Task],
    ) -> List[Task]:
        if not ordered_tasks:
            return list(base_route)

        candidate_route = list(base_route) + list(ordered_tasks)
        _profile, valid, _cost = self.evaluate_route_candidate(drone, candidate_route)
        if valid:
            return candidate_route

        repaired_routes = self._repair_tasks_into_existing_routes(
            [drone],
            {drone.id: list(base_route)},
            list(ordered_tasks),
        )
        return repaired_routes.get(drone.id, list(base_route))

    def _build_preferred_task_sets(
        self,
        current_drones: List[Drone],
        routes: Dict[str, List[Task]],
        tasks_pool: List[Task],
    ) -> Dict[str, set[str]]:
        preferred: Dict[str, set[str]] = {}
        for drone in current_drones:
            current_route = routes[drone.id]
            anchor = current_route[-1].location if current_route else drone.currentLocation
            preferred[drone.id] = {
                task.id for task in self.rank_candidate_tasks(
                    anchor,
                    tasks_pool,
                    limit=self.max_candidate_tasks,
                    drone=drone,
                )
            }
        return preferred

    def _find_best_insert(
        self,
        current_drones: List[Drone],
        tasks_pool: List[Task],
        routes: Dict[str, List[Task]],
        route_costs: Dict[str, float],
        preferred_task_ids: Dict[str, set[str]],
        *,
        preferred_only: bool,
    ) -> Tuple[Task, str, int] | None:
        best_insert = None
        min_cost_increase = float("inf")
        candidate_heap: List[Tuple[float, str, int, str, Task]] = []

        for task in tasks_pool:
            for drone in current_drones:
                if preferred_only and task.id not in preferred_task_ids.get(drone.id, set()):
                    continue

                current_route = routes[drone.id]
                old_cost = route_costs[drone.id]
                ranked_positions = self.rank_insertion_positions(drone, current_route, [task])
                remaining_positions = [
                    idx for idx in range(len(current_route) + 1)
                    if idx not in ranked_positions
                ]

                for position_group in (ranked_positions, remaining_positions):
                    if not position_group:
                        continue

                    found_valid_for_drone = False
                    for insert_index in position_group:
                        test_route = (
                            current_route[:insert_index]
                            + [task]
                            + current_route[insert_index:]
                        )
                        _profile, valid, cost = self.evaluate_route_candidate(drone, test_route)
                        if not valid:
                            continue

                        cost_increase = cost - old_cost
                        if cost_increase < min_cost_increase:
                            min_cost_increase = cost_increase
                            best_insert = (task, drone.id, insert_index)
                        heapq.heappush(candidate_heap, (cost_increase, drone.id, insert_index, task.id, task))
                        found_valid_for_drone = True

                    if found_valid_for_drone:
                        break

        if self.prefers_feasible_first():
            return self._select_first_exact_feasible_insert(
                current_drones,
                routes,
                candidate_heap,
            )

        reranked = self._rerank_candidates_exact(current_drones, routes, candidate_heap)
        return reranked or best_insert

    def _build_route_by_greedy_inserts(
        self,
        drone: Drone,
        tasks: List[Task],
    ) -> List[Task]:
        route: List[Task] = []
        remaining_tasks = list(tasks)

        while remaining_tasks:
            route_costs = {
                drone.id: (self._evaluate_route(drone, route)[1] if route else 0.0)
            }
            routes = {drone.id: route}
            preferred_task_ids = self._build_preferred_task_sets([drone], routes, remaining_tasks)
            best_insert = self._find_best_insert(
                [drone],
                remaining_tasks,
                routes,
                route_costs,
                preferred_task_ids,
                preferred_only=True,
            )
            if best_insert is None:
                best_insert = self._find_best_insert(
                    [drone],
                    remaining_tasks,
                    routes,
                    route_costs,
                    preferred_task_ids,
                    preferred_only=False,
                )
            if best_insert is None:
                break

            task, _drone_id, insert_index = best_insert
            route.insert(insert_index, task)
            remaining_tasks.remove(task)

        return route

    def _rerank_candidates_exact(
        self,
        current_drones: List[Drone],
        routes: Dict[str, List[Task]],
        candidate_heap: List[Tuple[float, str, int, str, Task]],
    ) -> Tuple[Task, str, int] | None:
        if not candidate_heap:
            return None

        top_candidates = heapq.nsmallest(self.max_exact_rerank_candidates, candidate_heap)
        if len(top_candidates) <= 1:
            return None
        if self.should_skip_empty_route_exact_rerank(
            routes,
            top_candidates,
            drone_id_index=1,
            insert_index_index=2,
        ):
            return None
        if not self.should_run_exact_rerank([candidate[0] for candidate in top_candidates]):
            return None

        started_at = perf_counter()
        drone_map = {drone.id: drone for drone in current_drones}
        exact_old_costs: Dict[str, float] = {}
        best_insert = None
        best_exact_increase = float("inf")
        best_exact_cost = float("inf")
        exact_evaluations = 0

        try:
            for _estimated_increase, drone_id, insert_index, _task_id, task in top_candidates:
                drone = drone_map.get(drone_id)
                if drone is None:
                    continue
                current_route = routes[drone_id]
                exact_old_cost = exact_old_costs.get(drone_id)
                if exact_old_cost is None:
                    if current_route:
                        exact_evaluations += 1
                        _profile, valid, cost = self.evaluate_route_candidate_exact(drone, current_route)
                    else:
                        valid, cost = True, 0.0
                    exact_old_costs[drone_id] = cost if valid else float("inf")
                    exact_old_cost = exact_old_costs[drone_id]

                test_route = (
                    current_route[:insert_index]
                    + [task]
                    + current_route[insert_index:]
                )
                exact_evaluations += 1
                _profile, valid, exact_cost = self.evaluate_route_candidate_exact(drone, test_route)
                if not valid or not np.isfinite(exact_old_cost):
                    continue

                exact_increase = exact_cost - exact_old_cost
                if exact_increase < best_exact_increase:
                    best_exact_increase = exact_increase
                    best_exact_cost = exact_cost
                    best_insert = (task, drone_id, insert_index)
        finally:
            self.add_runtime_stat("candidateExactRerankTime", perf_counter() - started_at)
            self.add_runtime_stat("candidateExactRerankEvaluations", exact_evaluations)

        if best_insert is not None:
            task, drone_id, insert_index = best_insert
            drone = drone_map[drone_id]
            current_route = routes[drone_id]
            reranked_route = (
                current_route[:insert_index]
                + [task]
                + current_route[insert_index:]
            )
            self.record_planning_trace(
                phase="candidate_exact_rerank",
                drone=drone,
                route=reranked_route,
                valid=True,
                reason=f"top{len(top_candidates)} exact_rerank",
                cost=best_exact_cost,
            )

        return best_insert

    def _evaluate_route(self, drone: Drone, route: List[Task]) -> Tuple[bool, float]:
        return self.evaluate_route_with_completion_cost(drone, route)

    def generate_path_points(self, drone: Drone, route: List[Task]) -> List[PathPoint]:
        pts: List[PathPoint] = []
        current_time = drone.currentTime
        current_range = drone.remainingRange
        current_loc = drone.currentLocation

        if drone.currentTime == 0:
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
            path, dist = self._require_planned_route(
                current_loc,
                task.location,
                current_time=current_time,
                speed=drone.speed,
                label=f"{drone.id}->{task.id}",
            )
            for pt in path[1:-1]:
                segment_dist = np.linalg.norm(np.array(current_loc.as_tuple()) - np.array(pt.as_tuple()))
                current_time += self._segment_travel_time(
                    current_loc,
                    pt,
                    speed=drone.speed,
                    current_time=current_time,
                )
                current_range -= segment_dist
                current_loc = pt
                pts.append(
                    PathPoint(
                        location=pt,
                        time=current_time,
                        action="fly",
                        taskId=None,
                        remainingRange=current_range,
                    )
                )

            final_segment_dist = np.linalg.norm(
                np.array(current_loc.as_tuple()) - np.array(task.location.as_tuple())
            )
            current_time += self._segment_travel_time(
                current_loc,
                task.location,
                speed=drone.speed,
                current_time=current_time,
            )
            current_range -= final_segment_dist
            current_loc = task.location

            wait_time = max(0, task.timeWindow[0] - current_time)
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
                current_range -= wait_time * 2

            pts.append(
                PathPoint(
                    location=current_loc,
                    time=current_time,
                    action="serve",
                    taskId=task.id,
                    remainingRange=current_range,
                )
            )
            current_time += task.serviceDuration

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
                raise RuntimeError(f"{drone.id} 无法返回任何站点")
            return_depot_id, depot_loc, _return_dist = return_choice

        if drone.returnToDepotRequired and current_loc.as_tuple() != depot_loc.as_tuple():
            home_path, _ = self._require_planned_route(
                current_loc,
                depot_loc,
                current_time=current_time,
                speed=drone.speed,
                label=f"{drone.id} 返航",
            )

            for pt in home_path[1:]:
                segment_dist = np.linalg.norm(np.array(current_loc.as_tuple()) - np.array(pt.as_tuple()))
                current_range -= segment_dist
                current_time += self._segment_travel_time(
                    current_loc,
                    pt,
                    speed=drone.speed,
                    current_time=current_time,
                )
                current_loc = pt
                pts.append(
                    PathPoint(
                        location=pt,
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
                wait_duration = actual_land_time - current_time
                current_time = actual_land_time
                current_range -= wait_duration * 2

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
