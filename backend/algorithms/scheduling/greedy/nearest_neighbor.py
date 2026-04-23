"""
最近邻调度器。

这一版先对待选任务做几何粗筛，再做精确路径评估，
减少每轮为每架无人机遍历全部任务时的真实算路次数。
"""

import numpy as np
from typing import Dict, List, Tuple
import heapq
from time import perf_counter

from backend.algorithms.base import BaseScheduler
from backend.algorithms.registry import AlgorithmRegistry
from backend.models.domain import Drone, GeoPoint, PathPoint, Task


@AlgorithmRegistry.register_scheduler("improved_nearest_neighbor")
class NearestNeighborScheduler(BaseScheduler):
    def plan(self, current_drones: List[Drone], unassigned_tasks: List[Task]) -> Dict[str, List[Task]]:
        self.reset_route_profile_cache()
        self.reset_runtime_stats()
        routes = {drone.id: [] for drone in current_drones}
        tasks_pool = sorted(unassigned_tasks.copy(), key=lambda task: task.priority, reverse=True)
        virtual_locs = {drone.id: drone.currentLocation for drone in current_drones}

        while tasks_pool:
            self.planner.check_cancelled()
            assigned_this_round = False

            for drone in current_drones:
                if not tasks_pool:
                    break

                virtual_loc = virtual_locs[drone.id]
                best_task = self._pick_best_next_task(
                    drone,
                    routes[drone.id],
                    tasks_pool,
                    virtual_loc,
                )

                if best_task is None:
                    continue

                routes[drone.id].append(best_task)
                tasks_pool.remove(best_task)
                virtual_locs[drone.id] = best_task.location
                assigned_this_round = True

            if not assigned_this_round:
                break

        return routes

    def _pick_best_next_task(
        self,
        drone: Drone,
        current_route: List[Task],
        tasks_pool: List[Task],
        anchor: GeoPoint,
        *,
        candidate_tasks: List[Task] | None = None,
    ) -> Task | None:
        old_cost = self._evaluate_route(drone, current_route)[1] if current_route else 0.0
        preferred_tasks = candidate_tasks or self.rank_candidate_tasks(
            anchor,
            tasks_pool,
            limit=self.max_candidate_tasks,
            drone=drone,
        )
        remaining_tasks = [task for task in tasks_pool if task not in preferred_tasks]

        best_task = None
        best_estimated_increase = float("inf")
        best_anchor_dist = float("inf")
        candidate_heap: List[Tuple[float, float, str, Task]] = []

        for task_group in (preferred_tasks, remaining_tasks):
            if not task_group:
                continue

            found_valid_in_group = False
            for task in task_group:
                test_route = current_route + [task]
                _profile, valid, cost = self.evaluate_route_candidate(drone, test_route)
                if not valid:
                    continue

                estimated_increase = cost - old_cost
                anchor_dist = self._euclidean_distance(anchor, task.location)
                if (
                    estimated_increase < best_estimated_increase
                    or (
                        np.isclose(estimated_increase, best_estimated_increase)
                        and anchor_dist < best_anchor_dist
                    )
                ):
                    best_estimated_increase = estimated_increase
                    best_anchor_dist = anchor_dist
                    best_task = task
                heapq.heappush(candidate_heap, (estimated_increase, anchor_dist, task.id, task))
                found_valid_in_group = True

            if found_valid_in_group and best_task is not None:
                break

        if self.prefers_feasible_first():
            return self._select_first_exact_feasible_append_task(
                drone,
                current_route,
                candidate_heap,
            )

        reranked = self._rerank_append_candidates_exact(drone, current_route, candidate_heap)
        return reranked or best_task

    def _rerank_append_candidates_exact(
        self,
        drone: Drone,
        current_route: List[Task],
        candidate_heap: List[Tuple[float, float, str, Task]],
    ) -> Task | None:
        if not candidate_heap:
            return None

        top_candidates = heapq.nsmallest(self.max_exact_rerank_candidates, candidate_heap)
        if len(top_candidates) <= 1:
            return None
        if not self.should_run_exact_rerank([candidate[0] for candidate in top_candidates]):
            return None

        started_at = perf_counter()
        exact_evaluations = 0
        exact_old_cost = 0.0
        if current_route:
            exact_evaluations += 1
            _profile, valid, cost = self.evaluate_route_candidate_exact(drone, current_route)
            exact_old_cost = cost if valid else float("inf")

        best_task = None
        best_exact_increase = float("inf")
        best_exact_cost = float("inf")
        best_anchor_dist = float("inf")

        try:
            for _estimated_increase, anchor_dist, _task_id, task in top_candidates:
                test_route = current_route + [task]
                exact_evaluations += 1
                _profile, valid, exact_cost = self.evaluate_route_candidate_exact(drone, test_route)
                if not valid or not np.isfinite(exact_old_cost):
                    continue

                exact_increase = exact_cost - exact_old_cost
                if (
                    exact_increase < best_exact_increase
                    or (
                        np.isclose(exact_increase, best_exact_increase)
                        and anchor_dist < best_anchor_dist
                    )
                ):
                    best_exact_increase = exact_increase
                    best_exact_cost = exact_cost
                    best_anchor_dist = anchor_dist
                    best_task = task
        finally:
            self.add_runtime_stat("candidateExactRerankTime", perf_counter() - started_at)
            self.add_runtime_stat("candidateExactRerankEvaluations", exact_evaluations)

        if best_task is not None:
            self.record_planning_trace(
                phase="candidate_exact_rerank",
                drone=drone,
                route=current_route + [best_task],
                valid=True,
                reason=f"top{len(top_candidates)} exact_rerank_append",
                cost=best_exact_cost,
            )

        return best_task

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
                pts.append(PathPoint(current_loc, current_time, "wait", None, current_range))
                current_time = actual_takeoff_time
            pts.append(PathPoint(current_loc, current_time, "takeoff", None, current_range))
        else:
            pts.append(PathPoint(current_loc, current_time, "fly", None, current_range))

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
                pts.append(PathPoint(pt, current_time, "fly", None, current_range))

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
                pts.append(PathPoint(current_loc, current_time, "wait", task.id, current_range))
                current_time += wait_time
                current_range -= wait_time * 2

            pts.append(PathPoint(current_loc, current_time, "serve", task.id, current_range))
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
                pts.append(PathPoint(pt, current_time, "fly", None, current_range))

            actual_land_time = self.depot_mgr.get_available_time(return_depot_id, current_time)
            if actual_land_time > current_time:
                pts.append(PathPoint(depot_loc, current_time, "wait", "HOVER_FOR_LANDING", current_range))
                wait_duration = actual_land_time - current_time
                current_time = actual_land_time
                current_range -= wait_duration * 2

            pts.append(PathPoint(depot_loc, current_time, "land", None, current_range))

        return pts
