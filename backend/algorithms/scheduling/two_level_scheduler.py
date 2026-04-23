"""
双层调度器。

第一层按任务组分配给无人机。
第二层在组内确定访问顺序。

这一版避免在组内排序阶段频繁调用真实 planner，
组内排序优先使用几何近邻和依赖关系。
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

from backend.algorithms.base import BaseScheduler
from backend.algorithms.registry import AlgorithmRegistry
from backend.models.domain import Drone, PathPoint, Task

logger = logging.getLogger(__name__)


@AlgorithmRegistry.register_scheduler("two_level_scheduler")
class TwoLevelScheduler(BaseScheduler):
    def plan(self, current_drones: List[Drone], unassigned_tasks: List[Task]) -> Dict[str, List[Task]]:
        self.reset_route_profile_cache()
        routes = {drone.id: [] for drone in current_drones}

        task_groups = self._group_tasks_by_id(unassigned_tasks)
        sorted_groups = sorted(
            task_groups.items(),
            key=lambda item: max(task.priority for task in item[1]),
            reverse=True,
        )

        logger.info("开始双层调度: drones=%s groups=%s", len(current_drones), len(sorted_groups))

        for group_id, group_tasks in sorted_groups:
            best_insert = None
            min_cost_increase = float("inf")

            for drone in current_drones:
                current_route = routes[drone.id]
                ordered_tasks = self._order_tasks_in_group(group_tasks, current_route, drone)
                old_cost = self._evaluate_route(drone, current_route)[1] if current_route else 0.0
                ranked_positions = self.rank_insertion_positions(drone, current_route, ordered_tasks)
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
                            + ordered_tasks
                            + current_route[insert_index:]
                        )
                        valid, cost = self._evaluate_route(drone, test_route)
                        if not valid:
                            continue

                        battery_ok, _ = self.check_battery_constraint(drone, test_route)
                        time_ok, _ = self.check_flight_time_constraint(drone, test_route)
                        if not (battery_ok and time_ok):
                            continue

                        cost_increase = cost - old_cost
                        if cost_increase < min_cost_increase:
                            min_cost_increase = cost_increase
                            best_insert = (group_id, drone.id, insert_index, ordered_tasks)
                        found_valid_for_drone = True

                    if found_valid_for_drone:
                        break

            if best_insert is None:
                logger.warning("无法分配任务组 %s，已跳过", group_id)
                continue

            _, drone_id, insert_index, ordered_tasks = best_insert
            routes[drone_id][insert_index:insert_index] = ordered_tasks
            logger.info("任务组 %s 分配给无人机 %s，数量=%s", group_id, drone_id, len(ordered_tasks))

        return routes

    def _group_tasks_by_id(self, tasks: List[Task]) -> Dict[str, List[Task]]:
        groups: Dict[str, List[Task]] = {}
        for task in tasks:
            group_id = task.groupId or task.id
            groups.setdefault(group_id, []).append(task)
        return groups

    def _order_tasks_in_group(self, group_tasks: List[Task], current_route: List[Task], drone: Drone) -> List[Task]:
        if not group_tasks:
            return []

        has_dependencies = any(task.dependencies for task in group_tasks)
        if has_dependencies:
            return self._sort_by_dependencies(group_tasks)
        return self._optimize_task_order(group_tasks, current_route, drone)

    def _sort_by_dependencies(self, tasks: List[Task]) -> List[Task]:
        task_map = {task.id: task for task in tasks}
        sorted_tasks: List[Task] = []
        visited = set()

        def visit(task_id: str) -> None:
            if task_id in visited:
                return
            visited.add(task_id)

            task = task_map.get(task_id)
            if task and task.dependencies:
                for dep_id in task.dependencies:
                    if dep_id in task_map:
                        visit(dep_id)

            if task:
                sorted_tasks.append(task)

        for task in tasks:
            visit(task.id)

        return sorted_tasks

    def _optimize_task_order(self, tasks: List[Task], current_route: List[Task], drone: Drone) -> List[Task]:
        if len(tasks) <= 1:
            return list(tasks)

        current_loc = current_route[-1].location if current_route else drone.currentLocation
        remaining = set(range(len(tasks)))
        ordered: List[Task] = []

        while remaining:
            next_idx = min(
                remaining,
                key=lambda idx: self._euclidean_distance(current_loc, tasks[idx].location),
            )
            remaining.remove(next_idx)
            ordered.append(tasks[next_idx])
            current_loc = tasks[next_idx].location

        return ordered

    def _evaluate_route(self, drone: Drone, route: List[Task]) -> Tuple[bool, float]:
        return self.evaluate_route_with_distance_cost(drone, route)

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
            path, _dist = self._require_planned_route(
                current_loc,
                task.location,
                current_time=current_time,
                speed=drone.speed,
                label=f"{drone.id}->{task.id}",
            )

            for pt in path[1:-1]:
                step_dist = np.linalg.norm(np.array(current_loc.as_tuple()) - np.array(pt.as_tuple()))
                current_time += self._segment_travel_time(
                    current_loc,
                    pt,
                    speed=drone.speed,
                    current_time=current_time,
                )
                current_range -= step_dist
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

            arrival_dist = np.linalg.norm(
                np.array(current_loc.as_tuple()) - np.array(task.location.as_tuple())
            )
            current_time += self._segment_travel_time(
                current_loc,
                task.location,
                speed=drone.speed,
                current_time=current_time,
            )
            current_range -= arrival_dist
            current_loc = task.location

            pts.append(
                PathPoint(
                    location=current_loc,
                    time=current_time,
                    action="arrive",
                    taskId=task.id,
                    remainingRange=current_range,
                )
            )

            current_time += task.serviceDuration
            pts.append(
                PathPoint(
                    location=current_loc,
                    time=current_time,
                    action="service",
                    taskId=task.id,
                    remainingRange=current_range,
                )
            )

        if route and drone.returnToDepotRequired:
            return_choice = self._select_return_depot(
                drone,
                current_loc,
                current_time=current_time,
                speed=drone.speed,
                exact=True,
            )
            if return_choice is None:
                raise RuntimeError(f"{drone.id} 无法返回任何站点")
            _return_depot_id, depot_loc, _return_dist = return_choice
            path, _ = self._require_planned_route(
                current_loc,
                depot_loc,
                current_time=current_time,
                speed=drone.speed,
                label=f"{drone.id} 返航",
            )

            for pt in path[1:-1]:
                step_dist = np.linalg.norm(np.array(current_loc.as_tuple()) - np.array(pt.as_tuple()))
                current_time += self._segment_travel_time(
                    current_loc,
                    pt,
                    speed=drone.speed,
                    current_time=current_time,
                )
                current_range -= step_dist
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

            return_dist = np.linalg.norm(np.array(current_loc.as_tuple()) - np.array(depot_loc.as_tuple()))
            current_time += self._segment_travel_time(
                current_loc,
                depot_loc,
                speed=drone.speed,
                current_time=current_time,
            )
            current_range -= return_dist
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
