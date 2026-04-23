"""
容量优先调度器。

特点：
1. 任务按优先级和重量排序。
2. 无人机按容量排序。
3. 插入位置先粗筛，再精确评估，必要时回退全量扫描。
"""

import numpy as np
from typing import Dict, List, Tuple

from backend.algorithms.registry import AlgorithmRegistry
from backend.algorithms.scheduling.optimization.insertion import InsertionScheduler
from backend.models.domain import Drone, PathPoint, Task


@AlgorithmRegistry.register_scheduler("capacity_first")
class CapacityFirstScheduler(InsertionScheduler):
    def plan(self, current_drones: List[Drone], unassigned_tasks: List[Task]) -> Dict[str, List[Task]]:
        self.reset_route_profile_cache()
        self.reset_runtime_stats()
        routes = {drone.id: [] for drone in current_drones}
        sorted_tasks = sorted(unassigned_tasks, key=lambda task: (-task.priority, -task.weight))
        sorted_drones = sorted(current_drones, key=lambda drone: drone.capacity, reverse=True)

        for task in sorted_tasks:
            route_costs = {
                drone.id: (self._evaluate_route(drone, routes[drone.id])[1] if routes[drone.id] else 0.0)
                for drone in sorted_drones
            }
            preferred_task_ids = self._build_preferred_task_sets(sorted_drones, routes, [task])
            best_insert = self._find_best_insert(
                sorted_drones,
                [task],
                routes,
                route_costs,
                preferred_task_ids,
                preferred_only=True,
            )
            if best_insert is None:
                best_insert = self._find_best_insert(
                    sorted_drones,
                    [task],
                    routes,
                    route_costs,
                    preferred_task_ids,
                    preferred_only=False,
                )

            if best_insert is not None:
                assigned_task, best_drone_id, best_idx = best_insert
                routes[best_drone_id].insert(best_idx, assigned_task)

        return routes

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
                step_dist = np.linalg.norm(np.array(current_loc.as_tuple()) - np.array(pt.as_tuple()))
                current_time += self._segment_travel_time(
                    current_loc,
                    pt,
                    speed=drone.speed,
                    current_time=current_time,
                )
                current_range -= step_dist
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
                step_dist = np.linalg.norm(np.array(current_loc.as_tuple()) - np.array(pt.as_tuple()))
                current_range -= step_dist
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
