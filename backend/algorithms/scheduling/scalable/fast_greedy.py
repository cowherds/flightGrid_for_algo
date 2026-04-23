"""
快速贪心调度器。

设计目标是用非常低的计算代价先给出一份可执行分配，
尽量减少在调度阶段反复调用路径规划器。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from backend.algorithms.registry import AlgorithmRegistry
from backend.algorithms.scheduling.greedy.nearest_neighbor import NearestNeighborScheduler
from backend.models.domain import Drone, Task


@AlgorithmRegistry.register_scheduler("fast_greedy")
class FastGreedyScheduler(NearestNeighborScheduler):
    """优先速度的快速贪心调度器。"""

    def plan(self, current_drones: List[Drone], unassigned_tasks: List[Task]) -> Dict[str, List[Task]]:
        self.reset_route_profile_cache()
        self.reset_runtime_stats()
        routes = {drone.id: [] for drone in current_drones}
        if not current_drones or not unassigned_tasks:
            return routes

        tasks_pool = sorted(
            unassigned_tasks,
            key=lambda task: (-task.priority, task.timeWindow[1], task.location.x, task.location.y),
        )
        virtual_locs = {drone.id: drone.currentLocation for drone in current_drones}
        rough_ranges = {drone.id: float(drone.remainingRange or drone.maxRange or 0.0) for drone in current_drones}

        while tasks_pool:
            self.planner.check_cancelled()
            assigned_this_round = False

            for drone in current_drones:
                if not tasks_pool:
                    break

                current_loc = virtual_locs[drone.id]
                remaining_range = rough_ranges[drone.id]
                scored_candidates = []

                for task in tasks_pool:
                    candidate_route = routes[drone.id] + [task]
                    load_profile = self._analyze_route_load_profile(drone, candidate_route)
                    if not load_profile["valid"]:
                        continue

                    travel_distance = self._euclidean(current_loc, task.location)
                    if drone.returnToDepotRequired and self.depots:
                        return_distance = min(
                            self._euclidean(task.location, depot_loc)
                            for depot_loc in self.depots.values()
                        )
                    else:
                        return_distance = 0.0
                    if travel_distance + return_distance > remaining_range:
                        continue

                    score = travel_distance - (task.priority * 25.0)
                    scored_candidates.append((score, task))

                scored_candidates.sort(key=lambda item: (item[0], item[1].timeWindow[1], item[1].id))
                candidate_tasks = [task for _score, task in scored_candidates[: self.max_candidate_tasks]]
                best_task = self._pick_best_next_task(
                    drone,
                    routes[drone.id],
                    tasks_pool,
                    current_loc,
                    candidate_tasks=candidate_tasks,
                )

                if best_task is None:
                    continue

                routes[drone.id].append(best_task)
                tasks_pool.remove(best_task)
                virtual_locs[drone.id] = best_task.location
                rough_ranges[drone.id] = max(0.0, remaining_range - self._euclidean(current_loc, best_task.location))
                assigned_this_round = True

            if not assigned_this_round:
                break

        return routes

    def _euclidean(self, start, goal) -> float:
        return float(np.linalg.norm(np.array(start.as_tuple()) - np.array(goal.as_tuple())))
