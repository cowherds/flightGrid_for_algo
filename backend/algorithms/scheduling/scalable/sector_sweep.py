"""
扇区扫描调度器。

该算法先按空间角度对任务做 sweep 分区，再在分区内执行局部插入优化。
对大量空间分散订单时，通常能比纯插入法更快得到可用解。
"""

from __future__ import annotations

from math import atan2
from typing import Dict, List

from backend.algorithms.registry import AlgorithmRegistry
from backend.algorithms.scheduling.optimization.insertion import InsertionScheduler
from backend.models.domain import Drone, Task


@AlgorithmRegistry.register_scheduler("sector_sweep")
class SectorSweepScheduler(InsertionScheduler):
    """适合地理分散大规模场景的扫描式调度器。"""

    def plan(self, current_drones: List[Drone], unassigned_tasks: List[Task]) -> Dict[str, List[Task]]:
        self.reset_route_profile_cache()
        self.reset_runtime_stats()
        routes = {drone.id: [] for drone in current_drones}
        if not current_drones or not unassigned_tasks:
            return routes

        center_x = sum(drone.currentLocation.x for drone in current_drones) / len(current_drones)
        center_y = sum(drone.currentLocation.y for drone in current_drones) / len(current_drones)
        sorted_tasks = sorted(
            unassigned_tasks,
            key=lambda task: (
                atan2(task.location.y - center_y, task.location.x - center_x),
                -task.priority,
                task.location.x,
                task.location.y,
            ),
        )
        drones_by_capacity = sorted(current_drones, key=lambda drone: drone.capacity, reverse=True)

        current_drone_index = 0
        for task in sorted_tasks:
            assigned = False
            for offset in range(len(drones_by_capacity)):
                drone = drones_by_capacity[(current_drone_index + offset) % len(drones_by_capacity)]
                candidate_route = routes[drone.id] + [task]
                _profile, valid, _cost = self.evaluate_route_candidate(drone, candidate_route)
                if not valid:
                    continue

                routes[drone.id].append(task)
                current_drone_index = (current_drone_index + offset) % len(drones_by_capacity)
                assigned = True
                break

            if not assigned:
                continue

        for drone in current_drones:
            routes[drone.id] = self._rebalance_route(drone, routes[drone.id])

        return routes

    def _rebalance_route(self, drone: Drone, tasks: List[Task]) -> List[Task]:
        """在扇区内对访问顺序做一次轻量插入优化。"""
        ordered_tasks = sorted(tasks, key=lambda item: (-item.priority, item.timeWindow[1]))
        return self._build_route_by_greedy_inserts(drone, ordered_tasks)
