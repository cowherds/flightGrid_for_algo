"""
平衡 K-Means 调度器。

面向大规模场景时，单纯按最近质心分配任务容易把任务压到少数无人机上。
这里在聚类分配阶段同时考虑空间距离、当前任务数和载重占比，
先得到更均衡的任务簇，再对每架无人机执行局部插入优化。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from backend.algorithms.registry import AlgorithmRegistry
from backend.algorithms.scheduling.optimization.insertion import InsertionScheduler
from backend.models.domain import Drone, Task


@AlgorithmRegistry.register_scheduler("balanced_kmeans")
class BalancedKMeansScheduler(InsertionScheduler):
    """适合大规模场景的平衡聚类调度器。"""

    def plan(self, current_drones: List[Drone], unassigned_tasks: List[Task]) -> Dict[str, List[Task]]:
        self.reset_route_profile_cache()
        self.reset_runtime_stats()
        routes = {drone.id: [] for drone in current_drones}
        if not current_drones or not unassigned_tasks:
            return routes

        centroids = {
            drone.id: np.array(drone.currentLocation.as_tuple(), dtype=float)
            for drone in current_drones
        }
        clustered_tasks = {drone.id: [] for drone in current_drones}
        max_iterations = 4
        average_task_count = max(len(unassigned_tasks) / max(len(current_drones), 1), 1.0)

        for _ in range(max_iterations):
            clustered_tasks = {drone.id: [] for drone in current_drones}
            task_counts = {drone.id: 0 for drone in current_drones}
            task_weights = {drone.id: 0.0 for drone in current_drones}

            for task in sorted(unassigned_tasks, key=lambda item: (-item.priority, -item.weight)):
                task_vector = np.array(task.location.as_tuple(), dtype=float)
                best_drone_id = None
                best_score = float("inf")

                for drone in current_drones:
                    if task.weight > drone.capacity:
                        continue

                    distance_score = float(np.linalg.norm(task_vector - centroids[drone.id]))
                    count_penalty = (task_counts[drone.id] / average_task_count) * 60.0
                    capacity_penalty = (
                        (task_weights[drone.id] + task.weight) / max(drone.capacity, 1.0)
                    ) * 80.0
                    score = distance_score + count_penalty + capacity_penalty

                    if score < best_score:
                        best_score = score
                        best_drone_id = drone.id

                if best_drone_id is None:
                    continue

                clustered_tasks[best_drone_id].append(task)
                task_counts[best_drone_id] += 1
                task_weights[best_drone_id] += task.weight

            for drone in current_drones:
                assigned_tasks = clustered_tasks[drone.id]
                if assigned_tasks:
                    centroids[drone.id] = np.mean(
                        [np.array(task.location.as_tuple(), dtype=float) for task in assigned_tasks],
                        axis=0,
                    )
                else:
                    centroids[drone.id] = np.array(drone.currentLocation.as_tuple(), dtype=float)

        for drone in current_drones:
            routes[drone.id] = self._build_local_insertion_route(drone, clustered_tasks[drone.id])

        return routes

    def _build_local_insertion_route(self, drone: Drone, tasks: List[Task]) -> List[Task]:
        """对单架无人机负责的任务簇做局部插入优化。"""
        ordered_tasks = sorted(tasks, key=lambda item: (-item.priority, -item.weight))
        return self._build_route_by_greedy_inserts(drone, ordered_tasks)
