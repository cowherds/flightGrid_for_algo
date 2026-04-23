from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import __main__
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Iterable, List, Sequence

import numpy as np

from backend.algorithms.registry import AlgorithmRegistry
from backend.algorithms.scheduling.optimization.insertion import InsertionScheduler
from backend.algorithms.scheduling.optimization.ortools_vrp import (
    _build_no_fly_zone_payload,
    solve_vrp_cluster,
)
from backend.models.domain import Drone, Task

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ClusterChunk:
    cluster_id: int
    task_ids: tuple[str, ...]
    centroid_x: float
    centroid_y: float


def _cluster_payload_solver(payload: dict) -> dict[str, list[str]]:
    return solve_vrp_cluster(
        drones_payload=payload["drones_payload"],
        tasks_payload=payload["tasks_payload"],
        no_fly_zones_payload=payload["no_fly_zones_payload"],
        time_limit_s=payload.get("time_limit_s", 5),
        max_orders_per_drone=payload.get("max_orders_per_drone", -1),
        use_time_windows=payload.get("use_time_windows", False),
        use_dependency_precedence=payload.get("use_dependency_precedence", False),
        use_capacity_constraints=payload.get("use_capacity_constraints", False),
    )


def _split_large_group(task_ids: list[str], max_size: int) -> list[list[str]]:
    if len(task_ids) <= max_size:
        return [task_ids]
    return [task_ids[i : i + max_size] for i in range(0, len(task_ids), max_size)]


def _mini_batch_kmeans_numpy(
    coords: np.ndarray,
    n_clusters: int,
    *,
    batch_size: int = 1024,
    max_iter: int = 40,
    random_state: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n_samples = coords.shape[0]
    if n_samples == 0:
        return np.empty((0,), dtype=np.int32)
    n_clusters = max(1, min(n_clusters, n_samples))

    initial_indices = rng.choice(n_samples, size=n_clusters, replace=False)
    centers = coords[initial_indices].copy()
    counts = np.zeros((n_clusters,), dtype=np.int64)

    for _ in range(max_iter):
        if n_samples <= batch_size:
            batch = coords
        else:
            batch_idx = rng.choice(n_samples, size=batch_size, replace=False)
            batch = coords[batch_idx]

        d2 = ((batch[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1)
        for i in range(batch.shape[0]):
            label = int(labels[i])
            counts[label] += 1
            eta = 1.0 / float(counts[label])
            centers[label] = (1.0 - eta) * centers[label] + eta * batch[i]

    final_d2 = ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(final_d2, axis=1).astype(np.int32)


def _merge_cluster_result(
    cluster_result: dict[str, list[str]],
    *,
    drone_by_id: dict[str, Drone],
    task_by_id: dict[str, Task],
    routes: dict[str, list[Task]],
    assigned_ids: set[str],
    scheduler: InsertionScheduler,
) -> None:
    for drone_id, task_ids in cluster_result.items():
        drone = drone_by_id.get(drone_id)
        if drone is None:
            continue

        base_route = list(routes.get(drone_id, []))
        ordered_tasks: list[Task] = []
        for task_id in task_ids:
            task = task_by_id.get(task_id)
            if task is None or task.id in assigned_ids:
                continue
            ordered_tasks.append(task)

        routes[drone_id] = scheduler._merge_ordered_tasks_for_drone(drone, base_route, ordered_tasks)
        assigned_ids.update(task.id for task in routes[drone_id])


@AlgorithmRegistry.register_scheduler("hybrid_large_scale")
class HybridLargeScaleScheduler(InsertionScheduler):
    """
    Large-scale scheduler:
    1) MiniBatchKMeans clustering
    2) proportional drone split
    3) process-level parallel OR-Tools solving
    4) merge + heuristic repair
    """

    MIN_CLUSTER_SIZE = 100
    TARGET_CLUSTER_SIZE = 200
    MAX_CLUSTER_SIZE = 300
    CLUSTER_SOLVER_TIME_LIMIT = 5
    MIN_PARALLEL_TASKS = 300
    USE_TIME_WINDOWS = False
    USE_DEPENDENCY_PRECEDENCE = False
    USE_CAPACITY_CONSTRAINTS = True

    def plan(self, current_drones: List[Drone], unassigned_tasks: List[Task]) -> Dict[str, List[Task]]:
        self.reset_route_profile_cache()
        self.reset_runtime_stats()
        routes: Dict[str, List[Task]] = {drone.id: [] for drone in current_drones}
        if not current_drones or not unassigned_tasks:
            return routes

        min_parallel_tasks = self._resolve_int_env("FLIGHTGRID_HYBRID_MIN_PARALLEL_TASKS", self.MIN_PARALLEL_TASKS)
        if len(unassigned_tasks) < min_parallel_tasks:
            logger.info("Hybrid scheduler fallback to insertion for small task volume: %s", len(unassigned_tasks))
            return super().plan(current_drones, unassigned_tasks)

        task_by_id = {task.id: task for task in unassigned_tasks}
        drone_by_id = {drone.id: drone for drone in current_drones}

        cluster_started_at = perf_counter()
        clusters = self._build_clusters(unassigned_tasks)
        self.add_runtime_stat("clusterBuildTime", perf_counter() - cluster_started_at)
        self.add_runtime_stat("clusterCount", len(clusters))
        if not clusters:
            return super().plan(current_drones, unassigned_tasks)

        drone_groups = self._split_drones_for_clusters(current_drones, clusters)
        no_fly_zones_payload = _build_no_fly_zone_payload(getattr(self.planner.env, "constraints", []))

        cluster_payloads: dict[int, dict] = {}
        for cluster in clusters:
            assigned_drones = drone_groups.get(cluster.cluster_id, [])
            if not assigned_drones:
                continue
            cluster_payloads[cluster.cluster_id] = {
                "drones_payload": [
                    {
                        "id": drone.id,
                        "start_x": float(drone.currentLocation.x),
                        "start_y": float(drone.currentLocation.y),
                        "capacity": float(drone.capacity),
                    }
                    for drone in assigned_drones
                ],
                "tasks_payload": [
                    {
                        "id": task_id,
                        "x": float(task_by_id[task_id].location.x),
                        "y": float(task_by_id[task_id].location.y),
                        "tw_start": float(task_by_id[task_id].timeWindow[0]) if task_by_id[task_id].timeWindow else 0.0,
                        "tw_end": float(task_by_id[task_id].timeWindow[1]) if task_by_id[task_id].timeWindow else 86400.0,
                        "dependencies": list(task_by_id[task_id].dependencies or []),
                        "weight": float(task_by_id[task_id].weight),
                        "priority": int(task_by_id[task_id].priority if task_by_id[task_id].priority is not None else 1),
                    }
                    for task_id in cluster.task_ids
                    if task_id in task_by_id
                ],
                "no_fly_zones_payload": no_fly_zones_payload,
                "time_limit_s": self.CLUSTER_SOLVER_TIME_LIMIT,
                "max_orders_per_drone": getattr(self, "max_orders_per_drone", -1),
                "use_time_windows": self.USE_TIME_WINDOWS,
                "use_dependency_precedence": self.USE_DEPENDENCY_PRECEDENCE,
                "use_capacity_constraints": self.USE_CAPACITY_CONSTRAINTS,
            }

        assigned_ids: set[str] = set()
        failed_cluster_ids: set[int] = set()
        merge_time = 0.0

        main_file = getattr(__main__, "__file__", None)
        run_serial_only = (
            os.getenv("FLIGHTGRID_HYBRID_SERIAL_ONLY", "0") == "1"
            or not main_file
            or str(main_file).endswith("<stdin>")
            or not os.path.exists(str(main_file))
        )
        if run_serial_only:
            failed_cluster_ids = set(cluster_payloads.keys())
        else:
            futures_map = {}
            max_workers = self._resolve_max_workers(len(cluster_payloads))
            mp_ctx = mp.get_context("spawn")
            parallel_started_at = perf_counter()
            try:
                with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as pool:
                    for cluster_id, payload in cluster_payloads.items():
                        future = pool.submit(_cluster_payload_solver, payload)
                        futures_map[future] = cluster_id

                    for future in as_completed(futures_map):
                        cluster_id = futures_map[future]
                        try:
                            cluster_result = future.result()
                        except Exception as exc:  # pragma: no cover - defensive fallback
                            logger.exception("Cluster %s solve failed in process pool: %s", cluster_id, exc)
                            failed_cluster_ids.add(cluster_id)
                            continue
                        merge_started_at = perf_counter()
                        _merge_cluster_result(
                            cluster_result,
                            drone_by_id=drone_by_id,
                            task_by_id=task_by_id,
                            routes=routes,
                            assigned_ids=assigned_ids,
                            scheduler=self,
                        )
                        merge_time += perf_counter() - merge_started_at
            except Exception as exc:  # pragma: no cover - process pool bootstrap fallback
                logger.exception("Hybrid process pool bootstrap failed, fallback to serial solve: %s", exc)
                failed_cluster_ids = set(cluster_payloads.keys())
            finally:
                self.add_runtime_stat("clusterSolveTime", perf_counter() - parallel_started_at)

        # Safety fallback for failed clusters to avoid dropping work.
        serial_fallback_started_at = perf_counter()
        for cluster_id in sorted(failed_cluster_ids):
            payload = cluster_payloads.get(cluster_id)
            if not payload:
                continue
            try:
                cluster_result = _cluster_payload_solver(payload)
            except Exception as exc:  # pragma: no cover
                logger.exception("Cluster %s serial fallback failed: %s", cluster_id, exc)
                continue
            merge_started_at = perf_counter()
            _merge_cluster_result(
                cluster_result,
                drone_by_id=drone_by_id,
                task_by_id=task_by_id,
                routes=routes,
                assigned_ids=assigned_ids,
                scheduler=self,
            )
            merge_time += perf_counter() - merge_started_at
        self.add_runtime_stat("serialFallbackSolveTime", perf_counter() - serial_fallback_started_at)
        self.add_runtime_stat("serialFallbackClusterCount", len(failed_cluster_ids))
        self.add_runtime_stat("clusterMergeTime", merge_time)

        leftovers = [task for task in unassigned_tasks if task.id not in assigned_ids]
        self.add_runtime_stat("leftoverTaskCount", len(leftovers))
        if leftovers:
            logger.info("Hybrid scheduler repair stage for leftover tasks: %s", len(leftovers))
            routes = self._repair_tasks_progressive(current_drones, routes, leftovers)

        return routes

    @staticmethod
    def _resolve_int_env(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError:
            return default
        return max(1, value)

    def _resolve_max_workers(self, cluster_count: int) -> int:
        if cluster_count <= 0:
            return 1
        configured_workers = self._resolve_int_env("FLIGHTGRID_HYBRID_WORKERS", max(1, os.cpu_count() or 1))
        return max(1, min(cluster_count, configured_workers))

    def _build_clusters(self, tasks: Sequence[Task]) -> list[_ClusterChunk]:
        task_ids = [task.id for task in tasks]
        coords = np.array([[float(task.location.x), float(task.location.y)] for task in tasks], dtype=np.float64)
        coord_by_task_id = {
            task.id: (float(task.location.x), float(task.location.y))
            for task in tasks
        }
        task_count = len(task_ids)
        if task_count == 0:
            return []

        k = max(1, int(math.ceil(task_count / float(self.TARGET_CLUSTER_SIZE))))
        labels: np.ndarray
        use_sklearn = os.getenv("FLIGHTGRID_USE_SKLEARN_KMEANS", "0") == "1"
        if use_sklearn:
            try:
                from sklearn.cluster import MiniBatchKMeans

                model = MiniBatchKMeans(
                    n_clusters=k,
                    random_state=42,
                    batch_size=min(4096, max(256, task_count // 2)),
                    n_init=10,
                )
                labels = model.fit_predict(coords)
            except Exception as exc:  # pragma: no cover - sklearn fallback
                logger.warning("sklearn MiniBatchKMeans unavailable, fallback numpy minibatch: %s", exc)
                labels = _mini_batch_kmeans_numpy(
                    coords,
                    k,
                    batch_size=min(4096, max(256, task_count // 2)),
                    max_iter=50,
                    random_state=42,
                )
        else:
            labels = _mini_batch_kmeans_numpy(
                coords,
                k,
                batch_size=min(4096, max(256, task_count // 2)),
                max_iter=50,
                random_state=42,
            )

        grouped: dict[int, list[str]] = {}
        for idx, label in enumerate(labels.tolist()):
            grouped.setdefault(int(label), []).append(task_ids[idx])

        chunks: list[_ClusterChunk] = []
        next_cluster_id = 0
        for group_task_ids in grouped.values():
            for split_ids in _split_large_group(group_task_ids, self.MAX_CLUSTER_SIZE):
                split_coords = np.array([coord_by_task_id[task_id] for task_id in split_ids], dtype=np.float64)
                centroid = split_coords.mean(axis=0) if len(split_coords) > 0 else np.array([0.0, 0.0])
                chunks.append(
                    _ClusterChunk(
                        cluster_id=next_cluster_id,
                        task_ids=tuple(split_ids),
                        centroid_x=float(centroid[0]),
                        centroid_y=float(centroid[1]),
                    )
                )
                next_cluster_id += 1

        return chunks

    def _split_drones_for_clusters(
        self,
        drones: Sequence[Drone],
        clusters: Sequence[_ClusterChunk],
    ) -> dict[int, list[Drone]]:
        if not drones or not clusters:
            return {}

        total_tasks = sum(len(cluster.task_ids) for cluster in clusters)
        if total_tasks <= 0:
            return {}

        # Initial proportional allocation.
        allocations: dict[int, int] = {}
        remaining = len(drones)
        cluster_ids = [cluster.cluster_id for cluster in clusters]
        for cluster in clusters:
            ratio = len(cluster.task_ids) / float(total_tasks)
            count = max(1, int(round(ratio * len(drones))))
            allocations[cluster.cluster_id] = count
            remaining -= count

        # Rebalance to exact drone count.
        while remaining > 0:
            target = max(clusters, key=lambda c: len(c.task_ids))
            allocations[target.cluster_id] += 1
            remaining -= 1
        while remaining < 0:
            candidates = [cid for cid in cluster_ids if allocations[cid] > 1]
            if not candidates:
                break
            target = max(candidates, key=lambda cid: allocations[cid])
            allocations[target] -= 1
            remaining += 1

        groups: dict[int, list[Drone]] = {cluster.cluster_id: [] for cluster in clusters}
        available_drones = list(drones)
        for cluster in sorted(clusters, key=lambda item: len(item.task_ids), reverse=True):
            count = allocations.get(cluster.cluster_id, 1)
            if available_drones:
                available_drones.sort(
                    key=lambda drone: math.hypot(
                        float(drone.currentLocation.x) - cluster.centroid_x,
                        float(drone.currentLocation.y) - cluster.centroid_y,
                    )
                )
                assigned = list(available_drones[:count])
                del available_drones[:count]
            else:
                assigned = []

            if not assigned:
                assigned = [drones[cluster.cluster_id % len(drones)]]
            groups[cluster.cluster_id] = assigned

        return groups


@AlgorithmRegistry.register_scheduler("hybrid_large_scale_maxcpu")
class HybridLargeScaleMaxCPUScheduler(HybridLargeScaleScheduler):
    """
    激进并行版：默认更早进入分治并行，并拉长簇内求解时间窗口。
    适用于“CPU 打满优先”的场景。
    """

    TARGET_CLUSTER_SIZE = 140
    MAX_CLUSTER_SIZE = 220
    CLUSTER_SOLVER_TIME_LIMIT = 8
    MIN_PARALLEL_TASKS = 80


@AlgorithmRegistry.register_scheduler("hybrid_large_scale_fast")
class HybridLargeScaleFastScheduler(HybridLargeScaleScheduler):
    """
    速度优先版：簇更大，局部求解时间更短，尽快给出可用解。
    """

    TARGET_CLUSTER_SIZE = 260
    MAX_CLUSTER_SIZE = 360
    CLUSTER_SOLVER_TIME_LIMIT = 3
    MIN_PARALLEL_TASKS = 120


@AlgorithmRegistry.register_scheduler("hybrid_large_scale_quality")
class HybridLargeScaleQualityScheduler(HybridLargeScaleScheduler):
    """
    质量优先版：簇更小，局部求解时间更长，提高局部路线质量。
    """

    TARGET_CLUSTER_SIZE = 120
    MAX_CLUSTER_SIZE = 180
    CLUSTER_SOLVER_TIME_LIMIT = 12
    MIN_PARALLEL_TASKS = 80


@AlgorithmRegistry.register_scheduler("hybrid_large_scale_tw")
class HybridLargeScaleTimeWindowsScheduler(HybridLargeScaleScheduler):
    TARGET_CLUSTER_SIZE = 160
    MAX_CLUSTER_SIZE = 240
    CLUSTER_SOLVER_TIME_LIMIT = 8
    MIN_PARALLEL_TASKS = 120
    USE_TIME_WINDOWS = True


@AlgorithmRegistry.register_scheduler("hybrid_large_scale_dep")
class HybridLargeScaleDependencyScheduler(HybridLargeScaleScheduler):
    TARGET_CLUSTER_SIZE = 160
    MAX_CLUSTER_SIZE = 240
    CLUSTER_SOLVER_TIME_LIMIT = 8
    MIN_PARALLEL_TASKS = 120
    USE_DEPENDENCY_PRECEDENCE = True


@AlgorithmRegistry.register_scheduler("hybrid_large_scale_tw_dep")
class HybridLargeScaleTimeWindowsDependencyScheduler(HybridLargeScaleScheduler):
    TARGET_CLUSTER_SIZE = 140
    MAX_CLUSTER_SIZE = 220
    CLUSTER_SOLVER_TIME_LIMIT = 10
    MIN_PARALLEL_TASKS = 100
    USE_TIME_WINDOWS = True
    USE_DEPENDENCY_PRECEDENCE = True
