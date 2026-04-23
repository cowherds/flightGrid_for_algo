"""
LMTA 风格任务分配：预计算边代价（调用路径规划器），再集中式拍卖分配；
最终路线评估与轨迹生成复用插入调度器逻辑（含约束检查与 get_route 缓存复用）。

离线验证（与任意规划器组合）：在仓库根目录执行 ``python3 algo_run.py -s lmta -p <algorithmId>``，
或使用 ``--preset lmta+ovs``；场景 JSON 放在 ``backend/data/``（或 ``data/``）下。详见根目录 ``algo_run.py`` 文档。
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import numpy as np

from backend.algorithms.registry import AlgorithmRegistry
from backend.algorithms.scheduling.optimization.insertion import InsertionScheduler
from backend.models.domain import Drone, GeoPoint, Task


def _finite_dist(path_result: Tuple[list, float]) -> float:
    _path, dist = path_result
    if not math.isfinite(float(dist)):
        return float("inf")
    return float(dist)


@AlgorithmRegistry.register_scheduler("lmta")
class LMTAScheduler(InsertionScheduler):
    """
    两阶段：
    1) 对所有无人机-任务、任务-任务边调用 planner.get_route，填满规划器内部缓存；
    2) 按边际效用（Lambda^累计距离）做集中式拍卖，生成每架无人机的任务序列。

    Memory-aware（默认开启）：每轮拍卖中，**未中标**且本地路线状态未变的无人机，
    对已缓存任务 j 的边际效用 w(di,j) 可直接复用；仅剔除本轮已分配任务 j*，并在
    **中标机**更新序列后对其整表失效重算。从而减少重复的幂次与预算判断（距离
    矩阵仍由预计算阶段的 get_route 提供，语义不变）。

    参数（scheduler_parameters）：
    - lmta_lambda: 衰减系数，默认 0.95
    - lmta_max_tasks_per_drone: 每机最多任务数，默认 48
    - lmta_scale: 距离缩放（与论文中 grid scale 对齐），默认 1.0
    - lmta_memory_aware: 是否启用拍卖轮的边际效用缓存（默认 True）
    """

    def __init__(self, planner, depots, depot_mgr):
        super().__init__(planner, depots, depot_mgr)
        self.lmta_lambda = 0.95
        self.lmta_max_tasks_per_drone = 48
        self.lmta_scale = 1.0
        self.lmta_memory_aware = True
        self.last_lmta_bid_stats: Dict[str, int] = {"marginal_recomputed": 0, "marginal_reused": 0}

    def plan(self, current_drones: List[Drone], unassigned_tasks: List[Task]) -> Dict[str, List[Task]]:
        self.reset_route_profile_cache()
        self.reset_runtime_stats()

        if not unassigned_tasks:
            return {drone.id: [] for drone in current_drones}

        tasks_ordered = sorted(
            unassigned_tasks,
            key=lambda task: (len(task.dependencies or []), -int(task.priority), task.id),
        )
        n_t = len(tasks_ordered)
        n_d = len(current_drones)

        max_tasks = max(1, int(getattr(self, "lmta_max_tasks_per_drone", 48)))
        lam = float(getattr(self, "lmta_lambda", 0.95))
        lam = min(0.9999, max(0.1, lam))
        scale = float(getattr(self, "lmta_scale", 1.0))
        if scale <= 0:
            scale = 1.0

        self._precompute_route_cache(current_drones, tasks_ordered)

        dist_u2t = np.full((n_d, n_t), np.inf, dtype=float)
        dist_t2t = np.full((n_t, n_t), np.inf, dtype=float)
        dist_t2depot = np.full((n_d, n_t), np.inf, dtype=float)

        for di, drone in enumerate(current_drones):
            depot = self.depots.get(drone.depotId)
            for tj, task in enumerate(tasks_ordered):
                dist_u2t[di, tj] = _finite_dist(
                    self.planner.get_route(drone.currentLocation, task.location)
                )
                if depot is not None:
                    dist_t2depot[di, tj] = _finite_dist(self.planner.get_route(task.location, depot))

        for i in range(n_t):
            for j in range(n_t):
                if i == j:
                    dist_t2t[i, j] = 0.0
                else:
                    dist_t2t[i, j] = _finite_dist(
                        self.planner.get_route(tasks_ordered[i].location, tasks_ordered[j].location)
                    )

        Na: List[List[int]] = [[j for j in range(n_t)] for _ in range(n_d)]
        p: List[List[int]] = [[] for _ in range(n_d)]
        dist_cost = [0.0 for _ in range(n_d)]
        remaining = set(range(n_t))

        memory_aware = bool(getattr(self, "lmta_memory_aware", True))
        bid_cache: DefaultDict[int, Dict[int, float]] = defaultdict(dict)
        cache_dirty = [True] * n_d
        marginal_recomputed = 0
        marginal_reused = 0

        while remaining:
            self.planner.check_cancelled()
            bids: List[Tuple[float, int, int]] = []
            for di, drone in enumerate(current_drones):
                if len(p[di]) >= max_tasks:
                    continue
                available = [j for j in Na[di] if j in remaining]
                if not available:
                    continue

                if memory_aware:
                    bc = bid_cache[di]
                    if cache_dirty[di]:
                        bc.clear()
                        for j in available:
                            w = self._marginal_w_for_task(
                                di,
                                j,
                                drone,
                                p[di],
                                dist_cost[di],
                                dist_u2t,
                                dist_t2t,
                                dist_t2depot,
                                lam,
                                scale,
                            )
                            bc[j] = w
                            marginal_recomputed += 1
                        cache_dirty[di] = False
                    else:
                        for k in list(bc.keys()):
                            if k not in available:
                                del bc[k]
                        for j in available:
                            if j in bc:
                                marginal_reused += 1
                            else:
                                bc[j] = self._marginal_w_for_task(
                                    di,
                                    j,
                                    drone,
                                    p[di],
                                    dist_cost[di],
                                    dist_u2t,
                                    dist_t2t,
                                    dist_t2depot,
                                    lam,
                                    scale,
                                )
                                marginal_recomputed += 1

                    best_j = max(available, key=lambda j: bc.get(j, -1.0))
                    best_w = float(bc.get(best_j, -1.0))
                else:
                    best_j, best_w = self._best_marginal_task(
                        di,
                        drone,
                        available,
                        p[di],
                        dist_cost[di],
                        dist_u2t,
                        dist_t2t,
                        dist_t2depot,
                        lam,
                        scale,
                    )

                if best_w > 0 and math.isfinite(best_w):
                    bids.append((best_w, di, best_j))

            if not bids:
                break

            bids.sort(key=lambda item: (-item[0], item[1], item[2]))
            _w, a_star, j_star = bids[0]

            p[a_star].append(j_star)
            remaining.discard(j_star)
            dist_cost[a_star] = self._route_distance_after_append(
                a_star,
                p[a_star],
                dist_u2t,
                dist_t2t,
            )

            for di in range(n_d):
                if j_star in Na[di]:
                    Na[di].remove(j_star)
                if memory_aware:
                    bid_cache[di].pop(j_star, None)
                    if di == a_star:
                        cache_dirty[di] = True
                        bid_cache[di].clear()

        self.last_lmta_bid_stats = {
            "marginal_recomputed": marginal_recomputed,
            "marginal_reused": marginal_reused,
        }
        if memory_aware:
            self.add_runtime_stat("lmtaMarginalRecomputed", marginal_recomputed)
            self.add_runtime_stat("lmtaMarginalReused", marginal_reused)

        routes: Dict[str, List[Task]] = {drone.id: [] for drone in current_drones}
        for di, drone in enumerate(current_drones):
            ordered = [tasks_ordered[j] for j in p[di]]
            routes[drone.id] = self._feasible_prefix_route(drone, ordered)

        return routes

    def _precompute_route_cache(self, drones: List[Drone], tasks: List[Task]) -> None:
        """显式预热缓存：后续 evaluate_route / generate_path_points 直接复用 get_route 结果。"""
        for drone in drones:
            for task in tasks:
                self.planner.check_cancelled()
                self.planner.get_route(drone.currentLocation, task.location)
        for i, ti in enumerate(tasks):
            for tj in tasks[i + 1 :]:
                self.planner.check_cancelled()
                self.planner.get_route(ti.location, tj.location)
        for drone in drones:
            depot = self.depots.get(drone.depotId)
            if depot is None:
                continue
            for task in tasks:
                self.planner.check_cancelled()
                self.planner.get_route(task.location, depot)

    @staticmethod
    def _route_distance_after_append(
        di: int,
        seq: List[int],
        dist_u2t: np.ndarray,
        dist_t2t: np.ndarray,
    ) -> float:
        if not seq:
            return 0.0
        s = float(dist_u2t[di, seq[0]])
        for a in range(len(seq) - 1):
            s += float(dist_t2t[seq[a], seq[a + 1]])
        return s

    @staticmethod
    def _marginal_w_for_task(
        di: int,
        j: int,
        drone: Drone,
        seq: List[int],
        current_dist: float,
        dist_u2t: np.ndarray,
        dist_t2t: np.ndarray,
        dist_t2depot: np.ndarray,
        lam: float,
        scale: float,
    ) -> float:
        budget = float(drone.remainingRange or drone.maxRange or 1e12)
        if seq:
            step = float(dist_t2t[seq[-1], j])
            if not math.isfinite(step):
                return -1.0
            distance_j = current_dist + step
        else:
            step = float(dist_u2t[di, j])
            if not math.isfinite(step):
                return -1.0
            distance_j = step

        w = lam ** distance_j
        back = float(dist_t2depot[di, j])
        f_dist = distance_j * scale + back * scale
        if f_dist > budget:
            w = 0.001
        return float(w)

    def _best_marginal_task(
        self,
        di: int,
        drone: Drone,
        available: List[int],
        seq: List[int],
        current_dist: float,
        dist_u2t: np.ndarray,
        dist_t2t: np.ndarray,
        dist_t2depot: np.ndarray,
        lam: float,
        scale: float,
    ) -> Tuple[int, float]:
        best_j = available[0]
        best_w = -1.0
        for j in available:
            w = self._marginal_w_for_task(
                di,
                j,
                drone,
                seq,
                current_dist,
                dist_u2t,
                dist_t2t,
                dist_t2depot,
                lam,
                scale,
            )
            if w > best_w:
                best_w = w
                best_j = j
        return best_j, best_w

    def _feasible_prefix_route(self, drone: Drone, ordered: List[Task]) -> List[Task]:
        if not ordered:
            return []
        prefix: List[Task] = []
        for task in ordered:
            trial = prefix + [task]
            _profile, valid, _cost = self.evaluate_route_candidate(drone, trial)
            if not valid:
                continue
            prefix = trial
        return prefix
