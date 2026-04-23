"""
核心调度引擎 - 无人机任务调度系统的主控制器

该模块实现了无人机调度系统的核心引擎，负责：
1. 初始化路径规划器和任务调度器
2. 协调多架无人机的任务分配
3. 生成无人机的飞行轨迹

主要特性：
- 支持多种路径规划算法（A*、Dijkstra等）
- 支持多种调度算法（插入法、最近邻等）
- 集成排队管理，避免无人机在站点冲突
- 支持空间约束和时间窗口约束

时间复杂度：O(n*m*k) 其中n为无人机数量，m为任务数量，k为调度迭代次数
空间复杂度：O(n*m) 用于存储路由和轨迹信息
"""

from time import perf_counter
from typing import Any, Dict, List, Type
from backend.models.domain import Drone, Task, SpatialConstraint, PathPoint, GeoPoint
from backend.algorithms.utils.environment import CityEnvironment
from backend.algorithms.registry import AlgorithmRegistry
from backend.algorithms.utils.depot_manager import DepotManager
from backend.algorithms.discovery import load_algorithms_once
from backend.routing.oracle import RouteOracle

# 自动发现并加载算法模块，确保在引擎运行前注册表已装载
load_algorithms_once()


class DispatchEngine:
    """
    无人机调度引擎 - 核心调度控制器

    该类负责协调路径规划和任务调度，为多架无人机生成最优的任务分配和飞行轨迹。

    属性：
        env (CityEnvironment): 城市环境模型，包含地形、禁飞区等约束信息
        planner: 路径规划器实例，用于计算无人机间的最优路径
        SchedulerClass: 调度器类，用于分配任务给无人机

    适用场景：
        - 多无人机协同配送
        - 城市空中物流
        - 应急救援任务调度

    限制条件：
        - 假设无人机性能相同或相似
        - 任务优先级为静态值
        - 不支持动态任务插入（需要重新调度）
    """

    def __init__(
        self,
        constraints: List[SpatialConstraint],
        limits: tuple,
        planner_name: str = "a_star_3d",
        evaluation_planner_name: str | None = None,
        scheduler_name: str = "insertion_heuristic",
        planner_resolution: int | None = None,
        line_sample_step: float | None = None,
        final_line_sample_step: float | None = None,
        enable_planning_trace: bool = True,
        planner_parameters: dict[str, Any] | None = None,
        scheduler_parameters: dict[str, Any] | None = None,
        launch_mode: str = "serial",
        launch_interval: float | None = None,
        cancel_check=None,
        progress_callback=None,
        planner_class: Type[Any] | None = None,
        scheduler_class: Type[Any] | None = None,
        evaluation_planner_class: Type[Any] | None = None,
    ):
        """
        初始化调度引擎

        创建一个无人机调度引擎实例，用于协调路径规划和任务调度。
        该方法从算法注册表中获取指定的路径规划器和调度器，并初始化城市环境。

        若传入 ``planner_class`` / ``scheduler_class``（如 algo_plugins 动态加载），则不再从
        ``AlgorithmRegistry`` 按名称解析对应类；估价规划器默认仍用注册表中的快速 A*，除非
        显式传入 ``evaluation_planner_class`` 或 ``evaluation_planner_name``。

        参数：
            constraints (List[SpatialConstraint]): 空间约束列表（禁飞区、高度限制等）
                每个约束应包含立方体边界信息
            limits (tuple): 环境边界限制，格式为 (x_min, x_max, y_min, y_max, z_min, z_max)
                或 (x_max, y_max, z_max)，取决于环境模型的实现
            planner_name (str): 路径规划算法名称，默认为 "a_star_3d"
                可选值：
                - "a_star_3d": A* 算法（3D 版本）
                - "dijkstra": Dijkstra 算法
                - 其他已注册的路径规划算法
            scheduler_name (str): 任务调度算法名称，默认为 "insertion_heuristic"
                可选值：
                - "insertion_heuristic": 插入法启发式算法
                - "nearest_neighbor": 最近邻算法
                - "kmeans": K-means 聚类算法
                - 其他已注册的调度算法

        异常：
            KeyError: 如果指定的算法名称未在注册表中注册

        属性初始化：
            - env: 城市环境模型实例
            - planner: 路径规划器实例
            - SchedulerClass: 调度器类（延迟实例化）

        示例：
            >>> engine = DispatchEngine(
            ...     constraints=constraints,
            ...     limits=(500, 500, 150),
            ...     planner_name="a_star_3d",
            ...     scheduler_name="insertion_heuristic"
            ... )
        """
        # 初始化城市环境模型，加载所有空间约束
        # 环境模型用于碰撞检测和路径规划
        resolved_line_sample_step = (
            line_sample_step
            if line_sample_step is not None
            else max(6.0, min(float(planner_resolution or 20), 12.0))
        )
        resolved_final_line_sample_step = (
            final_line_sample_step
            if final_line_sample_step is not None
            else max(0.5, min(float(planner_resolution or 20) / 10.0, 2.0))
        )
        resolved_final_line_sample_step = min(
            float(resolved_line_sample_step),
            max(0.5, float(resolved_final_line_sample_step)),
        )

        self.evaluation_env = CityEnvironment(
            constraints,
            limits=limits,
            line_sample_step=resolved_line_sample_step,
        )
        self.final_env = CityEnvironment(
            constraints,
            limits=limits,
            line_sample_step=resolved_final_line_sample_step,
        )
        self.env = self.final_env

        if planner_class is not None:
            PlannerClass = planner_class
            if evaluation_planner_class is not None:
                EvaluationPlannerClass = evaluation_planner_class
                resolved_evaluation_planner_name = getattr(
                    EvaluationPlannerClass, "__name__", "evaluation_custom"
                )
            elif evaluation_planner_name:
                resolved_evaluation_planner_name = str(evaluation_planner_name)
                EvaluationPlannerClass = AlgorithmRegistry.get_path_planner(resolved_evaluation_planner_name)
            else:
                resolved_evaluation_planner_name = "weighted_a_star_fast_3d"
                EvaluationPlannerClass = AlgorithmRegistry.get_path_planner(resolved_evaluation_planner_name)
        else:
            resolved_evaluation_planner_name = self._resolve_default_evaluation_planner_name(
                planner_name,
                requested_name=evaluation_planner_name,
            )
            PlannerClass = AlgorithmRegistry.get_path_planner(planner_name)
            EvaluationPlannerClass = AlgorithmRegistry.get_path_planner(resolved_evaluation_planner_name)

        if scheduler_class is not None:
            SchedulerClass = scheduler_class
        else:
            SchedulerClass = AlgorithmRegistry.get_scheduler(scheduler_name)

        evaluation_planner_resolution = planner_resolution
        if planner_resolution is not None:
            configured_eval_resolution = self._coerce_int(
                (planner_parameters or {}).get("evaluation_grid_resolution"),
                int(planner_resolution),
            )
            if resolved_evaluation_planner_name != planner_name:
                configured_eval_resolution = max(
                    configured_eval_resolution,
                    int(round(float(planner_resolution) * 1.25)),
                )
            evaluation_planner_resolution = max(2, configured_eval_resolution)

        # 实例化路径规划器
        # 路径规划器需要环境模型来进行碰撞检测
        self.planner = PlannerClass(self.final_env, grid_resolution=planner_resolution)
        self.planner.cancel_check = cancel_check
        self.evaluation_planner = EvaluationPlannerClass(
            self.evaluation_env,
            grid_resolution=evaluation_planner_resolution,
        )
        self.evaluation_planner.cancel_check = cancel_check
        self.planner_name = planner_name
        self.evaluation_planner_name = resolved_evaluation_planner_name
        # 保存调度器类（延迟实例化，在run方法中创建）
        # 这样可以为每次调度创建一个新的调度器实例
        self.SchedulerClass = SchedulerClass
        self.cancel_check = cancel_check
        self.progress_callback = progress_callback
        self.enable_planning_trace = bool(enable_planning_trace)
        self.planner_parameters = dict(planner_parameters or {})
        self.scheduler_parameters = dict(scheduler_parameters or {})
        self.launch_mode = str(launch_mode or "serial").strip().lower()
        self.launch_interval = (
            None if launch_interval is None else max(0.0, float(launch_interval))
        )
        self.last_planning_trace = []
        self.last_phase_timings = {
            "schedulerTime": 0.0,
            "pathPlanningTime": 0.0,
            "candidateExactRerankTime": 0.0,
            "candidateExactGateTime": 0.0,
        }

        self._apply_planner_parameters(self.planner, self.planner_parameters)
        if self.evaluation_planner is not self.planner:
            self._apply_planner_parameters(self.evaluation_planner, self.planner_parameters)

    @staticmethod
    def _resolve_default_evaluation_planner_name(
        planner_name: str,
        *,
        requested_name: str | None,
    ) -> str:
        if requested_name:
            return requested_name

        fast_mapping = {
            "a_star_3d": "weighted_a_star_fast_3d",
            "dijkstra_3d": "weighted_a_star_fast_3d",
            "weighted_a_star_3d": "weighted_a_star_fast_3d",
            "weighted_a_star_quality_3d": "weighted_a_star_fast_3d",
            "ovs_2d_slice": "weighted_a_star_fast_3d",
        }
        return fast_mapping.get(planner_name, planner_name)

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return bool(value)

    def _apply_planner_parameters(self, planner, parameters: dict[str, Any]) -> None:
        if not parameters:
            return

        if hasattr(planner, "heuristic_weight") and "heuristic_weight" in parameters:
            planner.heuristic_weight = max(1.0, self._coerce_float(parameters.get("heuristic_weight"), planner.heuristic_weight))

        if hasattr(planner, "margin") and "margin" in parameters:
            planner.margin = max(1.0, self._coerce_float(parameters.get("margin"), planner.margin))

        if hasattr(planner, "max_depth") and "max_depth" in parameters:
            planner.max_depth = max(1, self._coerce_int(parameters.get("max_depth"), planner.max_depth))

        if hasattr(planner, "sample_step") and "sample_step" in parameters:
            planner.sample_step = max(0.25, self._coerce_float(parameters.get("sample_step"), planner.sample_step))

        if hasattr(planner, "push_step"):
            margin_value = getattr(planner, "margin", 1.0)
            sample_step_value = getattr(planner, "sample_step", 1.0)
            planner.push_step = max(1.0, min(float(margin_value), float(sample_step_value)))

        if hasattr(planner, "ingest_planner_parameters"):
            planner.ingest_planner_parameters(parameters)

    def _apply_scheduler_parameters(self, scheduler, parameters: dict[str, Any]) -> None:
        if not parameters:
            return

        if hasattr(scheduler, "max_candidate_tasks") and "max_candidate_tasks" in parameters:
            scheduler.max_candidate_tasks = max(1, self._coerce_int(parameters.get("max_candidate_tasks"), scheduler.max_candidate_tasks))

        if hasattr(scheduler, "max_exact_insertions") and "max_exact_insertions" in parameters:
            scheduler.max_exact_insertions = max(1, self._coerce_int(parameters.get("max_exact_insertions"), scheduler.max_exact_insertions))

        if hasattr(scheduler, "max_exact_rerank_candidates") and "max_exact_rerank_candidates" in parameters:
            scheduler.max_exact_rerank_candidates = max(1, self._coerce_int(parameters.get("max_exact_rerank_candidates"), scheduler.max_exact_rerank_candidates))

        if hasattr(scheduler, "exact_rerank_relative_gap") and "exact_rerank_relative_gap" in parameters:
            scheduler.exact_rerank_relative_gap = max(0.0, self._coerce_float(parameters.get("exact_rerank_relative_gap"), scheduler.exact_rerank_relative_gap))

        if hasattr(scheduler, "exact_rerank_absolute_gap") and "exact_rerank_absolute_gap" in parameters:
            scheduler.exact_rerank_absolute_gap = max(0.0, self._coerce_float(parameters.get("exact_rerank_absolute_gap"), scheduler.exact_rerank_absolute_gap))

        if hasattr(scheduler, "ORTOOLS_TIME_LIMIT_S") and "time_limit_s" in parameters:
            scheduler.ORTOOLS_TIME_LIMIT_S = max(1, self._coerce_int(parameters.get("time_limit_s"), scheduler.ORTOOLS_TIME_LIMIT_S))

        if hasattr(scheduler, "FIRST_SOLUTION_STRATEGY") and parameters.get("first_solution_strategy"):
            scheduler.FIRST_SOLUTION_STRATEGY = str(parameters.get("first_solution_strategy"))

        if hasattr(scheduler, "LOCAL_SEARCH_METAHEURISTIC") and parameters.get("local_search_metaheuristic"):
            scheduler.LOCAL_SEARCH_METAHEURISTIC = str(parameters.get("local_search_metaheuristic"))

        if hasattr(scheduler, "ALLOW_NODE_DROPPING") and "allow_node_dropping" in parameters:
            scheduler.ALLOW_NODE_DROPPING = self._coerce_bool(parameters.get("allow_node_dropping"), scheduler.ALLOW_NODE_DROPPING)

        if hasattr(scheduler, "DISJUNCTION_PENALTY_MODE") and parameters.get("disjunction_penalty_mode"):
            scheduler.DISJUNCTION_PENALTY_MODE = str(parameters.get("disjunction_penalty_mode"))

        if hasattr(scheduler, "MIN_CLUSTER_SIZE") and "min_cluster_size" in parameters:
            scheduler.MIN_CLUSTER_SIZE = max(1, self._coerce_int(parameters.get("min_cluster_size"), scheduler.MIN_CLUSTER_SIZE))

        if hasattr(scheduler, "TARGET_CLUSTER_SIZE") and "target_cluster_size" in parameters:
            scheduler.TARGET_CLUSTER_SIZE = max(1, self._coerce_int(parameters.get("target_cluster_size"), scheduler.TARGET_CLUSTER_SIZE))

        if hasattr(scheduler, "MAX_CLUSTER_SIZE") and "max_cluster_size" in parameters:
            scheduler.MAX_CLUSTER_SIZE = max(1, self._coerce_int(parameters.get("max_cluster_size"), scheduler.MAX_CLUSTER_SIZE))

        if hasattr(scheduler, "CLUSTER_SOLVER_TIME_LIMIT") and "cluster_solver_time_limit" in parameters:
            scheduler.CLUSTER_SOLVER_TIME_LIMIT = max(1, self._coerce_int(parameters.get("cluster_solver_time_limit"), scheduler.CLUSTER_SOLVER_TIME_LIMIT))

        if hasattr(scheduler, "MIN_PARALLEL_TASKS") and "min_parallel_tasks" in parameters:
            scheduler.MIN_PARALLEL_TASKS = max(1, self._coerce_int(parameters.get("min_parallel_tasks"), scheduler.MIN_PARALLEL_TASKS))

        if hasattr(scheduler, "repair_candidate_drones") and "repair_candidate_drones" in parameters:
            scheduler.repair_candidate_drones = max(1, self._coerce_int(parameters.get("repair_candidate_drones"), scheduler.repair_candidate_drones))

        if hasattr(scheduler, "repair_expand_candidate_drones") and "repair_expand_candidate_drones" in parameters:
            scheduler.repair_expand_candidate_drones = max(1, self._coerce_int(parameters.get("repair_expand_candidate_drones"), scheduler.repair_expand_candidate_drones))

        if hasattr(scheduler, "repair_global_chunk_size") and "repair_global_chunk_size" in parameters:
            scheduler.repair_global_chunk_size = max(1, self._coerce_int(parameters.get("repair_global_chunk_size"), scheduler.repair_global_chunk_size))

        if hasattr(scheduler, "feasible_first_mode") and "feasible_first_mode" in parameters:
            scheduler.feasible_first_mode = self._coerce_bool(
                parameters.get("feasible_first_mode"),
                scheduler.feasible_first_mode,
            )

        if hasattr(scheduler, "coarse_schedule_ignore_blocking_zones") and "coarse_schedule_ignore_blocking_zones" in parameters:
            scheduler.coarse_schedule_ignore_blocking_zones = self._coerce_bool(
                parameters.get("coarse_schedule_ignore_blocking_zones"),
                scheduler.coarse_schedule_ignore_blocking_zones,
            )

        if hasattr(scheduler, "milp_time_limit") and "milp_time_limit" in parameters:
            scheduler.milp_time_limit = max(0.1, self._coerce_float(parameters.get("milp_time_limit"), scheduler.milp_time_limit))

        if hasattr(scheduler, "alns_iterations") and "alns_iterations" in parameters:
            scheduler.alns_iterations = max(1, self._coerce_int(parameters.get("alns_iterations"), scheduler.alns_iterations))

        if hasattr(scheduler, "destroy_fraction") and "destroy_fraction" in parameters:
            scheduler.destroy_fraction = min(0.95, max(0.01, self._coerce_float(parameters.get("destroy_fraction"), scheduler.destroy_fraction)))

        if hasattr(scheduler, "repair_candidate_limit") and "repair_candidate_limit" in parameters:
            scheduler.repair_candidate_limit = max(1, self._coerce_int(parameters.get("repair_candidate_limit"), scheduler.repair_candidate_limit))

        if hasattr(scheduler, "prune_threshold") and "prune_threshold" in parameters:
            scheduler.prune_threshold = max(0.0, self._coerce_float(parameters.get("prune_threshold"), scheduler.prune_threshold))

        if hasattr(scheduler, "lmta_lambda") and "lmta_lambda" in parameters:
            scheduler.lmta_lambda = min(0.9999, max(0.1, self._coerce_float(parameters.get("lmta_lambda"), scheduler.lmta_lambda)))

        if hasattr(scheduler, "lmta_max_tasks_per_drone") and "lmta_max_tasks_per_drone" in parameters:
            scheduler.lmta_max_tasks_per_drone = max(
                1,
                self._coerce_int(parameters.get("lmta_max_tasks_per_drone"), scheduler.lmta_max_tasks_per_drone),
            )

        if hasattr(scheduler, "lmta_scale") and "lmta_scale" in parameters:
            scheduler.lmta_scale = max(1e-9, self._coerce_float(parameters.get("lmta_scale"), scheduler.lmta_scale))

        if hasattr(scheduler, "lmta_memory_aware") and "lmta_memory_aware" in parameters:
            scheduler.lmta_memory_aware = self._coerce_bool(
                parameters.get("lmta_memory_aware"),
                scheduler.lmta_memory_aware,
            )

    def _emit_progress(
        self,
        *,
        status: str,
        progress: int,
        message: str,
        planned_drones: int | None = None,
        total_drones: int | None = None,
        current_drone: str | None = None,
    ) -> None:
        if not callable(self.progress_callback):
            return
        self.progress_callback(
            status=status,
            progress=progress,
            message=message,
            planner_stats=self.get_planner_stats(),
            planned_drones=planned_drones,
            total_drones=total_drones,
            current_drone=current_drone,
        )

    def _collect_planner_phase_stats(self) -> dict[str, dict[str, float | int]]:
        planner_phase_stats = {
            "scheduler": {
                "routeRequests": 0,
                "cacheHits": 0,
                "cacheMisses": 0,
                "routeTime": 0.0,
            },
            "final": {
                "routeRequests": 0,
                "cacheHits": 0,
                "cacheMisses": 0,
                "routeTime": 0.0,
            },
            "other": {
                "routeRequests": 0,
                "cacheHits": 0,
                "cacheMisses": 0,
                "routeTime": 0.0,
            },
        }
        for source in {
            id(self.evaluation_planner): self.evaluation_planner.get_phase_stats(),
            id(self.planner): self.planner.get_phase_stats(),
        }.values():
            for phase, values in source.items():
                bucket = planner_phase_stats.setdefault(phase, {})
                for key, value in values.items():
                    bucket[key] = bucket.get(key, 0) + value
        return planner_phase_stats

    def get_planner_stats(self) -> dict:
        planner_phase_stats = self._collect_planner_phase_stats()
        total_requests = sum(int(bucket.get("routeRequests", 0)) for bucket in planner_phase_stats.values())
        total_hits = sum(int(bucket.get("cacheHits", 0)) for bucket in planner_phase_stats.values())
        total_misses = sum(int(bucket.get("cacheMisses", 0)) for bucket in planner_phase_stats.values())
        scheduler = planner_phase_stats.get("scheduler", {})
        final = planner_phase_stats.get("final", {})
        return {
            "routeRequests": total_requests,
            "cacheHits": total_hits,
            "cacheMisses": total_misses,
            "schedulerRouteRequests": int(scheduler.get("routeRequests", 0)),
            "schedulerCacheHits": int(scheduler.get("cacheHits", 0)),
            "schedulerCacheMisses": int(scheduler.get("cacheMisses", 0)),
            "finalRouteRequests": int(final.get("routeRequests", 0)),
            "finalCacheHits": int(final.get("cacheHits", 0)),
            "finalCacheMisses": int(final.get("cacheMisses", 0)),
        }

    def run(self, drones: List[Drone], tasks: List[Task], depots: Dict[str, GeoPoint]) -> Dict[str, List[PathPoint]]:
        """
        执行调度和轨迹规划

        该方法是调度引擎的主入口，执行以下步骤：
        1. 初始化排队管理器（防止无人机在站点冲突）
        2. 创建调度器实例并执行任务分配
        3. 为每架无人机生成详细的飞行轨迹

        参数：
            drones (List[Drone]): 可用无人机列表
                每架无人机应包含以下属性：
                - id: 无人机唯一标识符
                - currentLocation: 当前位置
                - currentTime: 当前时间
                - remainingRange: 剩余航程
                - currentLoad: 当前载重
                - capacity: 最大载重
                - speed: 飞行速度
                - depotId: 所属站点ID
            tasks (List[Task]): 待分配任务列表
                每个任务应包含以下属性：
                - id: 任务唯一标识符
                - location: 任务位置
                - weight: 任务重量
                - serviceDuration: 服务时间
                - timeWindow: 时间窗口（可选）
            depots (Dict[str, GeoPoint]): 站点字典，key为站点ID，value为地理坐标
                例如：{'depot_1': GeoPoint(x=0, y=0, z=0), ...}

        返回值：
            Dict[str, List[PathPoint]]: 无人机轨迹字典
                - key: 无人机ID
                - value: 该无人机的飞行轨迹点列表（包含位置、时间、动作等信息）
                每个 PathPoint 包含：
                - location: 地理位置
                - time: 到达时间
                - action: 动作类型 ('start', 'arrive', 'end')
                - taskId: 关联的任务ID（如果有）

        时间复杂度：O(n*m*k) 其中n为无人机数，m为任务数，k为调度迭代次数
        空间复杂度：O(n*m) 用于存储路由和轨迹

        调用场景：
            - 多无人机协同配送任务
            - 城市空中物流调度
            - 应急救援任务分配

        示例：
            >>> engine = DispatchEngine(constraints, limits)
            >>> trajectories = engine.run(drones, tasks, depots)
            >>> for drone_id, path_points in trajectories.items():
            ...     print(f"无人机 {drone_id} 的轨迹点数: {len(path_points)}")
            ...     for point in path_points:
            ...         print(f"  时间: {point.time}s, 位置: {point.location}, 动作: {point.action}")
        """
        # 初始化排队管理器，用于管理无人机在站点的起降时间槽
        # 防止多架无人机在同一时间使用同一站点
        occupy_duration = 5.0
        if self.launch_mode == "parallel":
            occupy_duration = 0.0
        elif self.launch_mode == "interval" and self.launch_interval is not None:
            occupy_duration = max(0.0, float(self.launch_interval))
        depot_mgr = DepotManager(occupy_duration=occupy_duration)

        # 创建调度器实例，传入路径规划器、站点信息和排队管理器
        # 调度器将使用这些信息来分配任务和规划路径
        scheduler = self.SchedulerClass(self.evaluation_planner, depots, depot_mgr)
        scheduler.trace_enabled = self.enable_planning_trace
        self._apply_scheduler_parameters(scheduler, self.scheduler_parameters)
        if getattr(scheduler, "coarse_schedule_ignore_blocking_zones", False):
            coarse_env = CityEnvironment(
                [],
                limits=(
                    self.evaluation_env.limit_x,
                    self.evaluation_env.limit_y,
                    self.evaluation_env.limit_z,
                ),
                line_sample_step=self.evaluation_env.line_sample_step,
            )
            self.evaluation_env = coarse_env
            self.evaluation_planner.env = coarse_env
            if hasattr(self.evaluation_planner, "reset_cache"):
                self.evaluation_planner.reset_cache()
        scheduler.route_oracle = RouteOracle(
            exact_planner=self.planner,
            estimate_planner=self.evaluation_planner,
        )
        if hasattr(scheduler, "reset_planning_trace"):
            scheduler.reset_planning_trace()
        if hasattr(scheduler, "reset_runtime_stats"):
            scheduler.reset_runtime_stats()
        total_drones = len(drones)
        self._emit_progress(
            status="processing",
            progress=28,
            message="调度算法分配任务中",
            planned_drones=0,
            total_drones=total_drones,
        )

        # 执行任务分配，返回每架无人机的任务列表
        # task_routes 是一个字典，key为无人机ID，value为该无人机的任务列表
        scheduler_started_at = perf_counter()
        self.evaluation_planner.set_measurement_phase("scheduler")
        if self.planner is not self.evaluation_planner:
            self.planner.set_measurement_phase("scheduler")
        try:
            task_routes = scheduler.plan(drones, tasks)
        finally:
            self.evaluation_planner.set_measurement_phase("other")
            if self.planner is not self.evaluation_planner:
                self.planner.set_measurement_phase("other")
        self.last_phase_timings["schedulerTime"] = perf_counter() - scheduler_started_at
        if hasattr(scheduler, "get_runtime_stats"):
            self.last_phase_timings.update(scheduler.get_runtime_stats())
        self._emit_progress(
            status="processing",
            progress=56,
            message="任务分配完成，开始逐机生成路径",
            planned_drones=0,
            total_drones=total_drones,
        )

        # 为每架无人机生成详细的飞行轨迹（包含中间路点、时间戳、动作等）
        path_planning_started_at = perf_counter()
        trajectory_results = {}
        scheduler.planner = self.planner
        if scheduler.route_oracle is None:
            scheduler.route_oracle = RouteOracle(
                exact_planner=self.planner,
                estimate_planner=self.planner,
            )
        else:
            scheduler.route_oracle.exact_planner = self.planner
            scheduler.route_oracle.estimate_planner = self.planner
        if hasattr(scheduler, "reset_route_profile_cache"):
            scheduler.reset_route_profile_cache()
        self.planner.set_measurement_phase("final")
        try:
            for index, d in enumerate(drones):
                if callable(self.cancel_check) and self.cancel_check():
                    raise RuntimeError("DISPATCH_CANCELLED")
                # 调用调度器的 generate_path_points 方法生成轨迹
                # 该方法将任务列表转换为详细的路径点列表
                trajectory_results[d.id] = scheduler.generate_path_points(d, task_routes[d.id])
                ratio = (index + 1) / max(total_drones, 1)
                self._emit_progress(
                    status="processing",
                    progress=int(56 + ratio * 36),
                    message=f"路径规划中：{d.id}（{index + 1}/{total_drones}）",
                    planned_drones=index + 1,
                    total_drones=total_drones,
                    current_drone=d.id,
                )
        finally:
            self.planner.set_measurement_phase("other")

        self._emit_progress(
            status="parsing",
            progress=94,
            message="路径生成完成，汇总结果中",
            planned_drones=total_drones,
            total_drones=total_drones,
        )
        self.last_planning_trace = (
            scheduler.get_planning_trace()
            if hasattr(scheduler, "get_planning_trace")
            else []
        )
        self.last_phase_timings["pathPlanningTime"] = perf_counter() - path_planning_started_at
        self.last_phase_timings["plannerPhaseStats"] = self._collect_planner_phase_stats()

        return trajectory_results
