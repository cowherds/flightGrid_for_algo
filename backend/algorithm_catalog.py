"""算法目录服务：统一算法发现、对外 ID 与描述信息。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from backend.algorithms.discovery import load_algorithms_once
from backend.algorithms.registry import AlgorithmRegistry

PLANNER_EXTERNAL_TO_INTERNAL: dict[str, str] = {
 "astar_v2": "a_star_3d",
 "astar": "a_star_3d",
 "a_star_3d": "a_star_3d",
 "weighted_astar_v2": "weighted_a_star_3d",
 "weighted_a_star_3d": "weighted_a_star_3d",
 "weighted_astar_fast": "weighted_a_star_fast_3d",
 "weighted_a_star_fast_3d": "weighted_a_star_fast_3d",
 "weighted_astar_quality": "weighted_a_star_quality_3d",
 "weighted_a_star_quality_3d": "weighted_a_star_quality_3d",
 "direct_safe": "direct_safe_3d",
 "direct_safe_3d": "direct_safe_3d",
 "direct_safe_conservative": "direct_safe_conservative_3d",
 "direct_safe_conservative_3d": "direct_safe_conservative_3d",
 "orthogonal_safe": "orthogonal_safe_3d",
 "orthogonal_safe_3d": "orthogonal_safe_3d",
 "geometric_heuristic": "geometric_heuristic_3d",
 "geometric_heuristic_3d": "geometric_heuristic_3d",
 "geometric_heuristic_speed": "geometric_heuristic_speed_3d",
 "geometric_heuristic_speed_3d": "geometric_heuristic_speed_3d",
 "geometric_heuristic_quality": "geometric_heuristic_quality_3d",
 "geometric_heuristic_quality_3d": "geometric_heuristic_quality_3d",
 "dijkstra": "dijkstra_3d",
 "dijkstra_3d": "dijkstra_3d",
 "ovs": "ovs_2d_slice",
 "ovs_2d_slice": "ovs_2d_slice",
}

SCHEDULER_EXTERNAL_TO_INTERNAL: dict[str, str] = {
 "nearest_neighbor": "improved_nearest_neighbor",
 "improved_nearest_neighbor": "improved_nearest_neighbor",
 "insertion_heuristic": "insertion_heuristic",
 "kmeans": "kmeans",
 "balanced_kmeans": "balanced_kmeans",
 "sector_sweep": "sector_sweep",
 "fast_greedy": "fast_greedy",
 "capacity_first": "capacity_first",
 "distance_first": "distance_first",
 "two_level_scheduler": "two_level_scheduler",
 "milp_alns": "milp_alns",
 "ortools_vrp": "ortools_vrp",
 "ortools_vrp_fast": "ortools_vrp_fast",
 "ortools_vrp_parallel": "ortools_vrp_parallel",
 "ortools_vrp_deep": "ortools_vrp_deep",
 "ortools_vrp_tw": "ortools_vrp_tw",
 "ortools_vrp_dep": "ortools_vrp_dep",
 "ortools_vrp_sa": "ortools_vrp_sa",
 "ortools_vrp_tabu": "ortools_vrp_tabu",
 "ortools_vrp_tw_dep": "ortools_vrp_tw_dep",
 "ortools_vrp_prize": "ortools_vrp_prize",
 "ortools_vrp_capacity": "ortools_vrp_capacity",
 "hybrid_large_scale": "hybrid_large_scale",
 "hybrid_large_scale_maxcpu": "hybrid_large_scale_maxcpu",
 "hybrid_large_scale_fast": "hybrid_large_scale_fast",
 "hybrid_large_scale_quality": "hybrid_large_scale_quality",
 "hybrid_large_scale_tw": "hybrid_large_scale_tw",
 "hybrid_large_scale_dep": "hybrid_large_scale_dep",
 "hybrid_large_scale_tw_dep": "hybrid_large_scale_tw_dep",
 "lmta": "lmta",
}

CANONICAL_SCHEDULERS = [
 "nearest_neighbor",
 "lmta",
 "insertion_heuristic",
 "kmeans",
 "two_level_scheduler",
 "balanced_kmeans",
 "sector_sweep",
 "fast_greedy",
 "capacity_first",
 "distance_first",
 "milp_alns",
 "ortools_vrp",
 "ortools_vrp_fast",
 "ortools_vrp_parallel",
 "ortools_vrp_deep",
 "ortools_vrp_tw",
 "ortools_vrp_dep",
 "ortools_vrp_sa",
 "ortools_vrp_tabu",
 "ortools_vrp_tw_dep",
 "ortools_vrp_prize",
 "ortools_vrp_capacity",
 "hybrid_large_scale",
 "hybrid_large_scale_maxcpu",
 "hybrid_large_scale_fast",
 "hybrid_large_scale_quality",
 "hybrid_large_scale_tw",
 "hybrid_large_scale_dep",
 "hybrid_large_scale_tw_dep",
]

CANONICAL_PLANNERS = [
 "ovs",
 "astar_v2",
 "weighted_astar_v2",
 "weighted_astar_fast",
 "weighted_astar_quality",
 "direct_safe",
 "direct_safe_conservative",
 "orthogonal_safe",
 "geometric_heuristic_speed",
 "geometric_heuristic_quality",
 "dijkstra",
]

HEURISTIC_SHARED_PARAMETERS = {
 "max_candidate_tasks": {
 "label": "候选任务数",
 "type": "int",
 "default": 12,
 "min": 1,
 "max": 64,
 "step": 1,
 "description": "每轮先保留多少个粗筛候选任务参与后续评估。",
 },
 "max_exact_insertions": {
 "label": "精确插入上限",
 "type": "int",
 "default": 4,
 "min": 1,
 "max": 32,
 "step": 1,
 "description": "每轮允许做多少次 exact 插入评估，越大越慢但可能更稳。",
 },
 "max_exact_rerank_candidates": {
 "label": "精排 Top-K",
 "type": "int",
 "default": 2,
 "min": 1,
 "max": 16,
 "step": 1,
 "description": "进入 exact rerank 的候选数量。",
 },
 "exact_rerank_relative_gap": {
 "label": "精排相对阈值",
 "type": "float",
 "default": 0.0,
 "min": 0.0,
 "max": 1.0,
 "step": 0.01,
 "description": "若估价第一名与第二名的相对差距已足够大，则跳过 exact rerank。",
 },
 "exact_rerank_absolute_gap": {
 "label": "精排绝对阈值",
 "type": "float",
 "default": 0.0,
 "min": 0.0,
 "max": 600.0,
 "step": 5.0,
 "description": "若估价第一名与第二名的绝对差距已足够大，则跳过 exact rerank。",
 },
}

COARSE_SCHEDULE_PARAMETER = {
 "coarse_schedule_ignore_blocking_zones": {
 "label": "粗排忽略禁区",
 "type": "bool",
 "default": True,
 "description": "调度候选估价阶段忽略阻塞禁区，仅在提交前用 exact 真路径校验可行性。结果优先场景建议开启。",
 },
}

FEASIBLE_FIRST_PARAMETER = {
 "feasible_first_mode": {
 "label": "可行优先",
 "type": "bool",
 "default": True,
 "description": "只要满足硬约束就尽快接受第一个 exact 可行候选，不再为更优路径反复精排或 leftover 修复。推荐作为默认模式。",
 },
}

REPAIR_PARAMETERS = {
 "repair_candidate_drones": {
 "label": "局部修复候选无人机",
 "type": "int",
 "default": 6,
 "min": 1,
 "max": 64,
 "step": 1,
 "description": "leftover 修复时优先检查多少架最近的无人机。",
 },
 "repair_expand_candidate_drones": {
 "label": "局部修复扩展候选",
 "type": "int",
 "default": 12,
 "min": 1,
 "max": 128,
 "step": 1,
 "description": "局部候选失败后，第二轮扩展到多少架无人机。",
 },
 "repair_global_chunk_size": {
 "label": "全局兜底分块大小",
 "type": "int",
 "default": 24,
 "min": 1,
 "max": 128,
 "step": 1,
 "description": "最终全局兜底修复按多大的任务块分批执行。",
 },
}

ORTOOLS_PARAMETERS = {
 "time_limit_s": {
 "label": "求解时限",
 "type": "int",
 "default": 5,
 "min": 1,
 "max": 120,
 "step": 1,
 "unit": "s",
 "description": "OR-Tools 本轮求解的最长时长。",
 },
 "first_solution_strategy": {
 "label": "初解策略",
 "type": "select",
 "default": "PATH_CHEAPEST_ARC",
 "options": [
 {"label": "最便宜弧", "value": "PATH_CHEAPEST_ARC"},
 {"label": "并行最便宜插入", "value": "PARALLEL_CHEAPEST_INSERTION"},
 {"label": "自动", "value": "AUTOMATIC"},
 ],
 "description": "初始解的构造方式。",
 },
 "local_search_metaheuristic": {
 "label": "局部搜索",
 "type": "select",
 "default": "GUIDED_LOCAL_SEARCH",
 "options": [
 {"label": "Guided Local Search", "value": "GUIDED_LOCAL_SEARCH"},
 {"label": "Tabu", "value": "TABU_SEARCH"},
 {"label": "模拟退火", "value": "SIMULATED_ANNEALING"},
 {"label": "自动", "value": "AUTOMATIC"},
 ],
 "description": "决定 OR-Tools 如何继续改良初解。",
 },
 "allow_node_dropping": {
 "label": "允许丢点",
 "type": "bool",
 "default": True,
 "description": "允许在超载场景下放弃部分点，否则更可能直接失败。",
 },
 "disjunction_penalty_mode": {
 "label": "丢点惩罚",
 "type": "select",
 "default": "fixed",
 "options": [
 {"label": "固定惩罚", "value": "fixed"},
 {"label": "奖励模式", "value": "prize"},
 ],
 "description": "控制被放弃任务的惩罚模型。",
 },
 **REPAIR_PARAMETERS,
}

HYBRID_PARAMETERS = {
 "min_cluster_size": {
 "label": "最小簇大小",
 "type": "int",
 "default": 100,
 "min": 20,
 "max": 1000,
 "step": 10,
 "description": "小于该规模时更倾向合并簇。",
 },
 "target_cluster_size": {
 "label": "目标簇大小",
 "type": "int",
 "default": 200,
 "min": 40,
 "max": 1500,
 "step": 10,
 "description": "聚类的目标平均任务量。",
 },
 "max_cluster_size": {
 "label": "最大簇大小",
 "type": "int",
 "default": 300,
 "min": 40,
 "max": 2000,
 "step": 10,
 "description": "超出后会继续拆分，避免局部求解过重。",
 },
 "cluster_solver_time_limit": {
 "label": "簇内求解时限",
 "type": "int",
 "default": 5,
 "min": 1,
 "max": 60,
 "step": 1,
 "unit": "s",
 "description": "每个簇交给局部求解器的时长。",
 },
 "min_parallel_tasks": {
 "label": "并行门槛",
 "type": "int",
 "default": 300,
 "min": 50,
 "max": 5000,
 "step": 10,
 "description": "任务数低于该值时回退到插入式调度。",
 },
 **REPAIR_PARAMETERS,
}

MILP_ALNS_PARAMETERS = {
 "milp_time_limit": {
 "label": "MILP 时限",
 "type": "float",
 "default": 5.0,
 "min": 0.5,
 "max": 120.0,
 "step": 0.5,
 "unit": "s",
 "description": "MILP 初解阶段的求解时长。",
 },
 "alns_iterations": {
 "label": "ALNS 迭代数",
 "type": "int",
 "default": 60,
 "min": 5,
 "max": 500,
 "step": 5,
 "description": "局部搜索迭代次数，越大越慢。",
 },
 "destroy_fraction": {
 "label": "破坏比例",
 "type": "float",
 "default": 0.25,
 "min": 0.05,
 "max": 0.8,
 "step": 0.05,
 "description": "每轮 ALNS 拆掉当前解的比例。",
 },
 "repair_candidate_limit": {
 "label": "修复候选上限",
 "type": "int",
 "default": 8,
 "min": 1,
 "max": 64,
 "step": 1,
 "description": "每轮 repair 会检查多少个候选组件。",
 },
 "prune_threshold": {
 "label": "剪枝阈值",
 "type": "float",
 "default": 0.0,
 "min": 0.0,
 "max": 100000.0,
 "step": 100.0,
 "description": "低收益插入的提前剪枝阈值。",
 },
}

WEIGHTED_ASTAR_PARAMETERS = {
 "heuristic_weight": {
 "label": "启发式权重",
 "type": "float",
 "default": 2.2,
 "min": 1.0,
 "max": 5.0,
 "step": 0.1,
 "description": "越大越快，但更偏贪心。",
 },
}

SAFE_PLANNER_PARAMETERS = {
 "margin": {
 "label": "安全边距",
 "type": "float",
 "default": 20.0,
 "min": 5.0,
 "max": 120.0,
 "step": 5.0,
 "unit": "m",
 "description": "绕障和抬升时额外留出的安全边距。",
 },
}

GEOMETRIC_PLANNER_PARAMETERS = {
 "margin": {
 "label": "安全边距",
 "type": "float",
 "default": 20.0,
 "min": 2.0,
 "max": 120.0,
 "step": 2.0,
 "unit": "m",
 "description": "翻越楼顶或侧绕障碍时额外预留的安全边距。",
 },
 "max_depth": {
 "label": "递归深度",
 "type": "int",
 "default": 12,
 "min": 2,
 "max": 32,
 "step": 1,
 "description": "允许几何拆分的最大层数，越大越稳但更慢。",
 },
 "sample_step": {
 "label": "采样步长",
 "type": "float",
 "default": 5.0,
 "min": 0.5,
 "max": 20.0,
 "step": 0.5,
 "unit": "m",
 "description": "沿直线检测首个障碍时的采样间距，越小越精细。",
 },
}

ALGORITHM_DESCRIPTIONS: dict[str, dict[str, Any]] = {
 "insertion_heuristic": {
 "name": "插入启发式调度",
 "description": "逐个把任务插入到现有路线。推荐配合“可行优先”使用，先出满足约束的结果，不追求最优距离。",
 "type": "scheduler",
 "solver_mode": "heuristic",
 "complexity": "O(n²)",
 "best_for": "中等规模问题（10-100个任务）",
 "parameters": {
 **HEURISTIC_SHARED_PARAMETERS,
 **FEASIBLE_FIRST_PARAMETER,
 **COARSE_SCHEDULE_PARAMETER,
 },
 },
 "nearest_neighbor": {
 "name": "最近邻调度",
 "description": "贪心算法，按当前位置逐步追加任务。推荐配合“可行优先”作为快速出解模式。",
 "type": "scheduler",
 "solver_mode": "heuristic",
 "complexity": "O(n²)",
 "best_for": "实时快速出解",
 "parameters": {
 **HEURISTIC_SHARED_PARAMETERS,
 **FEASIBLE_FIRST_PARAMETER,
 **COARSE_SCHEDULE_PARAMETER,
 },
 },
 "kmeans": {
 "name": "KMeans 聚类调度",
 "description": "先按空间邻近聚类任务，再将任务簇分配给无人机",
 "type": "scheduler",
 "solver_mode": "heuristic",
 "complexity": "O(k*n*m)",
 "best_for": "大规模空间聚类场景",
 "parameters": HEURISTIC_SHARED_PARAMETERS,
 },
 "two_level_scheduler": {
 "name": "双层调度",
 "description": "先按任务组分配，再在组内优化访问顺序",
 "type": "scheduler",
 "solver_mode": "heuristic",
 "complexity": "O(n^2)",
 "best_for": "多目标订单和依赖场景",
 "parameters": HEURISTIC_SHARED_PARAMETERS,
 },
 "balanced_kmeans": {
 "name": "平衡 KMeans 调度",
 "description": "聚类时同时考虑空间距离、任务数和载重占比",
 "type": "scheduler",
 "solver_mode": "heuristic",
 "complexity": "O(k*n*m)",
 "best_for": "大规模均衡分配场景",
 "parameters": HEURISTIC_SHARED_PARAMETERS,
 },
 "sector_sweep": {
 "name": "扇区扫描调度",
 "description": "按空间角度分区，再局部插入优化",
 "type": "scheduler",
 "solver_mode": "heuristic",
 "complexity": "O(n log n)",
 "best_for": "大规模地理分散场景",
 "parameters": HEURISTIC_SHARED_PARAMETERS,
 },
 "fast_greedy": {
 "name": "快速贪心调度",
 "description": "优先粗分配，减少调度阶段路径规划调用",
 "type": "scheduler",
 "solver_mode": "heuristic",
 "complexity": "O(n log n)",
 "best_for": "大规模场景快速出解",
 "parameters": HEURISTIC_SHARED_PARAMETERS,
 },
 "capacity_first": {
 "name": "容量优先调度",
 "description": "优先分配容量受限无人机，提升整体利用率",
 "type": "scheduler",
 "solver_mode": "heuristic",
 "complexity": "O(n log n)",
 "best_for": "容量受限场景",
 "parameters": HEURISTIC_SHARED_PARAMETERS,
 },
 "distance_first": {
 "name": "距离优先调度",
 "description": "优先分配远距离任务，避免任务堆积",
 "type": "scheduler",
 "solver_mode": "heuristic",
 "complexity": "O(n log n)",
 "best_for": "地理分散任务",
 "parameters": MILP_ALNS_PARAMETERS,
 },
 "milp_alns": {
 "name": "MILP + ALNS 一体化调度",
 "description": "融合 MILP 初解与 ALNS 局部搜索",
 "type": "scheduler",
 "solver_mode": "heuristic",
 "complexity": "近似 O(iter·n²)",
 "best_for": "复杂约束质量优先场景",
 "parameters": ORTOOLS_PARAMETERS,
 },
 "ortools_vrp": {
 "name": "OR-Tools VRP 调度",
 "description": "基于 Google OR-Tools 的车辆路径规划调度器，支持快速 5 秒内近似优化",
 "type": "scheduler",
 "solver_mode": "solver",
 "complexity": "依赖求解过程（近似 NP-hard）",
 "best_for": "中大规模多无人机调度",
 "parameters": ORTOOLS_PARAMETERS,
 },
 "ortools_vrp_fast": {
 "name": "OR-Tools VRP 调度（快速）",
 "description": "2 秒近似优化，优先速度，适合演示和快速试算",
 "type": "scheduler",
 "solver_mode": "solver",
 "complexity": "依赖求解过程（近似 NP-hard）",
 "best_for": "中等规模快速出解",
 "parameters": ORTOOLS_PARAMETERS,
 },
 "ortools_vrp_parallel": {
 "name": "OR-Tools VRP 调度（并行初解）",
 "description": "并行插入初解 + 局部搜索，兼顾稳定性和速度",
 "type": "scheduler",
 "solver_mode": "solver",
 "complexity": "依赖求解过程（近似 NP-hard）",
 "best_for": "中大规模均衡场景",
 "parameters": ORTOOLS_PARAMETERS,
 },
 "ortools_vrp_deep": {
 "name": "OR-Tools VRP 调度（深度）",
 "description": "更长求解时限与 Tabu 搜索，优先路线质量",
 "type": "scheduler",
 "solver_mode": "solver",
 "complexity": "依赖求解过程（近似 NP-hard）",
 "best_for": "中小规模高质量优化",
 "parameters": ORTOOLS_PARAMETERS,
 },
 "ortools_vrp_tw": {
 "name": "OR-Tools VRP 调度（时间窗）",
 "description": "启用时间窗约束，保证任务在窗口内被访问",
 "type": "scheduler",
 "solver_mode": "solver",
 "complexity": "依赖求解过程（近似 NP-hard）",
 "best_for": "存在明显时效约束的调度场景",
 "parameters": ORTOOLS_PARAMETERS,
 },
 "ortools_vrp_dep": {
 "name": "OR-Tools VRP 调度（依赖约束）",
 "description": "启用任务依赖顺序约束，保障先后执行关系",
 "type": "scheduler",
 "solver_mode": "solver",
 "complexity": "依赖求解过程（近似 NP-hard）",
 "best_for": "存在前置任务依赖链的调度场景",
 "parameters": ORTOOLS_PARAMETERS,
 },
 "ortools_vrp_sa": {
 "name": "OR-Tools VRP 调度（模拟退火）",
 "description": "采用模拟退火局部搜索，跳出局部最优能力更强",
 "type": "scheduler",
 "solver_mode": "solver",
 "complexity": "依赖求解过程（近似 NP-hard）",
 "best_for": "中等规模复杂场景",
 "parameters": ORTOOLS_PARAMETERS,
 },
 "ortools_vrp_tabu": {
 "name": "OR-Tools VRP 调度（Tabu）",
 "description": "采用禁忌搜索，适合在较长时限下追求更高质量解",
 "type": "scheduler",
 "solver_mode": "solver",
 "complexity": "依赖求解过程（近似 NP-hard）",
 "best_for": "中小规模质量优先",
 "parameters": ORTOOLS_PARAMETERS,
 },
 "ortools_vrp_tw_dep": {
 "name": "OR-Tools VRP 调度（时间窗+依赖）",
 "description": "同时约束时间窗与任务先后关系",
 "type": "scheduler",
 "solver_mode": "solver",
 "complexity": "依赖求解过程（近似 NP-hard）",
 "best_for": "复杂时序与前置约束并存场景",
 "parameters": ORTOOLS_PARAMETERS,
 },
 "ortools_vrp_prize": {
 "name": "OR-Tools VRP 调度（奖励收集）",
 "description": "允许丢弃部分低价值点，优先覆盖高优先级/高价值任务",
 "type": "scheduler",
 "solver_mode": "solver",
 "complexity": "依赖求解过程（近似 NP-hard）",
 "best_for": "超负载场景下的价值最大化",
 "parameters": ORTOOLS_PARAMETERS,
 },
 "ortools_vrp_capacity": {
 "name": "OR-Tools VRP 调度（容量约束）",
 "description": "显式建模载荷容量，避免超载分配",
 "type": "scheduler",
 "solver_mode": "solver",
 "complexity": "依赖求解过程（近似 NP-hard）",
 "best_for": "载荷约束明显的配送任务",
 "parameters": HYBRID_PARAMETERS,
 },
 "hybrid_large_scale": {
 "name": "超大规模混合分治调度",
 "description": "MiniBatchKMeans 聚类 + 多进程并行 OR-Tools 局部求解 + 结果合并",
 "type": "scheduler",
 "solver_mode": "hybrid",
 "complexity": "近似 O(k·local_solve)",
 "best_for": "万级以上任务规模",
 "parameters": HYBRID_PARAMETERS,
 },
 "hybrid_large_scale_maxcpu": {
 "name": "超大规模混合分治调度（MAX CPU）",
 "description": "激进并行版：更早进入多进程分治，优先吃满 CPU 追求吞吐",
 "type": "scheduler",
 "solver_mode": "hybrid",
 "complexity": "近似 O(k·local_solve)",
 "best_for": "中大规模到超大规模，速度优先",
 "parameters": HYBRID_PARAMETERS,
 },
 "hybrid_large_scale_fast": {
 "name": "超大规模混合分治调度（快速）",
 "description": "更大簇+更短局部求解，优先吞吐和响应速度",
 "type": "scheduler",
 "solver_mode": "hybrid",
 "complexity": "近似 O(k·local_solve)",
 "best_for": "万级任务快速滚动求解",
 "parameters": HYBRID_PARAMETERS,
 },
 "hybrid_large_scale_quality": {
 "name": "超大规模混合分治调度（质量）",
 "description": "更小簇+更长局部求解，提升局部路径质量",
 "type": "scheduler",
 "solver_mode": "hybrid",
 "complexity": "近似 O(k·local_solve)",
 "best_for": "大规模且结果复核场景",
 "parameters": HYBRID_PARAMETERS,
 },
 "hybrid_large_scale_tw": {
 "name": "超大规模混合分治调度（时间窗）",
 "description": "分治并行 + 簇内时间窗约束求解",
 "type": "scheduler",
 "solver_mode": "hybrid",
 "complexity": "近似 O(k·local_solve)",
 "best_for": "超大规模且存在时效窗口",
 "parameters": HYBRID_PARAMETERS,
 },
 "hybrid_large_scale_dep": {
 "name": "超大规模混合分治调度（依赖）",
 "description": "分治并行 + 簇内依赖顺序约束求解",
 "type": "scheduler",
 "solver_mode": "hybrid",
 "complexity": "近似 O(k·local_solve)",
 "best_for": "超大规模且存在任务前置关系",
 "parameters": HYBRID_PARAMETERS,
 },
 "hybrid_large_scale_tw_dep": {
 "name": "超大规模混合分治调度（时间窗+依赖）",
 "description": "分治并行 + 簇内时间窗和依赖联合约束",
 "type": "scheduler",
 "solver_mode": "hybrid",
 "complexity": "近似 O(k·local_solve)",
 "best_for": "超大规模复杂约束联动场景",
 "parameters": {},
 },
 "astar_v2": {
 "name": "A*路径规划（3D）",
 "description": "在3D空间中使用A*算法寻找最优路径",
 "type": "planner",
 "complexity": "O(n log n)",
 "best_for": "复杂环境（有禁飞区）",
 "parameters": WEIGHTED_ASTAR_PARAMETERS,
 },
 "weighted_astar_v2": {
 "name": "加权A*路径规划（快速）",
 "description": "使用更强启发式减少搜索节点",
 "type": "planner",
 "complexity": "O(n log n)",
 "best_for": "中大规模复杂场景",
 "parameters": WEIGHTED_ASTAR_PARAMETERS,
 },
 "weighted_astar_fast": {
 "name": "加权A*路径规划（极速）",
 "description": "更强启发式权重，优先搜索速度",
 "type": "planner",
 "complexity": "O(n log n)",
 "best_for": "复杂场景快速回放",
 "parameters": WEIGHTED_ASTAR_PARAMETERS,
 },
 "weighted_astar_quality": {
 "name": "加权A*路径规划（质量）",
 "description": "较保守启发式，平衡速度与路径质量",
 "type": "planner",
 "complexity": "O(n log n)",
 "best_for": "障碍较多场景",
 "parameters": SAFE_PLANNER_PARAMETERS,
 },
 "direct_safe": {
 "name": "安全直飞规划（极速）",
 "description": "优先直飞，受阻时安全绕行",
 "type": "planner",
 "complexity": "O(k)",
 "best_for": "大规模快速场景",
 "parameters": SAFE_PLANNER_PARAMETERS,
 },
 "direct_safe_conservative": {
 "name": "安全直飞规划（保守）",
 "description": "更高安全边距的直飞绕行策略",
 "type": "planner",
 "complexity": "O(k)",
 "best_for": "禁飞区密集和安全优先场景",
 "parameters": SAFE_PLANNER_PARAMETERS,
 },
 "orthogonal_safe": {
 "name": "正交安全规划（极速）",
 "description": "尝试少量正交折线路径",
 "type": "planner",
 "complexity": "O(k)",
 "best_for": "障碍较规整场景",
 "parameters": {},
 },
 "geometric_heuristic": {
 "name": "几何启发式规划（3D）",
 "description": "先定位首个受阻障碍，再在翻越楼顶和侧面绕行之间做几何递归拆分，适合规整建筑密集场景。",
 "type": "planner",
 "complexity": "近似 O(depth * k)",
 "best_for": "规整建筑、快速近似重规划、粗路径评估",
 "parameters": GEOMETRIC_PLANNER_PARAMETERS,
 },
 "geometric_heuristic_speed": {
 "name": "几何启发式规划（极速）",
 "description": "默认速度优先版本。先做局部锚点切分，再仅在障碍附近做几何绕行，适合调度阶段高频路径代价查询。",
 "type": "planner",
 "complexity": "近似 O(depth * k)",
 "best_for": "调度估价、滚动重规划、速度优先场景",
 "parameters": GEOMETRIC_PLANNER_PARAMETERS,
 },
 "geometric_heuristic_quality": {
 "name": "几何启发式规划（质量）",
 "description": "在局部锚点切分基础上启用局部可见图，牺牲部分速度以换取更自然、更短的绕障路径。",
 "type": "planner",
 "complexity": "近似 O(depth * k + local_graph)",
 "best_for": "多障碍局部绕行、路径展示、质量优先场景",
 "parameters": GEOMETRIC_PLANNER_PARAMETERS,
 },
 "dijkstra": {
 "name": "Dijkstra路径规划",
 "description": "经典最短路径算法，保证最优路径",
 "type": "planner",
 "complexity": "O(n²)",
 "best_for": "小规模场景",
 "parameters": {},
 },
}








def _build_exposed_ids(kind: str) -> list[str]:
    load_algorithms_once()

    if kind == "planner":
        registered = set(AlgorithmRegistry.get_available_path_planners())
        canonical = CANONICAL_PLANNERS
        external_to_internal = PLANNER_EXTERNAL_TO_INTERNAL
    else:
        registered = set(AlgorithmRegistry.get_available_schedulers())
        canonical = CANONICAL_SCHEDULERS
        external_to_internal = SCHEDULER_EXTERNAL_TO_INTERNAL

    exposed: list[str] = []
    for external_id in canonical:
        internal_id = external_to_internal.get(external_id, external_id)
        if internal_id in registered:
            exposed.append(external_id)

    known_internal = set(external_to_internal.values())
    for internal_id in sorted(registered):
        if internal_id not in known_internal:
            exposed.append(internal_id)

    return exposed


def _build_descriptions(
    exposed_schedulers: list[str],
    exposed_planners: list[str],
) -> dict[str, dict[str, Any]]:
    descriptions: dict[str, dict[str, Any]] = {}
    for algorithm_id in exposed_schedulers + exposed_planners:
        if algorithm_id in ALGORITHM_DESCRIPTIONS:
            descriptions[algorithm_id] = ALGORITHM_DESCRIPTIONS[algorithm_id]
            continue

        algo_type = "scheduler" if algorithm_id in exposed_schedulers else "planner"
        descriptions[algorithm_id] = {
            "name": algorithm_id,
            "description": "自动发现算法（未提供说明）",
            "type": algo_type,
            "solver_mode": "heuristic" if algo_type == "scheduler" else None,
            "complexity": "未知",
            "best_for": "通用场景",
            "parameters": {},
        }

    return descriptions


def get_available_algorithms_payload() -> dict[str, Any]:
    schedulers = _build_exposed_ids("scheduler")
    planners = _build_exposed_ids("planner")
    descriptions = _build_descriptions(schedulers, planners)
    return {
        "schedulers": schedulers,
        "planners": planners,
        "descriptions": descriptions,
        "timestamp": datetime.now().isoformat(),
    }


def resolve_external_to_internal(kind: str, algorithm_id: str | None) -> str:
    if kind == "planner":
        default = "a_star_3d"
        mapping = PLANNER_EXTERNAL_TO_INTERNAL
    else:
        default = "insertion_heuristic"
        mapping = SCHEDULER_EXTERNAL_TO_INTERNAL

    candidate = (algorithm_id or "").strip()
    if not candidate:
        return default

    return mapping.get(candidate, candidate)
