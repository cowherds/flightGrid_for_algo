"""
场景 JSON → 调度引擎：任意「任务分配（调度器）」+「路径规划器」组合。

- 解耦：分别指定外部 `schedulingId` 与 `algorithmId`（与 API / catalog 一致）。
- 耦合到场景：`use_scene_algorithms=True` 时使用 scene 内已保存的 `schedulingId` + `algorithmId`。

离线验证请使用仓库根目录的 ``algo_run.py``（``-s`` / ``-p`` / ``--preset`` / ``--from-scene-algorithms``），
场景数据默认放在 ``backend/data/*.json``。

**无装饰器开发**：在 ``backend/algorithms/algo_plugins.json`` 中为规划器/调度器配置
``{"module": "...", "class": "..."}`` 后，``algo_run`` 会优先按该清单 ``import`` 类，无需
``@AlgorithmRegistry.register_*``（未在清单中命中时仍依赖注册表）。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from time import perf_counter
from typing import Any

from backend.algorithms.core.engine import DispatchEngine
from backend.algorithms.plugin_loader import try_load_planner_class, try_load_scheduler_class
from backend.algorithm_catalog import resolve_external_to_internal
from backend.models.algorithm_api import (
    AlgorithmRequest,
    PlanningConfig,
    to_core_depots,
    to_core_drones,
    to_core_no_fly_zones,
    to_core_targets,
)


def build_space_limits(request: AlgorithmRequest) -> tuple[int, int, int]:
    """从站点、目标、禁飞区与无人机位置估计场景边界盒（与 dispatch 路由思路一致）。"""
    max_x = max(
        [depot.x for depot in request.depots]
        + [target.x for target in request.targets]
        + [vertex.x for zone in request.no_fly_zones for vertex in zone.vertices]
        + [drone.x for drone in request.drones]
        + [1000.0],
    )
    max_y = max(
        [depot.y for depot in request.depots]
        + [target.y for target in request.targets]
        + [vertex.y for zone in request.no_fly_zones for vertex in zone.vertices]
        + [drone.y for drone in request.drones]
        + [1000.0],
    )
    max_z = max(
        [depot.z for depot in request.depots]
        + [target.z for target in request.targets]
        + [zone.maxAltitude for zone in request.no_fly_zones]
        + [drone.maxAltitude for drone in request.drones]
        + [150.0],
    )
    return (int(max_x + 50), int(max_y + 50), int(max_z + 20))


def load_scene_bundle(path: Path) -> dict[str, Any]:
    """读取 JSON；支持 `pocas_scene_bundle`（取 `scene`）或已是 scene 根对象。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("packageType") == "pocas_scene_bundle" and isinstance(raw.get("scene"), dict):
        return raw["scene"]
    if isinstance(raw.get("scene"), dict):
        return raw["scene"]
    return raw


def algorithm_request_from_scene(
    scene: dict[str, Any],
    *,
    planner_external_id: str = "ovs",
    scheduler_external_id: str = "lmta",
    use_scene_algorithms: bool = False,
    extra_planning: dict[str, Any] | None = None,
) -> AlgorithmRequest:
    """
    由 scene 构造 `AlgorithmRequest`。

    - use_scene_algorithms=False：使用 planner_external_id / scheduler_external_id（解耦 CLI 指定）。
    - use_scene_algorithms=True：优先使用 scene 内 `algorithmId` 与 `schedulingId`（与场景文件耦合）。
    """
    planning_raw: dict[str, Any] = dict(scene.get("planningConfig") or {})
    if extra_planning:
        for k, v in extra_planning.items():
            if k in ("schedulerParameters", "plannerParameters") and isinstance(v, dict):
                inner = dict(planning_raw.get(k) or {})
                inner.update(v)
                planning_raw[k] = inner
            else:
                planning_raw[k] = v

    planning_raw.setdefault("schedulerParameters", {})
    planning_raw.setdefault("plannerParameters", {})
    planning_config = PlanningConfig.model_validate(planning_raw)

    if use_scene_algorithms:
        algorithm_id = str(scene.get("algorithmId") or planner_external_id).strip()
        scheduling_id = str(scene.get("schedulingId") or scheduler_external_id).strip()
    else:
        algorithm_id = str(planner_external_id).strip()
        scheduling_id = str(scheduler_external_id).strip()

    payload: dict[str, Any] = {
        "algorithmId": algorithm_id,
        "schedulingId": scheduling_id,
        "maxOrdersPerDrone": scene.get("maxOrdersPerDrone", -1),
        "drones": scene["drones"],
        "orders": scene["orders"],
        "targets": scene["targets"],
        "depots": scene["depots"],
        "no_fly_zones": scene.get("no_fly_zones", []),
        "executionStates": scene["executionStates"],
        "planningConfig": planning_config.model_dump(mode="json"),
    }
    return AlgorithmRequest.model_validate(payload)


def _backfill_missing_depots(
    core_depots: dict[str, Any],
    core_drones: list[Any],
) -> None:
    missing = {d.depotId for d in core_drones if d.depotId and d.depotId not in core_depots}
    if not missing:
        return
    from backend.models.domain import GeoPoint as CoreGeoPoint

    for d in core_drones:
        if d.depotId in missing:
            core_depots[d.depotId] = CoreGeoPoint(
                x=d.currentLocation.x,
                y=d.currentLocation.y,
                z=d.currentLocation.z,
            )


def run_scene_dispatch(
    request: AlgorithmRequest,
    *,
    enable_planning_trace: bool = False,
    evaluation_planner_external_id: str | None = None,
) -> dict[str, Any]:
    """
    执行一次完整调度。

    evaluation_planner_external_id：估价用规划器外部 ID；省略则由 DispatchEngine 按主规划器自动选择。
    """
    preprocess_started = perf_counter()
    core_drones = to_core_drones(request.drones, request.executionStates, request.planningConfig)
    core_targets = to_core_targets(request.targets, request.orders)
    core_constraints = to_core_no_fly_zones(request.no_fly_zones)
    core_depots = to_core_depots(request.depots)
    _backfill_missing_depots(core_depots, core_drones)
    preprocess_time = perf_counter() - preprocess_started

    planner_resolved = resolve_external_to_internal("planner", request.algorithmId)
    scheduler_resolved = resolve_external_to_internal("scheduler", request.schedulingId)
    planner_cls = try_load_planner_class(request.algorithmId, alternate_key=planner_resolved)
    scheduler_cls = try_load_scheduler_class(request.schedulingId, alternate_key=scheduler_resolved)

    planner_engine_name = planner_resolved if planner_cls is None else request.algorithmId
    scheduler_engine_name = scheduler_resolved if scheduler_cls is None else request.schedulingId
    planner_label = f"plugin:{request.algorithmId}" if planner_cls is not None else planner_resolved
    scheduler_label = f"plugin:{request.schedulingId}" if scheduler_cls is not None else scheduler_resolved

    eval_internal: str | None = None
    if evaluation_planner_external_id:
        eval_internal = resolve_external_to_internal("planner", evaluation_planner_external_id)

    space_limits = build_space_limits(request)

    ts = max(2, int(request.planningConfig.timeStep or 20))
    line_sample = max(6.0, min(float(request.planningConfig.timeStep or 20), 12.0))
    final_line = max(0.5, min(float(request.planningConfig.timeStep or 20) / 10.0, 2.0))

    engine = DispatchEngine(
        constraints=core_constraints,
        limits=space_limits,
        planner_name=planner_engine_name,
        evaluation_planner_name=eval_internal,
        scheduler_name=scheduler_engine_name,
        planner_resolution=ts,
        line_sample_step=line_sample,
        final_line_sample_step=final_line,
        enable_planning_trace=enable_planning_trace,
        planner_parameters=dict(request.planningConfig.plannerParameters or {}),
        scheduler_parameters=dict(request.planningConfig.schedulerParameters or {}),
        launch_mode=request.planningConfig.launchMode,
        launch_interval=request.planningConfig.launchInterval,
        cancel_check=None,
        progress_callback=None,
        planner_class=planner_cls,
        scheduler_class=scheduler_cls,
    )

    run_started = perf_counter()
    raw_routes = engine.run(core_drones, core_targets, core_depots)
    run_time = perf_counter() - run_started

    assigned_ids: set[str] = set()
    for route in raw_routes.values():
        for pt in route:
            tid = getattr(pt, "taskId", None)
            if tid and tid != "HOVER_FOR_LANDING":
                assigned_ids.add(str(tid))

    all_target_ids = [t.id for t in core_targets]
    unassigned = [tid for tid in all_target_ids if tid not in assigned_ids]

    return {
        "algorithm_id_external": request.algorithmId,
        "scheduling_id_external": request.schedulingId,
        "planner_name_internal": planner_label,
        "scheduler_name_internal": scheduler_label,
        "evaluation_planner_name_internal": getattr(engine, "evaluation_planner_name", None),
        "space_limits": space_limits,
        "preprocess_time_s": preprocess_time,
        "engine_run_time_s": run_time,
        "total_elapsed_s": preprocess_time + run_time,
        "targets_total": len(all_target_ids),
        "targets_assigned": len(assigned_ids),
        "unassigned_target_ids": unassigned,
        "planner_stats": engine.get_planner_stats(),
        "phase_timings": dict(getattr(engine, "last_phase_timings", {}) or {}),
        "routes": raw_routes,
    }


def compute_route_aggregate_metrics(routes: dict[str, Any]) -> dict[str, Any]:
    """
    根据引擎返回的航迹（PathPoint 列表）统计路径长度、点数、makespan 等。
    """
    total_length = 0.0
    total_points = 0
    makespan_s = 0.0
    per_drone: list[tuple[str, float, int, float]] = []

    for drone_id, pts in routes.items():
        if not pts:
            per_drone.append((str(drone_id), 0.0, 0, 0.0))
            continue
        seg = 0.0
        for i in range(1, len(pts)):
            a = pts[i - 1].location
            b = pts[i].location
            seg += sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        end_t = float(getattr(pts[-1], "time", 0.0) or 0.0)
        total_length += seg
        total_points += len(pts)
        makespan_s = max(makespan_s, end_t)
        per_drone.append((str(drone_id), seg, len(pts), end_t))

    per_drone.sort(key=lambda x: x[1], reverse=True)
    return {
        "total_path_length_m": total_length,
        "total_path_points": total_points,
        "makespan_s": makespan_s,
        "drone_count": len(routes),
        "drones_with_nonempty_route": sum(1 for _, L, n, _ in per_drone if n > 0),
        "per_drone_sorted_by_length": per_drone,
    }


def _flatten_stats(prefix: str, obj: Any, lines: list[str], indent: str = "  ") -> None:
    if isinstance(obj, dict):
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            v = obj[k]
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            _flatten_stats(key, v, lines, indent)
    elif isinstance(obj, (list, tuple)):
        lines.append(f"{indent}{prefix}: {obj!r}")
    else:
        lines.append(f"{indent}{prefix}: {obj}")


def format_dispatch_metrics_report(
    *,
    scene_path: Path,
    dispatch_out: dict[str, Any],
    route_agg: dict[str, Any],
    preset_name: str | None = None,
    evaluation_planner_external: str | None = None,
    max_unassigned_list: int = 80,
    max_per_drone_lines: int = 64,
) -> str:
    """
    生成「算法：」开头的纯文本评价报告（UTF-8 一行一条指标，便于存档对比）。
    """
    lines: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    sch_ext = dispatch_out.get("scheduling_id_external", "")
    pln_ext = dispatch_out.get("algorithm_id_external", "")
    sch_in = dispatch_out.get("scheduler_name_internal", "")
    pln_in = dispatch_out.get("planner_name_internal", "")
    eval_in = dispatch_out.get("evaluation_planner_name_internal")

    lines.append("算法：")
    lines.append(f"  任务分配（外部 ID）: {sch_ext}")
    lines.append(f"  任务分配（内部注册名）: {sch_in}")
    lines.append(f"  路径规划（外部 ID）: {pln_ext}")
    lines.append(f"  路径规划（内部注册名）: {pln_in}")
    if eval_in:
        lines.append(f"  估价规划器（内部）: {eval_in}")
    if evaluation_planner_external:
        lines.append(f"  估价规划器（外部，请求参数）: {evaluation_planner_external}")
    if preset_name:
        lines.append(f"  预设名称: {preset_name}")
    lines.append("")

    lines.append("场景与运行环境：")
    lines.append(f"  场景文件: {scene_path.resolve()}")
    lines.append(f"  报告生成时间(UTC): {now}")
    lim = dispatch_out.get("space_limits")
    if lim is not None:
        lines.append(f"  空间边界 (x_max, y_max, z_max): {lim}")
    lines.append("")

    tt = int(dispatch_out.get("targets_total", 0))
    ta = int(dispatch_out.get("targets_assigned", 0))
    rate = (100.0 * ta / tt) if tt else 0.0
    lines.append("任务分配结果：")
    lines.append(f"  任务总数: {tt}")
    lines.append(f"  已分配任务数: {ta}")
    lines.append(f"  未分配任务数: {tt - ta}")
    lines.append(f"  任务分配率(%): {rate:.2f}")
    un = list(dispatch_out.get("unassigned_target_ids") or [])
    if un:
        show = un[:max_unassigned_list]
        lines.append(f"  未分配任务 ID（至多列 {max_unassigned_list} 个）: {', '.join(show)}")
        if len(un) > max_unassigned_list:
            lines.append(f"  … 共 {len(un)} 个未分配，其余略")
    lines.append("")

    lines.append("耗时（秒）：")
    lines.append(f"  预处理: {float(dispatch_out.get('preprocess_time_s', 0.0)):.6f}")
    lines.append(f"  引擎求解（调度+航迹）: {float(dispatch_out.get('engine_run_time_s', 0.0)):.6f}")
    lines.append(f"  合计墙钟: {float(dispatch_out.get('total_elapsed_s', 0.0)):.6f}")
    lines.append("")

    lines.append("航迹几何统计：")
    lines.append(f"  总航迹长度(米，各机路径长度之和): {route_agg.get('total_path_length_m', 0.0):.3f}")
    lines.append(f"  路径点总数: {route_agg.get('total_path_points', 0)}")
    lines.append(f"  Makespan(秒，各机末点时间最大值): {route_agg.get('makespan_s', 0.0):.3f}")
    lines.append(f"  有路径的无人机数: {route_agg.get('drones_with_nonempty_route', 0)} / {route_agg.get('drone_count', 0)}")
    lines.append("")
    lines.append("各机航迹长度（米，降序，便于找长跑机）：")
    for row in list(route_agg.get("per_drone_sorted_by_length") or [])[:max_per_drone_lines]:
        did, length_m, npt, end_t = row
        lines.append(f"  {did}: length_m={length_m:.3f}, path_points={npt}, end_time_s={end_t:.3f}")
    if len(route_agg.get("per_drone_sorted_by_length") or []) > max_per_drone_lines:
        lines.append(f"  … 仅展示前 {max_per_drone_lines} 架，其余略")
    lines.append("")

    lines.append("路径规划器计数：")
    stats = dispatch_out.get("planner_stats") or {}
    if isinstance(stats, dict):
        for k in sorted(stats.keys()):
            lines.append(f"  {k}: {stats[k]}")
    else:
        lines.append(f"  {stats!r}")
    lines.append("")

    lines.append("调度各阶段耗时与其它计时（秒或引擎自定义字段）：")
    pt = dispatch_out.get("phase_timings") or {}
    if isinstance(pt, dict):
        flat: list[str] = []
        _flatten_stats("", pt, flat, indent="  ")
        lines.extend(flat if flat else ["  (无)"])
    else:
        lines.append(f"  {pt!r}")

    lines.append("")
    lines.append("— end of report —")
    return "\n".join(lines) + "\n"


def write_dispatch_metrics_txt(
    output_path: Path,
    *,
    scene_path: Path,
    dispatch_out: dict[str, Any],
    route_agg: dict[str, Any] | None = None,
    preset_name: str | None = None,
    evaluation_planner_external: str | None = None,
) -> Path:
    """写入 UTF-8 文本报告；``route_agg`` 省略时会现场计算。"""
    agg = route_agg if route_agg is not None else compute_route_aggregate_metrics(dispatch_out["routes"])
    text = format_dispatch_metrics_report(
        scene_path=scene_path,
        dispatch_out=dispatch_out,
        route_agg=agg,
        preset_name=preset_name,
        evaluation_planner_external=evaluation_planner_external,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path.resolve()


def run_scene_dispatch_on_json_file(
    json_path: Path,
    *,
    planner_external_id: str = "ovs",
    scheduler_external_id: str = "lmta",
    use_scene_algorithms: bool = False,
    extra_planning: dict[str, Any] | None = None,
    enable_planning_trace: bool = False,
    evaluation_planner_external_id: str | None = None,
) -> dict[str, Any]:
    """加载 bundle JSON 并运行调度。"""
    scene = load_scene_bundle(json_path)
    req = algorithm_request_from_scene(
        scene,
        planner_external_id=planner_external_id,
        scheduler_external_id=scheduler_external_id,
        use_scene_algorithms=use_scene_algorithms,
        extra_planning=extra_planning,
    )
    return run_scene_dispatch(
        req,
        enable_planning_trace=enable_planning_trace,
        evaluation_planner_external_id=evaluation_planner_external_id,
    )
