"""
算法统一接口数据模型。

本模块严格按 `algorithmApi_json.md` 中约定的命名提供请求/响应模型，
并负责把外部协议转换成当前后端调度内核使用的领域模型。
"""

from __future__ import annotations

from time import perf_counter
from typing import Dict, List, Optional, Literal, Any

from pydantic import BaseModel, Field, model_validator

from backend.models.domain import (
    Drone as CoreDrone,
    GeoPoint as CoreGeoPoint,
    PathPoint as CorePathPoint,
    SpatialConstraint as CoreConstraint,
    Task as CoreTask,
)


class XYPoint(BaseModel):
    """二维坐标点。"""

    x: float
    y: float


class Depot(BaseModel):
    """站点/仓库定义。"""

    id: str
    name: str
    x: float
    y: float
    z: float
    dronesCapacity: int
    launchInterval: Optional[float] = None
    serviceTime: Optional[float] = None
    type: Optional[str] = "depot"


class Drone(BaseModel):
    """无人机定义。"""

    id: str
    depotId: str
    name: str
    x: float
    y: float
    z: float
    defaultSpeed: float
    maxSpeed: float
    safetyDistance: float
    capacity: Optional[float] = 0.0
    maxFlightRange: float
    maxAltitude: float
    currentSpeed: Optional[float] = 0.0
    status: Optional[str] = "idle"


class Order(BaseModel):
    """订单定义。"""

    orderId: str
    priority: int
    weight: float
    finishTime: float
    description: Optional[str] = None


class Target(BaseModel):
    """目标点定义。"""

    id: str
    orderId: str
    x: float
    y: float
    z: float
    weight: float
    stayTime: float
    startTimeWindow: Optional[float] = 0.0
    endTimeWindow: Optional[float] = None
    dependencies: List[str] = Field(default_factory=list)
    requiredService: Optional[int] = None
    type: Optional[Literal["delivery", "pickup", "inspection"]] = None
    metadata: Optional[dict] = None


class NoFlyZone(BaseModel):
    """区域定义。协议名称保持 no_fly_zones 兼容，语义扩展为多类型空域区域。"""

    id: str
    vertices: List[XYPoint]
    startActiveTime: Optional[float] = None
    endActiveTime: Optional[float] = None
    maxAltitude: float
    minAltitude: float
    zoneKind: Literal["weather_slow", "solid_block", "airspace_block"] = "airspace_block"
    zoneName: Optional[str] = None
    weatherType: Optional[str] = None
    speedFactor: Optional[float] = None
    allowPassThrough: Optional[bool] = None


class PathPoint(BaseModel):
    """路径点定义。"""

    x: float
    y: float
    z: float
    time: float
    autoContinue: bool = True


class PlanningConfig(BaseModel):
    """规划配置。"""

    timeStep: int = 60
    enableCollisionAvoidance: bool = True
    enablePlanningTrace: bool = True
    enableFastRouteEvaluation: bool = True
    evaluationAlgorithmId: Optional[str] = None
    freezeHorizon: Optional[float] = None
    conflictMode: Literal["spatiotemporal", "spatial"] = "spatiotemporal"
    # 求解范式：heuristic(经典启发式) / solver(求解器) / hybrid(分治混合)
    solverMode: Literal["heuristic", "solver", "hybrid"] = "heuristic"
    # 策略版本，便于灰度与回放一致性
    strategyVersion: str = "v1_heuristic"
    weatherType: Literal["clear", "cloudy", "windy", "rain", "fog", "storm"] = "clear"
    weatherSpeedFactor: Optional[float] = None
    launchMode: Literal["serial", "parallel", "interval"] = "serial"
    launchInterval: Optional[float] = None
    returnToDepotMode: Literal["required", "optional", "forbidden"] = "required"
    schedulerParameters: Dict[str, Any] = Field(default_factory=dict)
    plannerParameters: Dict[str, Any] = Field(default_factory=dict)


class DroneExecutionState(BaseModel):
    """无人机执行状态。"""

    droneId: str
    x: float
    y: float
    z: float
    currentTime: float
    remainingBattery: Optional[float] = None
    remainingRange: Optional[float] = None
    assignedTargetIds: List[str] = Field(default_factory=list)
    completedTargetIds: List[str] = Field(default_factory=list)
    remainingPath: List[PathPoint] = Field(default_factory=list)
    status: Optional[str] = "idle"


class DronePlanStatistics(BaseModel):
    """单架无人机规划统计信息。"""

    totalDistance: float
    totalTime: float
    waitingTime: float = 0.0
    serviceTime: float = 0.0
    estimatedEnergyUsed: float = 0.0


class DronePlanResult(BaseModel):
    """单架无人机规划结果。"""

    droneId: str
    success: bool
    startTime: float
    endTime: float
    targetCount: int
    targetIds: List[str]
    path: List[PathPoint]
    statistics: DronePlanStatistics


class PlannerPerformance(BaseModel):
    """路径规划器性能统计。"""

    routeRequests: int = 0
    cacheHits: int = 0
    cacheMisses: int = 0
    schedulerRouteRequests: int = 0
    schedulerCacheHits: int = 0
    schedulerCacheMisses: int = 0
    finalRouteRequests: int = 0
    finalCacheHits: int = 0
    finalCacheMisses: int = 0


class DispatchPhaseTimings(BaseModel):
    """调度各阶段耗时拆分。"""

    preprocessTime: float = 0.0
    schedulerTime: float = 0.0
    clusterBuildTime: float = 0.0
    clusterSolveTime: float = 0.0
    serialFallbackSolveTime: float = 0.0
    clusterMergeTime: float = 0.0
    candidateExactRerankTime: float = 0.0
    candidateExactGateTime: float = 0.0
    localRepairTime: float = 0.0
    globalRepairTime: float = 0.0
    pathPlanningTime: float = 0.0
    schedulerRouteEvaluationTime: float = 0.0
    finalPathPlanningTime: float = 0.0
    finalValidationTime: float = 0.0
    collisionAnalysisTime: float = 0.0
    responseBuildTime: float = 0.0
    totalTime: float = 0.0


class AssignmentReason(BaseModel):
    """任务分配解释条目。"""

    type: Literal["distance", "capacity", "time_window", "risk", "other"] = "other"
    label: str
    value: str
    score: Optional[float] = None


class AssignmentInsight(BaseModel):
    """单架无人机的分配解释摘要。"""

    droneId: str
    assignedTargets: int = 0
    totalAssignedWeight: float = 0.0
    avgLegDistance: float = 0.0
    timeWindowHitRate: float = 0.0
    reasons: List[AssignmentReason] = Field(default_factory=list)


class FailureReasonSummary(BaseModel):
    """失败原因聚合统计。"""

    category: str
    count: int = 0
    examples: List[str] = Field(default_factory=list)


class AlgorithmRequest(BaseModel):
    """文档约定的统一请求体。"""

    algorithmId: str
    schedulingId: str
    maxOrdersPerDrone: Optional[int] = -1
    drones: List[Drone]
    orders: List[Order]
    targets: List[Target]
    depots: List[Depot]
    no_fly_zones: List[NoFlyZone]
    executionStates: List[DroneExecutionState]
    planningConfig: PlanningConfig

    @model_validator(mode="after")
    def validate_references(self) -> "AlgorithmRequest":
        """校验实体之间的引用关系，避免错误协议直接进入调度内核。"""

        order_ids = {order.orderId for order in self.orders}
        drone_ids = {drone.id for drone in self.drones}
        target_ids = {target.id for target in self.targets}

        missing_order_ids = sorted({
            target.orderId
            for target in self.targets
            if target.orderId not in order_ids
        })
        if missing_order_ids:
            raise ValueError(f"targets 引用了不存在的 orderId: {', '.join(missing_order_ids)}")

        missing_drone_ids = sorted({
            state.droneId
            for state in self.executionStates
            if state.droneId not in drone_ids
        })
        if missing_drone_ids:
            raise ValueError(f"executionStates 引用了不存在的 droneId: {', '.join(missing_drone_ids)}")

        missing_dependency_ids = sorted({
            dependency_id
            for target in self.targets
            for dependency_id in target.dependencies
            if dependency_id not in target_ids
        })
        if missing_dependency_ids:
            raise ValueError(f"targets.dependencies 引用了不存在的 targetId: {', '.join(missing_dependency_ids)}")

        return self


class CollisionDetail(BaseModel):
    """结构化碰撞详情。"""

    id: str
    type: Literal["spatiotemporal", "spatial"]
    droneA: str
    droneB: str
    drones: List[str]
    minDistance: float
    time: Optional[float] = None
    timeRange: Optional[List[float]] = None
    location: Optional[Dict[str, float]] = None
    severity: Literal["critical", "warning", "info"] = "warning"
    description: str = ""


class ApiErrorDetail(BaseModel):
    """结构化错误明细。"""

    field: Optional[str] = None
    message: str
    entityId: Optional[str] = None
    type: Optional[str] = None


class ApiError(BaseModel):
    """统一错误结构。"""

    code: str
    summary: str
    details: List[ApiErrorDetail] = Field(default_factory=list)


class ScenarioValidationRequest(BaseModel):
    """随机场景合法性校验请求。"""

    algorithmId: Optional[str] = "astar_v2"
    schedulingId: Optional[str] = "insertion_heuristic"
    maxOrdersPerDrone: Optional[int] = -1
    drones: List[Drone] = Field(default_factory=list)
    orders: List[Order] = Field(default_factory=list)
    targets: List[Target] = Field(default_factory=list)
    depots: List[Depot] = Field(default_factory=list)
    no_fly_zones: List[NoFlyZone] = Field(default_factory=list)
    executionStates: List[DroneExecutionState] = Field(default_factory=list)
    planningConfig: PlanningConfig = Field(default_factory=PlanningConfig)


class ScenarioValidationResponse(BaseModel):
    """随机场景合法性校验响应。"""

    success: bool
    error: Optional[ApiError] = None


class AlgorithmResponse(BaseModel):
    """文档约定的统一响应体。"""

    success: bool
    assignments: Dict[str, DronePlanResult]
    unassignedTasks: List[str] = Field(default_factory=list)
    totalDistance: float
    completedOrders: int
    executionTime: float
    error: str | ApiError = ""
    plannerPerformance: PlannerPerformance = Field(default_factory=PlannerPerformance)
    phaseTimings: DispatchPhaseTimings = Field(default_factory=DispatchPhaseTimings)
    collisions: List[CollisionDetail] = Field(default_factory=list)
    planningTrace: List[Dict[str, Any]] = Field(default_factory=list)
    assignmentInsights: Dict[str, AssignmentInsight] = Field(default_factory=dict)
    failureReasons: Dict[str, FailureReasonSummary] = Field(default_factory=dict)


def build_execution_state_map(execution_states: List[DroneExecutionState]) -> Dict[str, DroneExecutionState]:
    """按无人机 ID 建立执行状态索引。"""

    return {state.droneId: state for state in execution_states}


def to_core_depots(depots: List[Depot]) -> Dict[str, CoreGeoPoint]:
    """把外部站点数组转换成内核站点字典。"""

    return {
        depot.id: CoreGeoPoint(x=depot.x, y=depot.y, z=depot.z)
        for depot in depots
    }


def _resolve_weather_speed_factor(planning_config: PlanningConfig | None) -> float:
    if planning_config is None:
        return 1.0

    explicit_factor = getattr(planning_config, "weatherSpeedFactor", None)
    if explicit_factor is not None:
        try:
            return max(0.05, float(explicit_factor))
        except (TypeError, ValueError):
            pass

    weather_type = str(getattr(planning_config, "weatherType", "clear") or "clear").strip().lower()
    weather_defaults = {
        "clear": 1.0,
        "cloudy": 1.0,
        "windy": 1.0,
        "rain": 0.85,
        "fog": 1.0,
        "storm": 1.0,
    }
    return float(weather_defaults.get(weather_type, 1.0))


def _resolve_return_to_depot_required(planning_config: PlanningConfig | None) -> bool:
    mode = str(getattr(planning_config, "returnToDepotMode", "required") or "required").strip().lower()
    return mode == "required"


def to_core_drones(
    drones: List[Drone],
    execution_states: List[DroneExecutionState],
    planning_config: PlanningConfig | None = None,
) -> List[CoreDrone]:
    """把外部无人机和执行状态合并后转换成内核无人机。"""

    execution_state_map = build_execution_state_map(execution_states)
    core_drones: List[CoreDrone] = []
    weather_speed_factor = _resolve_weather_speed_factor(planning_config)
    return_to_depot_required = _resolve_return_to_depot_required(planning_config)

    for drone in drones:
        execution_state = execution_state_map.get(drone.id)
        current_location = CoreGeoPoint(
            x=execution_state.x if execution_state else drone.x,
            y=execution_state.y if execution_state else drone.y,
            z=execution_state.z if execution_state else drone.z,
        )
        current_time = execution_state.currentTime if execution_state else 0.0
        remaining_range = (
            execution_state.remainingRange
            if execution_state and execution_state.remainingRange is not None
            else drone.maxFlightRange
        )
        battery = (
            execution_state.remainingBattery
            if execution_state and execution_state.remainingBattery is not None
            else 100.0
        )

        core_drones.append(
            CoreDrone(
                id=drone.id,
                depotId=drone.depotId,
                maxRange=drone.maxFlightRange,
                speed=max(0.1, drone.defaultSpeed * weather_speed_factor),
                capacity=drone.capacity or 0.0,
                returnToDepotRequired=return_to_depot_required,
                currentLocation=current_location,
                remainingRange=remaining_range,
                currentLoad=0.0,
                currentTime=current_time,
                battery=battery,
                maxFlightTime=None,
                energyPerMeter=0.01,
            )
        )

    return core_drones


def to_core_targets(targets: List[Target], orders: List[Order]) -> List[CoreTask]:
    """把文档中的 Target 转成当前内核任务。"""

    order_map = {order.orderId: order for order in orders}
    max_priority = max((order.priority for order in orders), default=1)
    order_target_sequence: Dict[str, int] = {}
    core_targets: List[CoreTask] = []

    for target in targets:
        order = order_map.get(target.orderId)
        order_target_sequence[target.orderId] = order_target_sequence.get(target.orderId, 0) + 1
        internal_priority = max_priority - (order.priority if order else 0)
        metadata = target.metadata or {}
        target_type = str(target.type or metadata.get("type") or "delivery").strip().lower()
        if target_type not in {"delivery", "pickup", "inspection"}:
            target_type = "delivery"
        end_time_window = (
            target.endTimeWindow
            if target.endTimeWindow is not None
            else (order.finishTime if order else 86400.0)
        )

        core_targets.append(
            CoreTask(
                id=target.id,
                type=target_type,
                location=CoreGeoPoint(x=target.x, y=target.y, z=target.z),
                weight=target.weight,
                timeWindow=(target.startTimeWindow or 0.0, end_time_window),
                serviceDuration=target.stayTime,
                is_sudden=False,
                priority=internal_priority,
                groupId=target.orderId,
                sequence=order_target_sequence[target.orderId] - 1,
                dependencies=target.dependencies,
                metadata=metadata,
            )
        )

    return core_targets


def to_core_no_fly_zones(no_fly_zones: List[NoFlyZone]) -> List[CoreConstraint]:
    """把外部区域转换成内核多边形约束。"""

    core_constraints: List[CoreConstraint] = []

    for no_fly_zone in no_fly_zones:
        vertices = [(vertex.x, vertex.y) for vertex in no_fly_zone.vertices]
        if len(vertices) < 3:
            continue
        x_values = [vertex[0] for vertex in vertices]
        y_values = [vertex[1] for vertex in vertices]
        zone_kind = str(no_fly_zone.zoneKind or "airspace_block").strip().lower()
        allow_pass_through = (
            bool(no_fly_zone.allowPassThrough)
            if no_fly_zone.allowPassThrough is not None
            else zone_kind == "weather_slow"
        )
        bounding_box = (
            min(x_values),
            min(y_values),
            no_fly_zone.minAltitude,
            max(x_values),
            max(y_values),
            no_fly_zone.maxAltitude,
        )
        core_constraints.append(
            CoreConstraint(
                id=no_fly_zone.id,
                kind="no_fly",
                shape="polygon",
                box=bounding_box,
                polygon=vertices,
                startActiveTime=no_fly_zone.startActiveTime,
                endActiveTime=no_fly_zone.endActiveTime,
                zoneKind=zone_kind,
                name=no_fly_zone.zoneName or no_fly_zone.id,
                weatherType=no_fly_zone.weatherType,
                speedFactor=no_fly_zone.speedFactor,
                allowPassThrough=allow_pass_through,
            )
        )

    return core_constraints


def to_algorithm_path_point(path_point: CorePathPoint) -> PathPoint:
    """把内核路径点转换成文档路径点。"""

    return PathPoint(
        x=path_point.location.x,
        y=path_point.location.y,
        z=path_point.location.z,
        time=path_point.time,
        autoContinue=True,
    )


def _classify_failure_reason(reason: str) -> str:
    text = (reason or "").strip()
    if not text:
        return "other"
    if "时间窗" in text:
        return "time_window"
    if "载重" in text or "容量" in text:
        return "capacity"
    if "航程" in text or "电池" in text:
        return "range"
    if "禁飞" in text or "无法到达" in text or "返回站点" in text:
        return "no_fly_zone"
    return "other"


def _build_failure_reason_summary(planning_trace: List[Dict[str, Any]]) -> Dict[str, FailureReasonSummary]:
    summary: Dict[str, FailureReasonSummary] = {}
    for item in planning_trace:
        if bool(item.get("valid", True)):
            continue
        reason = str(item.get("reason") or "")
        category = _classify_failure_reason(reason)
        if category not in summary:
            summary[category] = FailureReasonSummary(category=category, count=0, examples=[])
        summary[category].count += 1
        if reason and reason not in summary[category].examples and len(summary[category].examples) < 3:
            summary[category].examples.append(reason)
    return summary


def _build_assignment_insights(
    request: AlgorithmRequest,
    raw_routes: Dict[str, List[CorePathPoint]],
) -> Dict[str, AssignmentInsight]:
    target_map = {target.id: target for target in request.targets}
    drone_map = {drone.id: drone for drone in request.drones}

    insights: Dict[str, AssignmentInsight] = {}
    for drone in request.drones:
        route = raw_routes.get(drone.id, [])
        target_ids = [
            path_point.taskId
            for path_point in route
            if path_point.taskId and path_point.taskId != "HOVER_FOR_LANDING"
        ]
        unique_target_ids = list(dict.fromkeys(target_ids))

        total_weight = sum((target_map.get(target_id).weight for target_id in unique_target_ids if target_map.get(target_id)), 0.0)
        total_distance = sum(
            (
                (route[index + 1].location.x - route[index].location.x) ** 2
                + (route[index + 1].location.y - route[index].location.y) ** 2
                + (route[index + 1].location.z - route[index].location.z) ** 2
            ) ** 0.5
            for index in range(len(route) - 1)
        )
        avg_leg_distance = total_distance / max(len(route) - 1, 1)

        arrival_map: Dict[str, float] = {}
        for point in route:
            if point.taskId and point.taskId != "HOVER_FOR_LANDING" and point.taskId not in arrival_map:
                arrival_map[point.taskId] = point.time

        window_hits = 0
        for target_id in unique_target_ids:
            target = target_map.get(target_id)
            if not target:
                continue
            arrival = arrival_map.get(target_id, 0.0)
            end_window = target.endTimeWindow if target.endTimeWindow is not None else float("inf")
            if arrival <= end_window:
                window_hits += 1
        hit_rate = window_hits / max(len(unique_target_ids), 1)

        capacity = float(drone_map.get(drone.id).capacity or 0.0) if drone_map.get(drone.id) else 0.0
        capacity_ratio = (total_weight / capacity) if capacity > 0 else 0.0
        time_window_ratio = hit_rate
        distance_score = 1.0 / (1.0 + avg_leg_distance / 1000.0)

        insight = AssignmentInsight(
            droneId=drone.id,
            assignedTargets=len(unique_target_ids),
            totalAssignedWeight=round(total_weight, 3),
            avgLegDistance=round(avg_leg_distance, 3),
            timeWindowHitRate=round(hit_rate, 4),
            reasons=[
                AssignmentReason(
                    type="distance",
                    label="平均航段距离",
                    value=f"{avg_leg_distance:.2f} m",
                    score=round(distance_score, 4),
                ),
                AssignmentReason(
                    type="capacity",
                    label="载重利用率",
                    value=f"{(capacity_ratio * 100):.1f}%",
                    score=round(min(max(capacity_ratio, 0.0), 1.5), 4),
                ),
                AssignmentReason(
                    type="time_window",
                    label="时间窗命中率",
                    value=f"{(time_window_ratio * 100):.1f}%",
                    score=round(min(max(time_window_ratio, 0.0), 1.0), 4),
                ),
            ],
        )
        insights[drone.id] = insight
    return insights


def build_algorithm_response(
    raw_routes: Dict[str, List[CorePathPoint]],
    request: AlgorithmRequest,
    started_at: float,
    planner_stats: Optional[Dict[str, int]] = None,
    phase_timings: Optional[Dict[str, float]] = None,
    collisions: Optional[List[CollisionDetail | Dict[str, object]]] = None,
    planning_trace: Optional[List[Dict[str, Any]]] = None,
    execution_time: Optional[float] = None,
    include_detailed_fields: bool = True,
    include_planning_trace: bool = False,
) -> AlgorithmResponse:
    """把内核调度结果包装成文档约定的统一响应。"""

    target_to_order_map = {target.id: target.orderId for target in request.targets}
    assignments: Dict[str, DronePlanResult] = {}
    total_distance = 0.0
    completed_order_ids: set[str] = set()
    assigned_target_ids: set[str] = set()

    for drone in request.drones:
        route = raw_routes.get(drone.id, [])
        target_ids = [
            path_point.taskId
            for path_point in route
            if path_point.taskId and path_point.taskId != "HOVER_FOR_LANDING"
        ]
        unique_target_ids = list(dict.fromkeys(target_ids))
        assigned_target_ids.update(unique_target_ids)
        for target_id in unique_target_ids:
            order_id = target_to_order_map.get(target_id)
            if order_id:
                completed_order_ids.add(order_id)

        path = [to_algorithm_path_point(path_point) for path_point in route]
        start_time = path[0].time if path else 0.0
        end_time = path[-1].time if path else 0.0
        total_time = max(0.0, end_time - start_time)
        total_service_time = sum(
            target.stayTime
            for target in request.targets
            if target.id in unique_target_ids
        )
        total_waiting_time = sum(
            max(0.0, path[index + 1].time - path[index].time)
            for index in range(len(path) - 1)
            if index < len(route) - 1 and route[index + 1].action == "wait"
        )
        total_distance_for_drone = sum(
            (
                (path[index + 1].x - path[index].x) ** 2
                + (path[index + 1].y - path[index].y) ** 2
                + (path[index + 1].z - path[index].z) ** 2
            ) ** 0.5
            for index in range(len(path) - 1)
        )
        total_distance += total_distance_for_drone

        assignments[drone.id] = DronePlanResult(
            droneId=drone.id,
            success=len(path) > 0,
            startTime=start_time,
            endTime=end_time,
            targetCount=len(unique_target_ids),
            targetIds=unique_target_ids,
            path=path,
            statistics=DronePlanStatistics(
                totalDistance=total_distance_for_drone,
                totalTime=total_time,
                waitingTime=total_waiting_time,
                serviceTime=total_service_time,
                estimatedEnergyUsed=total_distance_for_drone * 0.01,
            ),
        )

    planning_trace_list = planning_trace or []
    unassigned_target_ids = [
        target.id for target in request.targets
        if target.id not in assigned_target_ids
    ]
    return AlgorithmResponse(
        success=True,
        assignments=assignments,
        unassignedTasks=unassigned_target_ids,
        totalDistance=total_distance,
        completedOrders=len(completed_order_ids),
        executionTime=execution_time if execution_time is not None else perf_counter() - started_at,
        error="",
        plannerPerformance=PlannerPerformance(**(planner_stats or {})),
        phaseTimings=DispatchPhaseTimings(**(phase_timings or {})),
        collisions=[
            collision if isinstance(collision, CollisionDetail) else CollisionDetail(**collision)
            for collision in (collisions or [])
        ],
        planningTrace=planning_trace_list if include_planning_trace else [],
        assignmentInsights=_build_assignment_insights(request, raw_routes) if include_detailed_fields else {},
        failureReasons=_build_failure_reason_summary(planning_trace_list) if include_detailed_fields else {},
    )
