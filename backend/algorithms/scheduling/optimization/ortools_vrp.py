from __future__ import annotations

import logging
import math
import sys
from typing import Dict, Iterable, List, Sequence

import numpy as np

from backend.algorithms.registry import AlgorithmRegistry
from backend.algorithms.scheduling.optimization.insertion import InsertionScheduler
from backend.algorithms.utils.cpp_bridge import BLOCKED_DISTANCE, get_matrix
from backend.models.domain import Drone, SpatialConstraint, Task

logger = logging.getLogger(__name__)
_ORTOOLS_IMPORTED = False
_ORTOOLS_IMPORT_ERROR = None


def _ensure_ortools():
    global _ORTOOLS_IMPORTED, _ORTOOLS_IMPORT_ERROR
    if _ORTOOLS_IMPORTED:
        return

    try:
        from ortools.constraint_solver import pywrapcp as _pywrapcp  # type: ignore
        from ortools.constraint_solver import routing_enums_pb2 as _routing_enums_pb2  # type: ignore
        globals()["pywrapcp"] = _pywrapcp
        globals()["routing_enums_pb2"] = _routing_enums_pb2
        _ORTOOLS_IMPORTED = True
        _ORTOOLS_IMPORT_ERROR = None
        return
    except Exception as exc:
        _ORTOOLS_IMPORT_ERROR = exc

    local_candidates = [
        "/home/moon/worspace/3-27/or-tools",
        "/home/moon/worspace/3-27/or-tools/python",
        "/home/moon/worspace/3-27/or-tools/build/python",
    ]
    for path in local_candidates:
        if path not in sys.path:
            sys.path.append(path)
        try:
            from ortools.constraint_solver import pywrapcp as _pywrapcp  # type: ignore
            from ortools.constraint_solver import routing_enums_pb2 as _routing_enums_pb2  # type: ignore
            globals()["pywrapcp"] = _pywrapcp
            globals()["routing_enums_pb2"] = _routing_enums_pb2
            _ORTOOLS_IMPORTED = True
            _ORTOOLS_IMPORT_ERROR = None
            return
        except Exception as exc:
            _ORTOOLS_IMPORT_ERROR = exc

    raise ImportError(f"Unable to import ortools: {_ORTOOLS_IMPORT_ERROR}")


def _distance_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _constraint_to_polygon(constraint: SpatialConstraint, samples: int = 16) -> list[tuple[float, float]]:
    if constraint.shape == "polygon" and constraint.polygon:
        return [(float(x), float(y)) for x, y in constraint.polygon]

    if constraint.shape == "box" and constraint.box:
        min_x, min_y, _min_z, max_x, max_y, _max_z = constraint.box
        return [
            (float(min_x), float(min_y)),
            (float(max_x), float(min_y)),
            (float(max_x), float(max_y)),
            (float(min_x), float(max_y)),
        ]

    if constraint.shape == "cylinder" and constraint.cylinder:
        center_x, center_y, radius, _min_z, _max_z = constraint.cylinder
        if radius <= 0:
            return []
        polygon: list[tuple[float, float]] = []
        for i in range(max(8, samples)):
            theta = (2.0 * math.pi * i) / float(max(8, samples))
            polygon.append(
                (
                    float(center_x + radius * math.cos(theta)),
                    float(center_y + radius * math.sin(theta)),
                )
            )
        return polygon

    return []


def _build_no_fly_zone_payload(constraints: Iterable[SpatialConstraint]) -> list[dict]:
    zones: list[dict] = []
    for constraint in constraints:
        vertices = _constraint_to_polygon(constraint)
        if len(vertices) < 3:
            continue
        zones.append(
            {
                "id": constraint.id,
                "vertices": [{"x": x, "y": y} for x, y in vertices],
            }
        )
    return zones


def _euclidean_distance_matrix(points_xy: Sequence[tuple[float, float]]) -> list[list[float]]:
    n = len(points_xy)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _distance_xy(points_xy[i], points_xy[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


def solve_vrp_cluster(
    drones_payload: Sequence[dict],
    tasks_payload: Sequence[dict],
    no_fly_zones_payload: Sequence[dict],
    *,
    time_limit_s: int = 5,
    max_orders_per_drone: int = -1,
    first_solution_strategy: str = "PATH_CHEAPEST_ARC",
    local_search_metaheuristic: str = "GUIDED_LOCAL_SEARCH",
    use_time_windows: bool = False,
    use_dependency_precedence: bool = False,
    use_capacity_constraints: bool = False,
    allow_node_dropping: bool = True,
    disjunction_penalty_mode: str = "fixed",
) -> dict[str, list[str]]:
    """
    Solve a single cluster VRP and return task-id routes keyed by drone-id.

    This function uses plain Python payloads, making it safe for multiprocessing.
    """
    routes: dict[str, list[str]] = {str(d["id"]): [] for d in drones_payload}
    _ensure_ortools()
    if not drones_payload or not tasks_payload:
        return routes

    # Node layout: [depots-for-drones] + [tasks]
    depot_nodes: list[tuple[float, float]] = []
    starts: list[int] = []
    ends: list[int] = []

    for idx, drone in enumerate(drones_payload):
        depot_nodes.append((float(drone["start_x"]), float(drone["start_y"])))
        starts.append(idx)
        ends.append(idx)

    task_nodes = [(float(task["x"]), float(task["y"])) for task in tasks_payload]
    all_nodes = depot_nodes + task_nodes
    task_node_indices = [len(depot_nodes) + i for i in range(len(task_nodes))]

    try:
        matrix = get_matrix(all_nodes, list(no_fly_zones_payload))
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning("C++ distance matrix unavailable, fallback to euclidean: %s", exc)
        matrix = _euclidean_distance_matrix(all_nodes)

    int_matrix = np.asarray(matrix, dtype=np.float64)
    int_matrix = np.where(np.isfinite(int_matrix), int_matrix, BLOCKED_DISTANCE)
    int_matrix = np.rint(int_matrix).astype(np.int64)
    int_matrix = np.clip(int_matrix, 0, np.int64(9_000_000_000_000_000_000))

    manager = pywrapcp.RoutingIndexManager(
        int(len(all_nodes)),
        int(len(drones_payload)),
        starts,
        ends,
    )
    routing = pywrapcp.RoutingModel(manager)

    def transit_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(int_matrix[from_node, to_node])

    transit_idx = routing.RegisterTransitCallback(transit_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    for idx, node in enumerate(task_node_indices):
        penalty = int(1_000_000_000)
        if allow_node_dropping:
            if str(disjunction_penalty_mode).lower() == "prize":
                priority = int(tasks_payload[idx].get("priority", 1) or 1)
                if priority <= 0:
                    priority = 1
                weight = float(tasks_payload[idx].get("weight", 1.0) or 1.0)
                reward = (priority * 10_000) + int(max(0.0, weight) * 1000.0)
                penalty = int(max(500, reward))
            else:
                penalty = int(1_000_000_000)
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    if max_orders_per_drone and max_orders_per_drone > 0:
        def visit_callback(index: int) -> int:
            node = manager.IndexToNode(index)
            return 1 if node in task_node_indices else 0

        visit_idx = routing.RegisterUnaryTransitCallback(visit_callback)
        routing.AddDimensionWithVehicleCapacity(
            visit_idx,
            0,
            [int(max_orders_per_drone)] * len(drones_payload),
            True,
            "TaskCount",
        )

    if use_capacity_constraints:
        capacity_scale = 1000.0

        def demand_callback(index: int) -> int:
            node = manager.IndexToNode(index)
            if node in task_node_indices:
                task_pos = node - len(depot_nodes)
                weight = float(tasks_payload[task_pos].get("weight", 1.0) or 0.0)
                return int(max(0.0, weight) * capacity_scale)
            return 0

        demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
        vehicle_capacities: list[int] = []
        for drone in drones_payload:
            cap = float(drone.get("capacity", 0.0) or 0.0)
            if cap <= 0:
                cap = 1_000_000.0
            vehicle_capacities.append(int(cap * capacity_scale))
        routing.AddDimensionWithVehicleCapacity(
            demand_idx,
            0,
            vehicle_capacities,
            True,
            "Capacity",
        )

    time_dimension = None
    if use_time_windows:
        max_tw_end = 0
        for task in tasks_payload:
            tw_end = int(task.get("tw_end", 0) or 0)
            if tw_end > max_tw_end:
                max_tw_end = tw_end
        horizon = max(60, max_tw_end + 600)
        routing.AddDimension(
            transit_idx,
            int(max(0, horizon // 4)),
            int(horizon),
            False,
            "Time",
        )
        time_dimension = routing.GetDimensionOrDie("Time")
        for idx, task in enumerate(tasks_payload):
            node = task_node_indices[idx]
            index = manager.NodeToIndex(node)
            tw_start = int(task.get("tw_start", 0) or 0)
            tw_end = int(task.get("tw_end", horizon) or horizon)
            tw_end = max(tw_start, tw_end)
            time_dimension.CumulVar(index).SetRange(tw_start, tw_end)
        for vehicle_idx in range(len(drones_payload)):
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(vehicle_idx)))
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(vehicle_idx)))

    if use_dependency_precedence and tasks_payload:
        task_id_to_node = {
            str(tasks_payload[idx]["id"]): task_node_indices[idx]
            for idx in range(len(tasks_payload))
        }
        if time_dimension is None:
            one_callback = routing.RegisterTransitCallback(lambda _from_idx, _to_idx: 1)
            routing.AddDimension(
                one_callback,
                0,
                max(2, len(all_nodes) + 2),
                True,
                "VisitOrder",
            )
            time_dimension = routing.GetDimensionOrDie("VisitOrder")

        for idx, task in enumerate(tasks_payload):
            task_node = task_node_indices[idx]
            task_index = manager.NodeToIndex(task_node)
            for dep_task_id in task.get("dependencies", []) or []:
                dep_node = task_id_to_node.get(str(dep_task_id))
                if dep_node is None:
                    continue
                dep_index = manager.NodeToIndex(dep_node)
                routing.solver().Add(routing.VehicleVar(dep_index) == routing.VehicleVar(task_index))
                routing.solver().Add(
                    time_dimension.CumulVar(dep_index) <= time_dimension.CumulVar(task_index)
                )

    first_strategy_map = {
        "PATH_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        "PARALLEL_CHEAPEST_INSERTION": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        "AUTOMATIC": routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
    }
    local_search_map = {
        "GUIDED_LOCAL_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        "SIMULATED_ANNEALING": routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
        "TABU_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
        "AUTOMATIC": routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
    }

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = first_strategy_map.get(
        str(first_solution_strategy).upper(),
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    )
    search_parameters.local_search_metaheuristic = local_search_map.get(
        str(local_search_metaheuristic).upper(),
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    )
    search_parameters.time_limit.seconds = int(max(1, time_limit_s))
    search_parameters.log_search = False

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        logger.warning("OR-Tools returned no solution for cluster, fallback empty assignment.")
        return routes

    node_to_task_id = {
        task_node_indices[i]: str(tasks_payload[i]["id"])
        for i in range(len(tasks_payload))
    }
    drone_ids = [str(d["id"]) for d in drones_payload]

    for vehicle_idx, drone_id in enumerate(drone_ids):
        index = routing.Start(vehicle_idx)
        vehicle_route: list[str] = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node in node_to_task_id:
                vehicle_route.append(node_to_task_id[node])
            index = solution.Value(routing.NextVar(index))
        routes[drone_id] = vehicle_route

    return routes


@AlgorithmRegistry.register_scheduler("ortools_vrp")
class ORToolsScheduler(InsertionScheduler):
    """
    OR-Tools VRP based scheduler.
    """

    ORTOOLS_TIME_LIMIT_S = 5
    FIRST_SOLUTION_STRATEGY = "PATH_CHEAPEST_ARC"
    LOCAL_SEARCH_METAHEURISTIC = "GUIDED_LOCAL_SEARCH"
    USE_TIME_WINDOWS = False
    USE_DEPENDENCY_PRECEDENCE = False
    USE_CAPACITY_CONSTRAINTS = False
    ALLOW_NODE_DROPPING = True
    DISJUNCTION_PENALTY_MODE = "fixed"

    def plan(self, current_drones: List[Drone], unassigned_tasks: List[Task]) -> Dict[str, List[Task]]:
        self.reset_route_profile_cache()
        routes: Dict[str, List[Task]] = {drone.id: [] for drone in current_drones}
        if not current_drones or not unassigned_tasks:
            return routes
        try:
            _ensure_ortools()
        except Exception as exc:
            logger.warning("OR-Tools unavailable, fallback to insertion heuristic: %s", exc)
            return super().plan(current_drones, unassigned_tasks)

        task_by_id = {task.id: task for task in unassigned_tasks}

        drones_payload = [
            {
                "id": drone.id,
                "start_x": float(drone.currentLocation.x),
                "start_y": float(drone.currentLocation.y),
                "capacity": float(drone.capacity),
            }
            for drone in current_drones
        ]
        tasks_payload = [
            {
                "id": task.id,
                "x": float(task.location.x),
                "y": float(task.location.y),
                "tw_start": float(task.timeWindow[0]) if task.timeWindow else 0.0,
                "tw_end": float(task.timeWindow[1]) if task.timeWindow else 86400.0,
                "dependencies": list(task.dependencies or []),
                "weight": float(task.weight),
                "priority": int(task.priority if task.priority is not None else 1),
            }
            for task in unassigned_tasks
        ]

        constraints = getattr(self.planner.env, "constraints", []) if getattr(self, "planner", None) else []
        no_fly_zones_payload = _build_no_fly_zone_payload(constraints)

        try:
            raw_routes = solve_vrp_cluster(
                drones_payload=drones_payload,
                tasks_payload=tasks_payload,
                no_fly_zones_payload=no_fly_zones_payload,
                time_limit_s=self.ORTOOLS_TIME_LIMIT_S,
                max_orders_per_drone=getattr(self, "max_orders_per_drone", -1),
                first_solution_strategy=self.FIRST_SOLUTION_STRATEGY,
                local_search_metaheuristic=self.LOCAL_SEARCH_METAHEURISTIC,
                use_time_windows=self.USE_TIME_WINDOWS,
                use_dependency_precedence=self.USE_DEPENDENCY_PRECEDENCE,
                use_capacity_constraints=self.USE_CAPACITY_CONSTRAINTS,
                allow_node_dropping=self.ALLOW_NODE_DROPPING,
                disjunction_penalty_mode=self.DISJUNCTION_PENALTY_MODE,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("OR-Tools scheduler failed, fallback to insertion heuristic: %s", exc)
            return super().plan(current_drones, unassigned_tasks)

        assigned_ids: set[str] = set()
        for drone in current_drones:
            ordered_tasks: list[Task] = []
            for task_id in raw_routes.get(drone.id, []):
                task = task_by_id.get(task_id)
                if task is None or task.id in assigned_ids:
                    continue
                ordered_tasks.append(task)

            routes[drone.id] = self._merge_ordered_tasks_for_drone(
                drone,
                routes.get(drone.id, []),
                ordered_tasks,
            )
            assigned_ids.update(task.id for task in routes[drone.id])

        leftovers = [task for task in unassigned_tasks if task.id not in assigned_ids]
        if leftovers:
            routes = self._repair_tasks_progressive(current_drones, routes, leftovers)

        return routes


@AlgorithmRegistry.register_scheduler("ortools_vrp_fast")
class ORToolsFastScheduler(ORToolsScheduler):
    ORTOOLS_TIME_LIMIT_S = 2
    FIRST_SOLUTION_STRATEGY = "PATH_CHEAPEST_ARC"
    LOCAL_SEARCH_METAHEURISTIC = "AUTOMATIC"


@AlgorithmRegistry.register_scheduler("ortools_vrp_parallel")
class ORToolsParallelInitScheduler(ORToolsScheduler):
    ORTOOLS_TIME_LIMIT_S = 4
    FIRST_SOLUTION_STRATEGY = "PARALLEL_CHEAPEST_INSERTION"
    LOCAL_SEARCH_METAHEURISTIC = "GUIDED_LOCAL_SEARCH"


@AlgorithmRegistry.register_scheduler("ortools_vrp_deep")
class ORToolsDeepScheduler(ORToolsScheduler):
    ORTOOLS_TIME_LIMIT_S = 12
    FIRST_SOLUTION_STRATEGY = "PATH_CHEAPEST_ARC"
    LOCAL_SEARCH_METAHEURISTIC = "TABU_SEARCH"


@AlgorithmRegistry.register_scheduler("ortools_vrp_tw")
class ORToolsTimeWindowsScheduler(ORToolsScheduler):
    ORTOOLS_TIME_LIMIT_S = 8
    FIRST_SOLUTION_STRATEGY = "PARALLEL_CHEAPEST_INSERTION"
    LOCAL_SEARCH_METAHEURISTIC = "GUIDED_LOCAL_SEARCH"
    USE_TIME_WINDOWS = True


@AlgorithmRegistry.register_scheduler("ortools_vrp_dep")
class ORToolsDependencyScheduler(ORToolsScheduler):
    ORTOOLS_TIME_LIMIT_S = 8
    FIRST_SOLUTION_STRATEGY = "PARALLEL_CHEAPEST_INSERTION"
    LOCAL_SEARCH_METAHEURISTIC = "GUIDED_LOCAL_SEARCH"
    USE_DEPENDENCY_PRECEDENCE = True


@AlgorithmRegistry.register_scheduler("ortools_vrp_sa")
class ORToolsSimulatedAnnealingScheduler(ORToolsScheduler):
    ORTOOLS_TIME_LIMIT_S = 8
    FIRST_SOLUTION_STRATEGY = "PARALLEL_CHEAPEST_INSERTION"
    LOCAL_SEARCH_METAHEURISTIC = "SIMULATED_ANNEALING"


@AlgorithmRegistry.register_scheduler("ortools_vrp_tabu")
class ORToolsTabuScheduler(ORToolsScheduler):
    ORTOOLS_TIME_LIMIT_S = 10
    FIRST_SOLUTION_STRATEGY = "PARALLEL_CHEAPEST_INSERTION"
    LOCAL_SEARCH_METAHEURISTIC = "TABU_SEARCH"


@AlgorithmRegistry.register_scheduler("ortools_vrp_tw_dep")
class ORToolsTimeWindowsDependencyScheduler(ORToolsScheduler):
    ORTOOLS_TIME_LIMIT_S = 10
    FIRST_SOLUTION_STRATEGY = "PARALLEL_CHEAPEST_INSERTION"
    LOCAL_SEARCH_METAHEURISTIC = "GUIDED_LOCAL_SEARCH"
    USE_TIME_WINDOWS = True
    USE_DEPENDENCY_PRECEDENCE = True


@AlgorithmRegistry.register_scheduler("ortools_vrp_prize")
class ORToolsPrizeCollectingScheduler(ORToolsScheduler):
    ORTOOLS_TIME_LIMIT_S = 8
    FIRST_SOLUTION_STRATEGY = "PARALLEL_CHEAPEST_INSERTION"
    LOCAL_SEARCH_METAHEURISTIC = "GUIDED_LOCAL_SEARCH"
    ALLOW_NODE_DROPPING = True
    DISJUNCTION_PENALTY_MODE = "prize"


@AlgorithmRegistry.register_scheduler("ortools_vrp_capacity")
class ORToolsCapacityScheduler(ORToolsScheduler):
    ORTOOLS_TIME_LIMIT_S = 8
    FIRST_SOLUTION_STRATEGY = "PARALLEL_CHEAPEST_INSERTION"
    LOCAL_SEARCH_METAHEURISTIC = "GUIDED_LOCAL_SEARCH"
    USE_CAPACITY_CONSTRAINTS = True
    ALLOW_NODE_DROPPING = True
