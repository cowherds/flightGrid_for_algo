from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

from backend.algorithms.base import BasePathPlanner
from backend.models.domain import GeoPoint
from backend.routing.cache import SimpleLRUCache
from backend.routing.types import EdgeEstimate, EdgeExactResult, EdgeQuery, RouteCertificate


@dataclass(frozen=True)
class _StaticVisibilityGraph:
    nodes: tuple[tuple[float, float], ...]
    graph: tuple[tuple[tuple[int, float], ...], ...]


class RouteOracle:
    """
    Routing abstraction for the next-generation scheduling stack.

    Current integration scope:
    - provide a stable location for edge estimate / exact solve APIs
    - centralize edge-level caching
    - enable later scheduler rewrites without binding directly to planner classes
    """

    def __init__(
        self,
        *,
        exact_planner: BasePathPlanner,
        estimate_planner: BasePathPlanner | None = None,
        max_cache_size: int = 20000,
        enable_geometry_estimate: bool = True,
    ):
        self.exact_planner = exact_planner
        self.estimate_planner = estimate_planner or exact_planner
        self._estimate_cache = SimpleLRUCache[tuple, EdgeEstimate](max_size=max_cache_size)
        self._exact_cache = SimpleLRUCache[tuple, EdgeExactResult](max_size=max_cache_size)
        self.enable_geometry_estimate = bool(enable_geometry_estimate)
        self._visibility_graph_cache: dict[float, _StaticVisibilityGraph] = {}

    @staticmethod
    def _cache_key(query: EdgeQuery, *, planner: BasePathPlanner) -> tuple:
        start_time = query.start_time
        speed = query.speed
        planner_env = getattr(planner, "env", None)
        if planner_env is not None and not getattr(planner_env, "has_time_dependent_constraints", False):
            start_time = None
            speed = None

        return (
            query.start.as_tuple(),
            query.goal.as_tuple(),
            None if start_time is None else round(float(start_time), 6),
            None if speed is None else round(float(speed), 6),
        )

    def estimate_edge(self, query: EdgeQuery) -> EdgeEstimate:
        planner = self.estimate_planner
        cached = self._estimate_cache.get(self._cache_key(query, planner=planner))
        if cached is not None:
            return cached

        geometry_result = self._estimate_edge_with_geometry(query)
        if geometry_result is not None:
            self._estimate_cache.set(self._cache_key(query, planner=planner), geometry_result)
            return geometry_result

        path, distance = planner.get_route(
            query.start,
            query.goal,
            current_time=query.start_time,
            speed=query.speed,
        )
        result = EdgeEstimate(
            feasible=distance != float("inf"),
            distance=float(distance),
            reason="" if distance != float("inf") else "edge_unreachable",
            metadata={"pathPointCount": len(path)},
        )
        self._estimate_cache.set(self._cache_key(query, planner=planner), result)
        return result

    def _estimate_edge_with_geometry(self, query: EdgeQuery) -> EdgeEstimate | None:
        if not self.enable_geometry_estimate:
            return None

        planner = self.estimate_planner
        env = getattr(planner, "env", None)
        if env is None:
            return None
        if getattr(env, "has_time_dependent_blocking_constraints", False):
            return None

        start = query.start
        goal = query.goal
        direct_distance = math.dist(start.as_tuple(), goal.as_tuple())
        if direct_distance <= 1e-9:
            return EdgeEstimate(
                feasible=True,
                distance=0.0,
                reason="",
                metadata={"mode": "geometry_zero"},
            )

        if env.line_of_sight(start, goal, start_time=query.start_time, end_time=query.start_time):
            return EdgeEstimate(
                feasible=True,
                distance=float(direct_distance),
                reason="",
                metadata={"mode": "geometry_direct"},
            )

        visibility_distance = self._estimate_static_visibility_distance(query)
        if visibility_distance is None:
            return None

        return EdgeEstimate(
            feasible=math.isfinite(visibility_distance),
            distance=float(visibility_distance),
            reason="" if math.isfinite(visibility_distance) else "edge_unreachable",
            metadata={"mode": "geometry_visibility"},
        )

    def _estimate_static_visibility_distance(self, query: EdgeQuery) -> float | None:
        planner = self.estimate_planner
        env = getattr(planner, "env", None)
        if env is None:
            return None

        cruise_z = float(max(query.start.z, query.goal.z))
        visibility = self._get_static_visibility_graph(cruise_z)
        if visibility is None:
            return None
        if not visibility.nodes:
            return math.dist(query.start.as_tuple(), query.goal.as_tuple())

        start_xy = (float(query.start.x), float(query.start.y))
        goal_xy = (float(query.goal.x), float(query.goal.y))
        start_index = len(visibility.nodes)
        goal_index = start_index + 1
        graph = [list(edges) for edges in visibility.graph]
        graph.append([])
        graph.append([])
        start_point = GeoPoint(start_xy[0], start_xy[1], cruise_z)
        goal_point = GeoPoint(goal_xy[0], goal_xy[1], cruise_z)

        if env.line_of_sight(start_point, goal_point, start_time=query.start_time, end_time=query.start_time):
            distance = math.dist(start_xy, goal_xy)
            graph[start_index].append((goal_index, distance))
            graph[goal_index].append((start_index, distance))

        for node_index, (node_x, node_y) in enumerate(visibility.nodes):
            node_point = GeoPoint(node_x, node_y, cruise_z)

            if env.line_of_sight(start_point, node_point, start_time=query.start_time, end_time=query.start_time):
                distance = math.dist(start_xy, (node_x, node_y))
                graph[start_index].append((node_index, distance))
                graph[node_index].append((start_index, distance))

            if env.line_of_sight(goal_point, node_point, start_time=query.start_time, end_time=query.start_time):
                distance = math.dist(goal_xy, (node_x, node_y))
                graph[goal_index].append((node_index, distance))
                graph[node_index].append((goal_index, distance))

        shortest_xy = self._dijkstra_shortest_path(graph, start_index=start_index, goal_index=goal_index)
        if shortest_xy is None:
            return None

        vertical_cost = abs(float(query.start.z) - cruise_z) + abs(float(query.goal.z) - cruise_z)
        return float(shortest_xy + vertical_cost)

    def _get_static_visibility_graph(self, cruise_z: float) -> _StaticVisibilityGraph | None:
        cache_key = round(float(cruise_z), 6)
        cached = self._visibility_graph_cache.get(cache_key)
        if cached is not None:
            return cached

        visibility = self._build_static_visibility_graph(cache_key)
        if visibility is not None:
            self._visibility_graph_cache[cache_key] = visibility
        return visibility

    def _build_static_visibility_graph(self, cruise_z: float) -> _StaticVisibilityGraph | None:
        planner = self.estimate_planner
        env = getattr(planner, "env", None)
        if env is None:
            return None

        try:
            zones = list(env._collect_2d_zones_for_altitude(cruise_z))
        except Exception:
            return None
        if not zones:
            return _StaticVisibilityGraph(nodes=(), graph=())

        nodes: list[tuple[float, float]] = []

        for zone in zones:
            raw_vertices = [
                (float(vertex["x"]), float(vertex["y"]))
                for vertex in (zone.get("vertices") or [])
            ]
            if len(raw_vertices) < 3:
                continue
            inflated_vertices = self._inflate_polygon_vertices(raw_vertices, epsilon=1.0)
            nodes.extend(inflated_vertices)

        if not nodes:
            return _StaticVisibilityGraph(nodes=(), graph=())

        graph: list[list[tuple[int, float]]] = [[] for _ in nodes]
        for left in range(len(nodes)):
            ax, ay = nodes[left]
            point_a = GeoPoint(ax, ay, cruise_z)
            for right in range(left + 1, len(nodes)):
                bx, by = nodes[right]
                point_b = GeoPoint(bx, by, cruise_z)
                if not env.line_of_sight(point_a, point_b, start_time=None, end_time=None):
                    continue
                distance = math.dist((ax, ay), (bx, by))
                graph[left].append((right, distance))
                graph[right].append((left, distance))

        return _StaticVisibilityGraph(
            nodes=tuple(nodes),
            graph=tuple(tuple(edges) for edges in graph),
        )

    @staticmethod
    def _inflate_polygon_vertices(
        vertices: Iterable[tuple[float, float]],
        *,
        epsilon: float,
    ) -> list[tuple[float, float]]:
        points = list(vertices)
        if len(points) < 3:
            return points

        center_x = sum(x for x, _y in points) / float(len(points))
        center_y = sum(y for _x, y in points) / float(len(points))
        inflated: list[tuple[float, float]] = []
        for x, y in points:
            dx = x - center_x
            dy = y - center_y
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                inflated.append((x, y))
                continue
            scale = (length + epsilon) / length
            inflated.append((center_x + dx * scale, center_y + dy * scale))
        return inflated

    @staticmethod
    def _dijkstra_shortest_path(
        graph: list[list[tuple[int, float]]],
        *,
        start_index: int,
        goal_index: int,
    ) -> float | None:
        heap: list[tuple[float, int]] = [(0.0, start_index)]
        best = {start_index: 0.0}

        while heap:
            distance, node = heapq.heappop(heap)
            if node == goal_index:
                return distance
            if distance > best.get(node, float("inf")):
                continue
            for neighbor, edge_cost in graph[node]:
                candidate = distance + edge_cost
                if candidate >= best.get(neighbor, float("inf")):
                    continue
                best[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))

        return None

    def solve_edge_exact(self, query: EdgeQuery) -> EdgeExactResult:
        planner = self.exact_planner
        cached = self._exact_cache.get(self._cache_key(query, planner=planner))
        if cached is not None:
            return cached

        path, distance = planner.get_route(
            query.start,
            query.goal,
            current_time=query.start_time,
            speed=query.speed,
        )
        result = EdgeExactResult(
            feasible=distance != float("inf"),
            distance=float(distance),
            path=list(path),
            reason="" if distance != float("inf") else "edge_unreachable",
            metadata={"pathPointCount": len(path)},
        )
        self._exact_cache.set(self._cache_key(query, planner=planner), result)
        return result

    def certify_exact_route(self, points: list[GeoPoint]) -> RouteCertificate:
        if len(points) < 2:
            return RouteCertificate(feasible=True, exact=True, total_distance=0.0, edge_count=0)

        total_distance = 0.0
        for index in range(len(points) - 1):
            result = self.solve_edge_exact(EdgeQuery(start=points[index], goal=points[index + 1]))
            if not result.feasible:
                return RouteCertificate(
                    feasible=False,
                    exact=True,
                    total_distance=total_distance,
                    edge_count=index,
                    reason=result.reason or "edge_unreachable",
                )
            total_distance += result.distance

        return RouteCertificate(
            feasible=True,
            exact=True,
            total_distance=total_distance,
            edge_count=max(0, len(points) - 1),
        )

    @staticmethod
    def estimate_euclidean_distance(start: GeoPoint, goal: GeoPoint) -> float:
        return math.dist(start.as_tuple(), goal.as_tuple())
