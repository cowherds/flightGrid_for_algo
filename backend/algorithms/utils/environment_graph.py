from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from backend.algorithms.utils.environment import CityEnvironment
from backend.models.domain import GeoPoint, SpatialConstraint, Task


def _distance(a: GeoPoint, b: GeoPoint) -> float:
    return float(np.linalg.norm(np.array(a.as_tuple()) - np.array(b.as_tuple())))


def _point_key(point: GeoPoint, precision: int = 3) -> tuple[float, float, float]:
    return (round(point.x, precision), round(point.y, precision), round(point.z, precision))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    point: GeoPoint
    kind: str
    task_id: Optional[str] = None
    depot_id: Optional[str] = None
    constraint_id: Optional[str] = None


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    distance: float
    points: tuple[GeoPoint, ...]


@dataclass(frozen=True)
class RouteGeometry:
    node_ids: tuple[str, ...]
    points: tuple[GeoPoint, ...]
    distance: float
    env: Optional[CityEnvironment] = None

    def travel_time_at(self, speed: float, start_time: Optional[float] = None) -> float:
        if speed <= 0:
            return float("inf")
        if self.env is not None:
            return float(self.env.estimate_path_travel_time(self.points, speed, start_time=start_time))
        return self.distance / speed


class EnvironmentGraph:
    """Sparse spatial graph with cached shortest-path queries."""

    def __init__(
        self,
        env: CityEnvironment,
        nodes: Dict[str, GraphNode],
        max_neighbors: int = 12,
    ):
        self.env = env
        self.nodes = dict(nodes)
        self.max_neighbors = max(4, int(max_neighbors))
        self._point_index: Dict[tuple[float, float, float], str] = {
            _point_key(node.point): node_id for node_id, node in self.nodes.items()
        }
        self._neighbors: Dict[str, list[GraphEdge]] = {node_id: [] for node_id in self.nodes}
        self._route_cache: Dict[tuple, Optional[RouteGeometry]] = {}
        self._build_sparse_edges()

    def get_node(self, node_id: str) -> GraphNode:
        return self.nodes[node_id]

    def find_node_by_point(self, point: GeoPoint) -> Optional[str]:
        return self._point_index.get(_point_key(point))

    def route_distance(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        current_time: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> float:
        geometry = self.shortest_route(start, goal, current_time=current_time, speed=speed)
        return geometry.distance if geometry is not None else float("inf")

    def shortest_route(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        current_time: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> Optional[RouteGeometry]:
        cache_key = (
            _point_key(start),
            _point_key(goal),
            None if current_time is None else round(float(current_time), 2),
            None if speed in (None, 0) else round(float(speed), 3),
        )
        if cache_key in self._route_cache:
            return self._route_cache[cache_key]

        if _distance(start, goal) <= 1e-9:
            geometry = RouteGeometry(node_ids=("start", "goal"), points=(start, goal), distance=0.0, env=self.env)
            self._route_cache[cache_key] = geometry
            return geometry

        if self._segment_clear(start, goal, current_time=current_time, speed=speed, distance_offset=0.0):
            geometry = RouteGeometry(
                node_ids=("start", "goal"),
                points=(start, goal),
                distance=_distance(start, goal),
                env=self.env,
            )
            self._route_cache[cache_key] = geometry
            return geometry

        start_id = self.find_node_by_point(start) or "__start__"
        goal_id = self.find_node_by_point(goal) or "__goal__"
        node_points: Dict[str, GeoPoint] = {node_id: node.point for node_id, node in self.nodes.items()}
        node_points["__start__"] = start
        node_points["__goal__"] = goal
        node_points[start_id] = start
        node_points[goal_id] = goal

        open_heap: list[tuple[float, str]] = [(0.0, start_id)]
        came_from: Dict[str, str] = {}
        g_score: Dict[str, float] = {start_id: 0.0}
        visited: set[str] = set()

        while open_heap:
            self.check_cancelled()
            _, current_id = heapq.heappop(open_heap)
            if current_id in visited:
                continue
            visited.add(current_id)

            if current_id == goal_id:
                break

            current_point = node_points[current_id]
            current_distance = g_score[current_id]
            for neighbor_id, edge_distance in self._iter_neighbors(
                current_id,
                start,
                goal,
                current_time=current_time,
                speed=speed,
                distance_offset=current_distance,
            ):
                if neighbor_id in visited:
                    continue

                neighbor_point = node_points[neighbor_id]
                if not self._segment_clear(
                    current_point,
                    neighbor_point,
                    current_time=current_time,
                    speed=speed,
                    distance_offset=current_distance,
                ):
                    continue

                tentative = current_distance + edge_distance
                if tentative + 1e-9 >= g_score.get(neighbor_id, float("inf")):
                    continue

                came_from[neighbor_id] = current_id
                g_score[neighbor_id] = tentative
                heapq.heappush(open_heap, (tentative, neighbor_id))

        if goal_id not in came_from and goal_id != start_id:
            self._route_cache[cache_key] = None
            return None

        node_sequence = [goal_id]
        while node_sequence[-1] != start_id:
            parent = came_from.get(node_sequence[-1])
            if parent is None:
                self._route_cache[cache_key] = None
                return None
            node_sequence.append(parent)
        node_sequence.reverse()

        points = [node_points[node_id] for node_id in node_sequence]
        if points[0].as_tuple() != start.as_tuple():
            points.insert(0, start)
        if points[-1].as_tuple() != goal.as_tuple():
            points.append(goal)

        distance = 0.0
        for index in range(len(points) - 1):
            distance += _distance(points[index], points[index + 1])

        geometry = RouteGeometry(node_ids=tuple(node_sequence), points=tuple(points), distance=distance, env=self.env)
        self._route_cache[cache_key] = geometry
        return geometry

    def check_cancelled(self) -> None:
        cancel_check = getattr(self.env, "cancel_check", None)
        if callable(cancel_check) and cancel_check():
            raise RuntimeError("DISPATCH_CANCELLED")

    def _build_sparse_edges(self) -> None:
        node_ids = list(self.nodes.keys())
        if len(node_ids) < 2:
            return

        for node_id in node_ids:
            node = self.nodes[node_id]
            candidates: list[tuple[float, str]] = []
            for other_id in node_ids:
                if other_id == node_id:
                    continue
                other = self.nodes[other_id]
                distance = _distance(node.point, other.point)
                if distance <= 1e-9:
                    continue
                candidates.append((distance, other_id))

            candidates.sort(key=lambda item: item[0])
            for distance, other_id in candidates[: self.max_neighbors]:
                other = self.nodes[other_id]
                if not self._segment_clear(node.point, other.point):
                    continue
                self._add_edge(node_id, other_id, distance)
                self._add_edge(other_id, node_id, distance)

    def _add_edge(self, source: str, target: str, distance: float) -> None:
        self._neighbors.setdefault(source, []).append(
            GraphEdge(
                source=source,
                target=target,
                distance=distance,
                points=(self.nodes[source].point, self.nodes[target].point),
            )
        )

    def _iter_neighbors(
        self,
        current_id: str,
        start: GeoPoint,
        goal: GeoPoint,
        current_time: Optional[float] = None,
        speed: Optional[float] = None,
        distance_offset: float = 0.0,
    ) -> Iterable[tuple[str, float]]:
        if current_id == "__start__":
            candidates: list[tuple[float, str]] = []
            for neighbor_id, node in self.nodes.items():
                if self._segment_clear(
                    start,
                    node.point,
                    current_time=current_time,
                    speed=speed,
                    distance_offset=distance_offset,
                ):
                    candidates.append((_distance(start, node.point), neighbor_id))

            candidates.sort(key=lambda item: item[0])
            for distance, neighbor_id in candidates[: self.max_neighbors]:
                yield neighbor_id, distance

            if self._segment_clear(
                start,
                goal,
                current_time=current_time,
                speed=speed,
                distance_offset=distance_offset,
            ):
                yield "__goal__", _distance(start, goal)
            return

        if current_id in self._neighbors:
            for edge in self._neighbors[current_id]:
                yield edge.target, edge.distance

        current_point = self._point_for_node(current_id, start, goal)
        if current_id != "__goal__" and self._segment_clear(
            current_point,
            goal,
            current_time=current_time,
            speed=speed,
            distance_offset=distance_offset,
        ):
            yield "__goal__", _distance(current_point, goal)

    def _point_for_node(self, node_id: str, start: GeoPoint, goal: GeoPoint) -> GeoPoint:
        if node_id == "__start__":
            return start
        if node_id == "__goal__":
            return goal
        return self.nodes[node_id].point

    def _segment_clear(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        current_time: Optional[float] = None,
        speed: Optional[float] = None,
        distance_offset: float = 0.0,
    ) -> bool:
        if current_time is None or not speed or speed <= 0:
            return self.env.line_of_sight(start, goal)

        segment_start_time = current_time + distance_offset / speed
        segment_end_time = self.env.estimate_arrival_time(start, goal, segment_start_time, speed)
        return self.env.line_of_sight(
            start,
            goal,
            start_time=segment_start_time,
            end_time=segment_end_time,
        )


class EnvironmentGraphBuilder:
    """Build a sparse graph from the environment, depots and tasks."""

    def __init__(
        self,
        env: CityEnvironment,
        tasks: Sequence[Task],
        depots: Dict[str, GeoPoint],
        graph_clearance: float = 20.0,
        max_neighbors: int = 12,
    ):
        self.env = env
        self.tasks = list(tasks)
        self.depots = dict(depots)
        self.graph_clearance = max(1.0, float(graph_clearance))
        self.max_neighbors = max(4, int(max_neighbors))

    def build(self) -> EnvironmentGraph:
        nodes: Dict[str, GraphNode] = {}
        used_points: set[tuple[float, float, float]] = set()

        for depot_id, point in self.depots.items():
            node = GraphNode(node_id=f"depot::{depot_id}", point=point, kind="depot", depot_id=depot_id)
            nodes[node.node_id] = node
            used_points.add(_point_key(point))

        for task in self.tasks:
            node = GraphNode(node_id=f"task::{task.id}", point=task.location, kind="task", task_id=task.id)
            nodes[node.node_id] = node
            used_points.add(_point_key(task.location))

        anchor_index = 0
        for constraint in self.env.constraints:
            for anchor_point in self._generate_constraint_anchors(constraint):
                key = _point_key(anchor_point)
                if key in used_points:
                    continue
                if self.env.is_collision(anchor_point):
                    continue
                node_id = f"anchor::{constraint.id}::{anchor_index:03d}"
                anchor_index += 1
                nodes[node_id] = GraphNode(
                    node_id=node_id,
                    point=anchor_point,
                    kind="anchor",
                    constraint_id=constraint.id,
                )
                used_points.add(key)

        return EnvironmentGraph(self.env, nodes, max_neighbors=self.max_neighbors)

    def _generate_constraint_anchors(self, constraint: SpatialConstraint) -> List[GeoPoint]:
        min_z, max_z = constraint._vertical_limits()
        altitude_layers = self._candidate_altitudes(min_z, max_z)
        footprint = self._footprint_points(constraint)
        if not footprint:
            return []

        centroid_x = sum(point[0] for point in footprint) / len(footprint)
        centroid_y = sum(point[1] for point in footprint) / len(footprint)
        anchors: list[GeoPoint] = []

        for base_x, base_y in footprint:
            vector_x = base_x - centroid_x
            vector_y = base_y - centroid_y
            norm = math.hypot(vector_x, vector_y)
            if norm <= 1e-9:
                vector_x, vector_y = 1.0, 0.0
                norm = 1.0
            expanded_x = base_x + self.graph_clearance * vector_x / norm
            expanded_y = base_y + self.graph_clearance * vector_y / norm
            for altitude in altitude_layers:
                anchors.append(GeoPoint(expanded_x, expanded_y, altitude))

        return anchors

    def _candidate_altitudes(self, min_z: float, max_z: float) -> List[float]:
        altitudes = {
            _clamp(min_z - self.graph_clearance, 0.0, float(self.env.limit_z)),
            _clamp(max_z + self.graph_clearance, 0.0, float(self.env.limit_z)),
            _clamp((min_z + max_z) / 2.0, 0.0, float(self.env.limit_z)),
        }

        for task in self.tasks:
            altitudes.add(_clamp(float(task.location.z), 0.0, float(self.env.limit_z)))
        for depot in self.depots.values():
            altitudes.add(_clamp(float(depot.z), 0.0, float(self.env.limit_z)))

        filtered = sorted({round(value, 3) for value in altitudes if 0.0 <= value <= float(self.env.limit_z)})
        return filtered or [0.0]

    def _footprint_points(self, constraint: SpatialConstraint) -> List[tuple[float, float]]:
        if constraint.shape == "box" and constraint.box:
            min_x, min_y, _min_z, max_x, max_y, _max_z = constraint.box
            return [
                (min_x - self.graph_clearance, min_y - self.graph_clearance),
                (min_x - self.graph_clearance, max_y + self.graph_clearance),
                (max_x + self.graph_clearance, min_y - self.graph_clearance),
                (max_x + self.graph_clearance, max_y + self.graph_clearance),
            ]

        if constraint.shape == "polygon" and constraint.polygon:
            xs = [point[0] for point in constraint.polygon]
            ys = [point[1] for point in constraint.polygon]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            points = [
                (min_x - self.graph_clearance, min_y - self.graph_clearance),
                (min_x - self.graph_clearance, max_y + self.graph_clearance),
                (max_x + self.graph_clearance, min_y - self.graph_clearance),
                (max_x + self.graph_clearance, max_y + self.graph_clearance),
            ]
            points.extend(constraint.polygon)
            return points

        if constraint.shape == "cylinder" and constraint.cylinder:
            center_x, center_y, radius, _min_z, _max_z = constraint.cylinder
            ring_radius = radius + self.graph_clearance
            return [
                (
                    center_x + ring_radius * math.cos(2.0 * math.pi * index / 8.0),
                    center_y + ring_radius * math.sin(2.0 * math.pi * index / 8.0),
                )
                for index in range(8)
            ]

        if constraint.box:
            min_x, min_y, _min_z, max_x, max_y, _max_z = constraint.box
            return [
                (min_x - self.graph_clearance, min_y - self.graph_clearance),
                (min_x - self.graph_clearance, max_y + self.graph_clearance),
                (max_x + self.graph_clearance, min_y - self.graph_clearance),
                (max_x + self.graph_clearance, max_y + self.graph_clearance),
            ]

        return []
