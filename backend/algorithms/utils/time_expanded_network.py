from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from backend.algorithms.utils.environment_graph import EnvironmentGraph, RouteGeometry
from backend.models.domain import Drone, GeoPoint, Task


def _distance(a: GeoPoint, b: GeoPoint) -> float:
    return float(np.linalg.norm(np.array(a.as_tuple()) - np.array(b.as_tuple())))


def _interpolate_polyline(points: Sequence[GeoPoint], fraction: float) -> GeoPoint:
    if not points:
        raise ValueError("polyline is empty")
    if len(points) == 1:
        return points[0]

    fraction = max(0.0, min(1.0, fraction))
    segment_lengths: list[float] = []
    total = 0.0
    for index in range(len(points) - 1):
        length = _distance(points[index], points[index + 1])
        segment_lengths.append(length)
        total += length

    if total <= 1e-9:
        return points[0]

    target = total * fraction
    cumulative = 0.0
    for index, length in enumerate(segment_lengths):
        next_cumulative = cumulative + length
        if target <= next_cumulative or index == len(segment_lengths) - 1:
            if length <= 1e-9:
                return points[index + 1]
            local = (target - cumulative) / length
            start = points[index]
            end = points[index + 1]
            return GeoPoint(
                x=start.x + local * (end.x - start.x),
                y=start.y + local * (end.y - start.y),
                z=start.z + local * (end.z - start.z),
            )
        cumulative = next_cumulative

    return points[-1]


@dataclass(frozen=True)
class TimelineSegment:
    start_time: float
    end_time: float
    points: tuple[GeoPoint, ...]
    action: str
    task_id: Optional[str]

    @property
    def is_airborne(self) -> bool:
        return self.action == "fly"


@dataclass
class RouteTimeline:
    valid: bool
    reason: str
    segments: list[TimelineSegment]
    completion_time: float
    total_distance: float

    def segment_at(self, time: float) -> Optional[TimelineSegment]:
        for segment in self.segments:
            if segment.start_time <= time <= segment.end_time + 1e-9:
                return segment
        return None

    def action_at(self, time: float) -> str:
        segment = self.segment_at(time)
        if segment is not None:
            return segment.action
        return "ground"

    def is_airborne_at(self, time: float) -> bool:
        return self.action_at(time) == "fly"

    def position_at(self, time: float, start_location: GeoPoint, end_location: GeoPoint) -> GeoPoint:
        if not self.segments:
            return start_location if time <= self.completion_time else end_location

        if time <= self.segments[0].start_time:
            return self.segments[0].points[0]
        if time >= self.segments[-1].end_time:
            return self.segments[-1].points[-1]

        for segment in self.segments:
            if segment.start_time <= time <= segment.end_time + 1e-9:
                if segment.end_time <= segment.start_time + 1e-9:
                    return segment.points[-1]
                if segment.action in {"wait", "service"} or len(segment.points) == 1:
                    return segment.points[-1]
                fraction = (time - segment.start_time) / (segment.end_time - segment.start_time)
                return _interpolate_polyline(segment.points, fraction)

        return self.segments[-1].points[-1]


class TimeExpandedNetwork:
    """Discrete time sampling layer for route feasibility and conflict analysis."""

    def __init__(
        self,
        time_step: float = 30.0,
        horizon_start: float = 0.0,
        horizon_end: float = 3600.0,
        safe_radius: float = 20.0,
        conflict_penalty: float = 5_000.0,
        max_layers: Optional[int] = None,
        prune_threshold: float = 0.0,
    ):
        self.time_step = max(1.0, float(time_step))
        self.horizon_start = float(horizon_start)
        self.safe_radius = max(1.0, float(safe_radius))
        self.conflict_penalty = max(1.0, float(conflict_penalty))
        self.max_layers = None if max_layers is None else max(1, int(max_layers))
        self.prune_threshold = max(0.0, float(prune_threshold))

        raw_layer_count = int(math.ceil(max(self.time_step, float(horizon_end) - self.horizon_start) / self.time_step))
        if self.max_layers is not None:
            raw_layer_count = min(raw_layer_count, self.max_layers)
        self.layer_count = max(1, raw_layer_count)
        self.horizon_end = self.horizon_start + self.layer_count * self.time_step
        self.layers: tuple[float, ...] = tuple(
            self.horizon_start + index * self.time_step for index in range(self.layer_count + 1)
        )

    @classmethod
    def from_scene(
        cls,
        drones: Sequence[Drone],
        tasks: Sequence[Task],
        time_step: float,
        horizon_slack: float,
        safe_radius: float,
        conflict_penalty: float,
        max_layers: Optional[int] = None,
        prune_threshold: float = 0.0,
    ) -> "TimeExpandedNetwork":
        start_time = min((float(drone.currentTime or 0.0) for drone in drones), default=0.0)
        latest_task_end = max((float(task.timeWindow[1]) for task in tasks), default=start_time)
        horizon_end = latest_task_end + max(0.0, float(horizon_slack))
        if horizon_end <= start_time:
            horizon_end = start_time + max(1.0, float(time_step))
        return cls(
            time_step=time_step,
            horizon_start=start_time,
            horizon_end=horizon_end,
            safe_radius=safe_radius,
            conflict_penalty=conflict_penalty,
            max_layers=max_layers,
            prune_threshold=prune_threshold,
        )

    def time_index(self, time: float, round_up: bool = False) -> int:
        offset = (float(time) - self.horizon_start) / self.time_step
        if round_up:
            return max(0, int(math.ceil(offset - 1e-9)))
        return max(0, int(math.floor(offset + 1e-9)))

    def slot_time(self, index: int) -> float:
        return self.horizon_start + index * self.time_step

    def task_slot_bounds(self, task: Task) -> tuple[int, int]:
        return self.time_index(task.timeWindow[0]), self.time_index(task.timeWindow[1], round_up=True)

    def build_timeline(
        self,
        drone: Drone,
        route: Sequence[Task],
        graph: EnvironmentGraph,
    ) -> RouteTimeline:
        segments: list[TimelineSegment] = []
        current_time = float(drone.currentTime or 0.0)
        current_loc = drone.currentLocation
        total_distance = 0.0
        depot_node = graph.nodes.get(f"depot::{drone.depotId}")
        depot_loc = depot_node.point if depot_node is not None else current_loc

        def add_segment(
            start_time: float,
            end_time: float,
            points: Sequence[GeoPoint],
            action: str,
            task_id: Optional[str],
        ) -> None:
            segments.append(
                TimelineSegment(
                    start_time=start_time,
                    end_time=end_time,
                    points=tuple(points),
                    action=action,
                    task_id=task_id,
                )
            )

        for task in route:
            geometry = graph.shortest_route(
                current_loc,
                task.location,
                current_time=current_time,
                speed=drone.speed,
            )
            if geometry is None or geometry.distance == float("inf"):
                return RouteTimeline(
                    valid=False,
                    reason=f"unable to route to task {task.id}",
                    segments=segments,
                    completion_time=current_time,
                    total_distance=total_distance,
                )

            travel_time = geometry.travel_time_at(float(drone.speed or 0.0), start_time=current_time)
            if not math.isfinite(travel_time):
                return RouteTimeline(
                    valid=False,
                    reason=f"invalid speed for task {task.id}",
                    segments=segments,
                    completion_time=current_time,
                    total_distance=total_distance,
                )

            add_segment(current_time, current_time + travel_time, geometry.points, "fly", task.id)
            current_time += travel_time
            total_distance += geometry.distance
            current_loc = task.location

            if current_time > float(task.timeWindow[1]) + 1e-6:
                return RouteTimeline(
                    valid=False,
                    reason=f"task {task.id} missed time window",
                    segments=segments,
                    completion_time=current_time,
                    total_distance=total_distance,
                )

            wait_time = max(0.0, float(task.timeWindow[0]) - current_time)
            if wait_time > 0:
                add_segment(current_time, current_time + wait_time, (current_loc, current_loc), "wait", task.id)
                current_time += wait_time

            if task.serviceDuration > 0:
                add_segment(
                    current_time,
                    current_time + float(task.serviceDuration),
                    (current_loc, current_loc),
                    "service",
                    task.id,
                )
                current_time += float(task.serviceDuration)

        if drone.returnToDepotRequired and current_loc.as_tuple() != depot_loc.as_tuple():
            geometry = graph.shortest_route(
                current_loc,
                depot_loc,
                current_time=current_time,
                speed=drone.speed,
            )
            if geometry is None or geometry.distance == float("inf"):
                return RouteTimeline(
                    valid=False,
                    reason=f"unable to return drone {drone.id} to depot",
                    segments=segments,
                    completion_time=current_time,
                    total_distance=total_distance,
                )

            travel_time = geometry.travel_time_at(float(drone.speed or 0.0), start_time=current_time)
            if not math.isfinite(travel_time):
                return RouteTimeline(
                    valid=False,
                    reason=f"invalid speed for drone {drone.id}",
                    segments=segments,
                    completion_time=current_time,
                    total_distance=total_distance,
                )

            add_segment(current_time, current_time + travel_time, geometry.points, "fly", None)
            current_time += travel_time
            total_distance += geometry.distance

        return RouteTimeline(
            valid=True,
            reason="",
            segments=segments,
            completion_time=current_time,
            total_distance=total_distance,
        )

    def evaluate_routes(
        self,
        routes: Dict[str, Sequence[Task]],
        drones: Sequence[Drone],
        graph: EnvironmentGraph,
    ) -> tuple[float, list[str], int]:
        drone_map = {drone.id: drone for drone in drones}
        timelines: Dict[str, RouteTimeline] = {}

        for drone_id, route in routes.items():
            drone = drone_map[drone_id]
            timelines[drone_id] = self.build_timeline(drone, route, graph)

        warnings: list[str] = []
        conflict_count = 0
        for first_id, second_id in combinations(timelines.keys(), 2):
            first = timelines[first_id]
            second = timelines[second_id]
            if not first.valid or not second.valid:
                continue

            first_drone = drone_map[first_id]
            second_drone = drone_map[second_id]
            start = min(
                first.segments[0].start_time if first.segments else float(first_drone.currentTime or 0.0),
                second.segments[0].start_time if second.segments else float(second_drone.currentTime or 0.0),
            )
            end = max(first.completion_time, second.completion_time)

            for time in self._time_layers(start, end):
                if not first.is_airborne_at(time) and not second.is_airborne_at(time):
                    continue

                pos_a = first.position_at(time, first_drone.currentLocation, first_drone.currentLocation)
                pos_b = second.position_at(time, second_drone.currentLocation, second_drone.currentLocation)
                distance = _distance(pos_a, pos_b)
                if distance < self.safe_radius * 2.0:
                    conflict_count += 1
                    warnings.append(
                        f"T={time:6.1f}s | {first_id} vs {second_id} | "
                        f"distance={distance:.2f}m (<{self.safe_radius * 2.0:.2f}m)"
                    )

        penalty = conflict_count * self.conflict_penalty
        for timeline in timelines.values():
            if not timeline.valid:
                penalty += self.conflict_penalty * 10.0

        return penalty, warnings, conflict_count

    def _time_layers(self, start: float, end: float) -> Iterable[float]:
        if end < start:
            return []

        first = self.time_index(start, round_up=False)
        last = self.time_index(end, round_up=True)
        last = min(last, self.layer_count)
        return [self.slot_time(index) for index in range(first, last + 1)]
