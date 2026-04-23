from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.models.domain import GeoPoint


@dataclass(frozen=True)
class EdgeQuery:
    start: GeoPoint
    goal: GeoPoint
    start_time: float | None = None
    speed: float | None = None


@dataclass
class EdgeEstimate:
    feasible: bool
    distance: float
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeExactResult:
    feasible: bool
    distance: float
    path: list[GeoPoint] = field(default_factory=list)
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteCertificate:
    feasible: bool
    exact: bool
    total_distance: float
    edge_count: int = 0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
