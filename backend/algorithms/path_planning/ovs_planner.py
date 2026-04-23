"""
OVS 路径规划器：在固定高度平面上做 2D OVS，再抬升到三维航迹。

与 CityEnvironment 对齐：障碍物取自主环境在给定高度上的二维投影（与 C++ 快路径一致）。

离线验证（与任意调度器组合）：在仓库根目录执行 ``python3 algo_run.py -s <schedulingId> -p ovs``，
或使用 ``--preset lmta+ovs``；场景 JSON 放在 ``backend/data/``（或 ``data/``）下。详见根目录 ``algo_run.py`` 文档。
"""

from __future__ import annotations

import math
from typing import Any, List, Sequence, Tuple

import numpy as np

from backend.algorithms.base import BasePathPlanner
from backend.algorithms.path_planning.ovs_core import OVSPlannerCore, build_obstacle_from_xy_polygon
from backend.algorithms.registry import AlgorithmRegistry
from backend.models.domain import GeoPoint


def _clamp_z(z: float, env) -> float:
    return float(max(0.0, min(float(z), float(getattr(env, "limit_z", z)))))


def _resolve_plan_z(start: GeoPoint, goal: GeoPoint, env, mode: str) -> float:
    mode = (mode or "avg").strip().lower()
    if mode == "start":
        return _clamp_z(float(start.z), env)
    if mode == "goal":
        return _clamp_z(float(goal.z), env)
    if mode == "max":
        return _clamp_z(max(float(start.z), float(goal.z)), env)
    if mode == "min":
        return _clamp_z(min(float(start.z), float(goal.z)), env)
    return _clamp_z(0.5 * (float(start.z) + float(goal.z)), env)


def _zones_at_altitude(env, altitude: float) -> list[dict]:
    if hasattr(env, "_collect_2d_zones_for_altitude"):
        return env._collect_2d_zones_for_altitude(float(altitude))
    return []


def _build_poly_obs(env, altitude: float) -> list:
    zones = _zones_at_altitude(env, altitude)
    obstacles = []
    for i, zone in enumerate(zones):
        verts = zone.get("vertices") or []
        if len(verts) < 3:
            continue
        xy = [(float(v["x"]), float(v["y"])) for v in verts]
        obstacles.append(build_obstacle_from_xy_polygon(i, xy))
    return obstacles


def _assemble_3d_path(
    start: GeoPoint,
    goal: GeoPoint,
    plan_z: float,
    ovs_xy: Sequence[Tuple[float, float]],
) -> List[GeoPoint]:
    if len(ovs_xy) < 2:
        return []

    zp = float(plan_z)
    out: List[GeoPoint] = [start]

    if abs(float(start.z) - zp) > 1e-3:
        nxt = GeoPoint(float(start.x), float(start.y), zp)
        if math.dist(out[-1].as_tuple(), nxt.as_tuple()) > 1e-4:
            out.append(nxt)

    for px, py in ovs_xy[1:-1]:
        p = GeoPoint(float(px), float(py), zp)
        if math.dist(out[-1].as_tuple(), p.as_tuple()) > 1e-4:
            out.append(p)

    if abs(float(goal.z) - zp) > 1e-3:
        nxt = GeoPoint(float(goal.x), float(goal.y), zp)
        if math.dist(out[-1].as_tuple(), nxt.as_tuple()) > 1e-4:
            out.append(nxt)

    if math.dist(out[-1].as_tuple(), goal.as_tuple()) > 1e-4:
        out.append(goal)

    return out


@AlgorithmRegistry.register_path_planner("ovs_2d_slice")
class OVSPathPlanner(BasePathPlanner):
    """
    固定 z 平面上的 OVS；航迹端点使用真实三维起终点，中间航点为 plan_z。

    scheduler_parameters / planner_parameters 可选：
    - ovs_plan_z_mode: avg | start | goal | max | min
    - ovs_agent_radius: 机体安全半径（米）
    - ovs_inflation: 多边形膨胀附加量（米）
    """

    def __init__(self, env, grid_resolution=None, **kwargs: Any):
        super().__init__(env)
        self._grid_resolution = float(grid_resolution or 10.0)
        self._kwargs = dict(kwargs)
        self.cache: dict = {}

    def ingest_planner_parameters(self, parameters: dict[str, Any] | None) -> None:
        if not parameters:
            return
        for key, value in parameters.items():
            if str(key).startswith("ovs_"):
                self._kwargs[str(key)] = value

    def _robot_radius(self) -> float:
        raw = self._kwargs.get("ovs_agent_radius")
        if raw is not None:
            try:
                return max(0.1, float(raw))
            except (TypeError, ValueError):
                pass
        return max(0.5, min(self._grid_resolution * 0.35, 12.0))

    def _inflation(self) -> float:
        raw = self._kwargs.get("ovs_inflation")
        if raw is not None:
            try:
                return max(1e-4, float(raw))
            except (TypeError, ValueError):
                pass
        r = self._robot_radius()
        return max(0.05, min(r / 5.0, 1.2))

    def _plan_z_mode(self) -> str:
        return str(self._kwargs.get("ovs_plan_z_mode") or "avg")

    def get_route(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        current_time=None,
        speed=None,
    ) -> Tuple[List[GeoPoint], float]:
        timer_token = self.start_route_timer()
        try:
            self.check_cancelled()
            self.record_route_request()
            key = self._build_cache_key(start, goal, current_time, speed)
            if key in self.cache:
                self.record_cache_hit()
                return self.cache[key]
            self.record_cache_miss()

            estimated_arrival = self._estimate_arrival_time(
                start,
                goal,
                current_time=current_time,
                speed=speed,
            )

            if self.env.is_collision(start, current_time=current_time) or self.env.is_collision(
                goal,
                current_time=estimated_arrival,
            ):
                self.cache[key] = ([], float("inf"))
                return self.cache[key]

            plan_z = _resolve_plan_z(start, goal, self.env, self._plan_z_mode())
            poly_obs = _build_poly_obs(self.env, plan_z)

            margin = 10.0
            limit_x = float(getattr(self.env, "limit_x", max(start.x, goal.x) + margin))
            limit_y = float(getattr(self.env, "limit_y", max(start.y, goal.y) + margin))
            x_range = (-margin, limit_x + margin)
            y_range = (-margin, limit_y + margin)

            s_xy = (float(start.x), float(start.y))
            g_xy = (float(goal.x), float(goal.y))

            core = OVSPlannerCore()
            core.init_env(poly_obs, x_range, y_range, self._robot_radius(), inflation=self._inflation())
            core.set_start_and_goal(s_xy, g_xy)
            ovs_path = core.global_search()

            if not ovs_path:
                core.set_start_and_goal(g_xy, s_xy)
                ovs_path = core.global_search()
                ovs_path.reverse()

            if not ovs_path:
                if self.env.line_of_sight(start, goal, start_time=current_time, end_time=estimated_arrival):
                    d = float(np.linalg.norm(np.array(start.as_tuple()) - np.array(goal.as_tuple())))
                    self.cache[key] = ([start, goal], d)
                    return self.cache[key]
                self.cache[key] = ([], float("inf"))
                return self.cache[key]

            path3d = _assemble_3d_path(start, goal, plan_z, ovs_path)
            if len(path3d) < 2:
                self.cache[key] = ([], float("inf"))
                return self.cache[key]

            result = self._finalize_route_result(
                path3d,
                current_time=current_time,
                speed=speed,
            )
            self.cache[key] = result
            return result
        finally:
            self.finish_route_timer(timer_token)
