"""
OVS (guidance-point) 2D path planner core, adapted for FlightGrid obstacle polygons.

Reference flow: inflate obstacles → build visible graph vertices → greedy global
search with guidance points around blocking polygons → optional path smoothing.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from backend.algorithms.path_planning.ovs_geometry import (
    ensure_ccw,
    intersect_polygon_edges,
    is_intersect_polys,
    left_of,
    l2norm,
    normalize,
    point_line_dist,
    pos_in_polygons,
    smooth_path,
    unit_normal_vector,
    vertex_convexity_ccw,
)


@dataclass
class Vertex:
    point_: np.ndarray
    previous_: Optional["Vertex"] = None
    next_: Optional["Vertex"] = None
    convex_: bool = True
    is_disrupted: bool = False


@dataclass
class Obstacle2D:
    """Polygon obstacle with doubly-linked vertices (CCW outer boundary)."""

    idx: int
    tid: str
    vertices_: List[Vertex] = field(default_factory=list)
    vertices_pos: List[np.ndarray] = field(default_factory=list)


class OVSPlannerCore:
    def __init__(self) -> None:
        self.s_start: Optional[Tuple[float, float]] = None
        self.s_goal: Optional[Tuple[float, float]] = None
        self.x_range: Optional[Tuple[float, float]] = None
        self.y_range: Optional[Tuple[float, float]] = None
        self.radius: float = -1.0
        self.epsilon: float = 1e-5
        self.inflation: float = 0.01
        self.path: List[Tuple[float, float]] = []
        self.path_set: Set[Tuple[float, float]] = set()
        self.length: float = 0.0
        self.poly_obs: List[Obstacle2D] = []
        self.inflated_obs: List[Obstacle2D] = []
        self.visible_graph: List[Obstacle2D] = []
        self.check_time: float = 0.0

    def set_start_and_goal(self, s_start: Tuple[float, float], s_goal: Tuple[float, float]) -> None:
        self.s_start = s_start
        self.s_goal = s_goal
        self.path = []
        self.path_set = set()
        self.length = 0.0
        self.check_time = 0.0

    def init_env(
        self,
        obstacles: Sequence[Obstacle2D],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        radius: float,
        inflation: float = 0.01,
    ) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.poly_obs = list(obstacles)
        self.inflation = inflation
        self.radius = radius
        self.inflated_obs = []
        self.visible_graph = []

        agent_rad = self.radius + self.inflation
        combined_radius = agent_rad + self.inflation
        for poly in self.poly_obs:
            self._scale_polygon_vertices(poly, combined_radius)

        for vis in self.visible_graph:
            for v in vis.vertices_:
                px, py = float(v.point_[0]), float(v.point_[1])
                if self.x_range[0] < px < self.x_range[1] and self.y_range[0] < py < self.y_range[1]:
                    continue
                v.is_disrupted = True

    def _scale_polygon_vertices(self, polygon: Obstacle2D, combined_radius: float) -> None:
        vertices_pos: List[np.ndarray] = []
        graph_pos: List[np.ndarray] = []
        verts = polygon.vertices_

        for vertice in verts:
            p_prev = np.asarray(vertice.previous_.point_, dtype=float)
            p_cur = np.asarray(vertice.point_, dtype=float)
            p_next = np.asarray(vertice.next_.point_, dtype=float)
            p1 = normalize(p_prev - p_cur)
            p2 = normalize(p_next - p_cur)
            p1p2 = p2 - p1
            _n_left, n_right = unit_normal_vector(p1p2)
            denom = float(np.clip(np.dot(p1, p2), -1.0, 1.0))
            ang = math.acos(denom)
            sin_half = math.sin(ang / 2.0)
            if abs(sin_half) < 1e-6:
                n_scale = combined_radius * 10.0
            else:
                n_scale = combined_radius / sin_half
            vertices_pos.append(p_cur + n_scale * n_right)
            graph_pos.append(p_cur + (n_scale + self.inflation) * n_right)

        inf_vertices: List[Vertex] = []
        n = len(vertices_pos)
        for i in range(n):
            v = Vertex(point_=vertices_pos[i])
            inf_vertices.append(v)
        for i in range(n):
            inf_vertices[i].previous_ = inf_vertices[i - 1]
            inf_vertices[i].next_ = inf_vertices[(i + 1) % n]

        inflated = Obstacle2D(idx=polygon.idx, tid=f"poly{polygon.idx}", vertices_=inf_vertices, vertices_pos=list(vertices_pos))
        self.inflated_obs.append(inflated)

        vis_vertices: List[Vertex] = []
        for i in range(n):
            v = Vertex(point_=graph_pos[i])
            vis_vertices.append(v)
        for i in range(n):
            vis_vertices[i].previous_ = vis_vertices[i - 1]
            vis_vertices[i].next_ = vis_vertices[(i + 1) % n]

        visible = Obstacle2D(idx=polygon.idx, tid=f"vis{polygon.idx}", vertices_=vis_vertices, vertices_pos=list(graph_pos))
        self.visible_graph.append(visible)

    def global_search(self) -> List[Tuple[float, float]]:
        assert self.s_start is not None and self.s_goal is not None
        pos = np.array(self.s_start, dtype=float)
        pos_next = np.array(self.s_goal, dtype=float)
        if pos_in_polygons(pos, self.inflated_obs) or pos_in_polygons(pos_next, self.inflated_obs):
            return []

        intersected_obs = self._check_all_intersect_obstacles(pos, pos_next, self.inflated_obs)
        path: List[Tuple[float, float]] = [self.s_start]
        self.path_set = {self.s_start}

        while True:
            path_line, is_again_check = self._global_ovs_strategy(pos, pos_next, intersected_obs, path)
            if path_line is None:
                return []

            if not is_again_check or not is_intersect_polys(path_line[0], path_line[1], self.inflated_obs):
                path_line.pop(0)
                path += path_line
                self.path_set.add(path_line[-1])
                if math.hypot(path[-1][0] - self.s_goal[0], path[-1][1] - self.s_goal[1]) < 1e-3:
                    self.path, self.length = smooth_path(path, self.inflated_obs)
                    return self.path
                pos = np.array(path[-1], dtype=float)
                pos_next = np.array(self.s_goal, dtype=float)
            else:
                pos_next = np.array(path_line[-1], dtype=float)
            intersected_obs = self._check_all_intersect_obstacles(pos, pos_next, self.inflated_obs)

    def _check_all_intersect_obstacles(self, pos: np.ndarray, pos_next: np.ndarray, sorted_obs: Sequence[Obstacle2D]) -> List[Obstacle2D]:
        t1 = time.time()
        all_intersected: List[Obstacle2D] = []
        for obstacle in sorted_obs:
            poly = [v.point_ for v in obstacle.vertices_]
            if intersect_polygon_edges(pos, pos_next, poly):
                all_intersected.append(obstacle)
        self.check_time += time.time() - t1
        return all_intersected

    def _cal_candidate_points(
        self,
        pos: np.ndarray,
        pos_next: np.ndarray,
        intersect_ob: Obstacle2D,
        path: List[Tuple[float, float]],
    ):
        min_acceptable_dist = 0.5 * self.inflation
        vertices = self.visible_graph[intersect_ob.idx].vertices_
        left_cands: List[Tuple[np.ndarray, float]] = []
        left_cands_b: List[Tuple[np.ndarray, float]] = []
        right_cands: List[Tuple[np.ndarray, float]] = []
        right_cands_b: List[Tuple[np.ndarray, float]] = []
        max_dist_left = 0.0
        max_dist_right = 0.0

        for vertice in vertices:
            if not vertice.convex_ or vertice.is_disrupted:
                continue
            opt_p = vertice.point_
            poly_edges = [v.point_ for v in intersect_ob.vertices_]
            condition1 = not intersect_polygon_edges(pos, opt_p, poly_edges)
            condition2 = l2norm(opt_p, pos) > min_acceptable_dist
            condition3 = (float(opt_p[0]), float(opt_p[1])) not in self.path_set
            condition4 = float(np.dot(opt_p - pos, pos_next - opt_p)) > 0
            v_in_left_side = left_of(pos, pos_next, opt_p)
            dist = point_line_dist(pos, pos_next, opt_p)
            if v_in_left_side > 0:
                if dist > max_dist_left:
                    max_dist_left = dist
                if condition1 and condition2 and condition3:
                    if condition4:
                        left_cands.append((opt_p, dist))
                    else:
                        left_cands_b.append((opt_p, dist))
            elif v_in_left_side < 0:
                if dist > max_dist_right:
                    max_dist_right = dist
                if condition1 and condition2 and condition3:
                    if condition4:
                        right_cands.append((opt_p, dist))
                    else:
                        right_cands_b.append((opt_p, dist))

        if not left_cands and not right_cands:
            return left_cands_b, right_cands_b, max_dist_left, max_dist_right
        if left_cands and not right_cands:
            return left_cands, right_cands_b, max_dist_left, max_dist_right
        if not left_cands and right_cands:
            return left_cands_b, right_cands, max_dist_left, max_dist_right
        return left_cands, right_cands, max_dist_left, max_dist_right

    def _cal_guidance_point(
        self,
        pos: np.ndarray,
        pos_next: np.ndarray,
        intersect_ob: Obstacle2D,
        path: List[Tuple[float, float]],
    ):
        left_cands, right_cands, max_d_left, max_d_right = self._cal_candidate_points(pos, pos_next, intersect_ob, path)
        guide_point: List[Tuple[np.ndarray, float]] = []
        if left_cands or right_cands:
            p_left = max(left_cands, key=lambda px: px[1]) if left_cands else []
            p_right = max(right_cands, key=lambda px: px[1]) if right_cands else []
            if p_left and p_right:
                opt_ps = [p for p in (p_left, p_right) if not is_intersect_polys(
                    (float(p[0][0]), float(p[0][1])),
                    (float(pos_next[0]), float(pos_next[1])),
                    [intersect_ob],
                )]
                if opt_ps:
                    guide_point = list(min(opt_ps, key=lambda px: px[1]))
                else:
                    p_left = [p_left[0], max_d_left]
                    p_right = [p_right[0], max_d_right]
                    guide_point = list(min([p_left, p_right], key=lambda px: px[1]))
            elif p_left:
                guide_point = list(p_left)
            else:
                guide_point = list(p_right)
        return guide_point

    def _global_ovs_strategy(
        self,
        pos: np.ndarray,
        pos_next: np.ndarray,
        intersected_obs: List[Obstacle2D],
        path: List[Tuple[float, float]],
    ):
        if intersected_obs:
            vertex_points: List[Tuple[np.ndarray, float]] = []
            vertex_points_b: List[Tuple[np.ndarray, float]] = []
            for intersect_ob in intersected_obs:
                vertex_point = self._cal_guidance_point(pos, pos_next, intersect_ob, path)
                if len(vertex_point) > 0:
                    gp = vertex_point[0]
                    if not is_intersect_polys((float(pos[0]), float(pos[1])), (float(gp[0]), float(gp[1])), self.inflated_obs):
                        vertex_points.append((gp, vertex_point[1]))
                    else:
                        vertex_points_b.append((gp, vertex_point[1]))
            if vertex_points:
                gp = max(vertex_points, key=lambda px: px[1])[0]
                return [(float(pos[0]), float(pos[1])), (float(gp[0]), float(gp[1]))], False
            if vertex_points_b:
                gp = max(vertex_points_b, key=lambda px: px[1])[0]
                return [(float(pos[0]), float(pos[1])), (float(gp[0]), float(gp[1]))], True
            return None, None
        return [(float(pos[0]), float(pos[1])), (float(pos_next[0]), float(pos_next[1]))], False


def build_obstacle_from_xy_polygon(idx: int, xy: Sequence[Tuple[float, float]]) -> Obstacle2D:
    pts = [np.array([float(x), float(y)], dtype=float) for x, y in xy]
    pts = ensure_ccw(pts)
    vertices: List[Vertex] = []
    n = len(pts)
    for i in range(n):
        v = Vertex(point_=pts[i])
        vertices.append(v)
    for i in range(n):
        vertices[i].previous_ = vertices[i - 1]
        vertices[i].next_ = vertices[(i + 1) % n]
        prev_p = np.asarray(vertices[i].previous_.point_, dtype=float)
        cur_p = np.asarray(vertices[i].point_, dtype=float)
        next_p = np.asarray(vertices[i].next_.point_, dtype=float)
        vertices[i].convex_ = vertex_convexity_ccw(prev_p, cur_p, next_p)
    return Obstacle2D(idx=idx, tid=f"poly{idx}", vertices_=vertices, vertices_pos=list(pts))
