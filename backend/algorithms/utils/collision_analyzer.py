"""碰撞分析模块。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
from backend.models.domain import PathPoint


def _to_np(point: PathPoint) -> np.ndarray:
    return np.array(point.location.as_tuple(), dtype=float)


def _severity(distance: float, safe_radius: float) -> str:
    if distance < safe_radius:
        return "critical"
    if distance < safe_radius * 1.5:
        return "warning"
    return "info"


def _interpolate_position(pts: List[PathPoint], t: float) -> Optional[np.ndarray]:
    if not pts or t < pts[0].time or t > pts[-1].time:
        return None
    for i in range(len(pts) - 1):
        if pts[i].time <= t <= pts[i + 1].time:
            p1, p2 = pts[i], pts[i + 1]
            if p2.time == p1.time:
                return _to_np(p1)
            ratio = (t - p1.time) / (p2.time - p1.time)
            return _to_np(p1) + ratio * (_to_np(p2) - _to_np(p1))
    return _to_np(pts[-1])


def analyze_safety_collisions(
    results: Dict[str, List[PathPoint]],
    safe_radius: float = 20.0,
    mode: str = "spatiotemporal",
) -> Tuple[List[str], List[dict]]:
    warnings: List[str] = []
    collisions: List[dict] = []
    
    if len(results) < 2:
        return warnings, collisions
        
    safe_distance = safe_radius * 2
    uav_ids = sorted(results.keys())

    if mode == "spatial":
        for i in range(len(uav_ids)):
            for j in range(i + 1, len(uav_ids)):
                uav_a, uav_b = uav_ids[i], uav_ids[j]
                path_a = results.get(uav_a, [])
                path_b = results.get(uav_b, [])
                if not path_a or not path_b:
                    continue
                
                min_distance = float("inf")
                best_a: Optional[np.ndarray] = None
                best_b: Optional[np.ndarray] = None
                
                for point_a in path_a:
                    np_a = _to_np(point_a)
                    for point_b in path_b:
                        np_b = _to_np(point_b)
                        distance = float(np.linalg.norm(np_a - np_b))
                        if distance < min_distance:
                            min_distance = distance
                            best_a = np_a
                            best_b = np_b
                            
                if min_distance < safe_distance and best_a is not None and best_b is not None:
                    mid = (best_a + best_b) / 2.0
                    warn_msg = f"{uav_a} 与 {uav_b} 存在空间冲突！最小路径间距 {min_distance:.2f}m (<{safe_distance:.2f}m)"
                    warnings.append(warn_msg)
                    collisions.append({
                        "id": f"spatial_{uav_a}_{uav_b}",
                        "type": "spatial",
                        "droneA": uav_a,
                        "droneB": uav_b,
                        "drones": [uav_a, uav_b],
                        "minDistance": round(min_distance, 3),
                        "time": None,
                        "timeRange": None,
                        "location": {"x": round(float(mid[0]), 3), "y": round(float(mid[1]), 3), "z": round(float(mid[2]), 3)},
                        "severity": _severity(min_distance, safe_radius),
                        "description": warn_msg,
                    })
        return warnings, collisions

    max_time = 0.0
    for pts in results.values():
        if pts:
            max_time = max(max_time, float(pts[-1].time))
            
    collision_events = set()
    for t in np.arange(0.0, max_time + 1.0, 1.0):
        current_positions: Dict[str, np.ndarray] = {}
        for uav_id, pts in results.items():
            pos = _interpolate_position(pts, float(t))
            if pos is not None:
                current_positions[uav_id] = pos
                
        current_uavs = sorted(current_positions.keys())
        for i in range(len(current_uavs)):
            for j in range(i + 1, len(current_uavs)):
                uav_a, uav_b = current_uavs[i], current_uavs[j]
                pos_a, pos_b = current_positions[uav_a], current_positions[uav_b]
                distance = float(np.linalg.norm(pos_a - pos_b))
                
                if distance < safe_distance:
                    event_key = (uav_a, uav_b, round(float(t) / 5.0))
                    if event_key in collision_events:
                        continue
                        
                    mid = (pos_a + pos_b) / 2.0
                    warn_msg = f"T={t:5.1f}s | {uav_a} 与 {uav_b} 安全区域重叠！中心距离 {distance:.2f}m (<{safe_distance:.2f}m)"
                    warnings.append(warn_msg)
                    collisions.append({
                        "id": f"spatiotemporal_{uav_a}_{uav_b}_{int(t)}",
                        "type": "spatiotemporal",
                        "droneA": uav_a,
                        "droneB": uav_b,
                        "drones": [uav_a, uav_b],
                        "minDistance": round(distance, 3),
                        "time": round(float(t), 3),
                        "timeRange": [round(float(t), 3), round(float(t), 3)],
                        "location": {"x": round(float(mid[0]), 3), "y": round(float(mid[1]), 3), "z": round(float(mid[2]), 3)},
                        "severity": _severity(distance, safe_radius),
                        "description": warn_msg,
                    })
                    collision_events.add(event_key)
                    
    return warnings, collisions