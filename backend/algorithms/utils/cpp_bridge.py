from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable

import numpy as np

BLOCKED_DISTANCE = 999999.0

_LIB = None


def _candidate_library_paths() -> list[Path]:
    env_path = os.getenv("FLIGHTGRID_DISTANCE_LIB")
    paths: list[Path] = []
    if env_path:
        paths.append(Path(env_path))

    repo_root = Path(__file__).resolve().parents[3]
    paths.extend(
        [
            repo_root / "backend" / "cpp_engine" / "build" / "libdistance.so",
            repo_root / "backend" / "cpp_engine" / "libdistance.so",
            repo_root / "backend" / "cpp_engine" / "build" / "Release" / "libdistance.so",
        ]
    )
    return paths


def _load_library():
    global _LIB
    if _LIB is not None:
        return _LIB

    for path in _candidate_library_paths():
        if path.exists():
            lib = ctypes.CDLL(str(path))
            lib.calculate_distance_matrix.argtypes = [
                ctypes.POINTER(ctypes.c_double),  # points_xy
                ctypes.c_int,  # num_points
                ctypes.POINTER(ctypes.c_double),  # polygons_xy
                ctypes.POINTER(ctypes.c_int),  # polygon_vertex_counts
                ctypes.c_int,  # num_polygons
                ctypes.c_double,  # blocked_distance
                ctypes.POINTER(ctypes.c_double),  # out_matrix
            ]
            lib.calculate_distance_matrix.restype = None
            if hasattr(lib, "calculate_segment_blocked_flags"):
                lib.calculate_segment_blocked_flags.argtypes = [
                    ctypes.POINTER(ctypes.c_double),  # segments_xy [N,4]
                    ctypes.c_int,  # num_segments
                    ctypes.POINTER(ctypes.c_double),  # polygons_xy
                    ctypes.POINTER(ctypes.c_int),  # polygon_vertex_counts
                    ctypes.c_int,  # num_polygons
                    ctypes.POINTER(ctypes.c_ubyte),  # out_blocked
                ]
                lib.calculate_segment_blocked_flags.restype = None
            _LIB = lib
            return _LIB

    checked = ", ".join(str(p) for p in _candidate_library_paths())
    raise FileNotFoundError(
        "libdistance.so not found. Compile backend/cpp_engine first. "
        f"Checked paths: {checked}"
    )


def _to_points_array(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("points must be a 2D array-like with shape (N, >=2)")
    if arr.shape[0] == 0:
        raise ValueError("points must not be empty")
    return np.ascontiguousarray(arr[:, :2], dtype=np.float64)


def _extract_vertices(zone) -> list[tuple[float, float]]:
    if isinstance(zone, dict):
        vertices = zone.get("vertices")
        if isinstance(vertices, list) and vertices:
            out: list[tuple[float, float]] = []
            for v in vertices:
                if isinstance(v, dict):
                    if "x" in v and "y" in v:
                        out.append((float(v["x"]), float(v["y"])))
                    elif "lng" in v and "lat" in v:
                        out.append((float(v["lng"]), float(v["lat"])))
                elif isinstance(v, (list, tuple)) and len(v) >= 2:
                    out.append((float(v[0]), float(v[1])))
            return out
    return []


def _to_polygon_buffers(no_fly_zones: Iterable[object]) -> tuple[np.ndarray, np.ndarray]:
    flat_vertices: list[float] = []
    vertex_counts: list[int] = []

    for zone in no_fly_zones or []:
        vertices = _extract_vertices(zone)
        if len(vertices) < 3:
            continue
        vertex_counts.append(len(vertices))
        for x, y in vertices:
            flat_vertices.extend([x, y])

    if not vertex_counts:
        return (
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int32),
        )

    return (
        np.ascontiguousarray(flat_vertices, dtype=np.float64),
        np.ascontiguousarray(vertex_counts, dtype=np.int32),
    )


def get_matrix(points, no_fly_zones):
    """
    Calculate N x N distance matrix with no-fly-zone blocking.

    points:
      array-like of shape (N, >=2), using x/y in first two columns.
    no_fly_zones:
      list of dicts with "vertices", where each vertex has x/y or lng/lat.
    """

    lib = _load_library()

    points_xy = _to_points_array(points)
    num_points = int(points_xy.shape[0])
    out_matrix = np.empty((num_points, num_points), dtype=np.float64, order="C")

    polygons_xy, polygon_vertex_counts = _to_polygon_buffers(no_fly_zones)
    num_polygons = int(polygon_vertex_counts.size)

    points_ptr = points_xy.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    out_ptr = out_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    polygons_ptr = (
        polygons_xy.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if polygons_xy.size > 0
        else ctypes.POINTER(ctypes.c_double)()
    )
    counts_ptr = (
        polygon_vertex_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        if polygon_vertex_counts.size > 0
        else ctypes.POINTER(ctypes.c_int)()
    )

    lib.calculate_distance_matrix(
        points_ptr,
        num_points,
        polygons_ptr,
        counts_ptr,
        num_polygons,
        float(BLOCKED_DISTANCE),
        out_ptr,
    )

    return out_matrix.tolist()


def get_blocked_flags(segments, no_fly_zones):
    """
    Batch check whether each 2D segment intersects no-fly polygons.

    segments:
      array-like of shape (N, 4): [x1, y1, x2, y2].
    no_fly_zones:
      list of dicts with "vertices", where each vertex has x/y or lng/lat.
    Returns:
      List[bool] with length N.
    """
    lib = _load_library()
    if not hasattr(lib, "calculate_segment_blocked_flags"):
        raise NotImplementedError("calculate_segment_blocked_flags is not available in libdistance.so")

    seg_arr = np.asarray(segments, dtype=np.float64)
    if seg_arr.ndim != 2 or seg_arr.shape[1] != 4:
        raise ValueError("segments must be a 2D array-like with shape (N, 4)")
    if seg_arr.shape[0] == 0:
        return []
    seg_arr = np.ascontiguousarray(seg_arr, dtype=np.float64)

    polygons_xy, polygon_vertex_counts = _to_polygon_buffers(no_fly_zones)
    num_polygons = int(polygon_vertex_counts.size)
    out_blocked = np.zeros((seg_arr.shape[0],), dtype=np.uint8)

    segments_ptr = seg_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    out_ptr = out_blocked.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    polygons_ptr = (
        polygons_xy.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if polygons_xy.size > 0
        else ctypes.POINTER(ctypes.c_double)()
    )
    counts_ptr = (
        polygon_vertex_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        if polygon_vertex_counts.size > 0
        else ctypes.POINTER(ctypes.c_int)()
    )

    lib.calculate_segment_blocked_flags(
        segments_ptr,
        int(seg_arr.shape[0]),
        polygons_ptr,
        counts_ptr,
        num_polygons,
        out_ptr,
    )

    return [bool(v) for v in out_blocked.tolist()]
