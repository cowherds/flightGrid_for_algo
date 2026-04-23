import os


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


class Settings:
    PROJECT_NAME = "UAV 3D Routing API"
    VERSION = "2.0.0"

    # 路径规划默认参数
    DEFAULT_GRID_RESOLUTION = 20  # A* 搜索的三维网格分辨率 (米)
    DEFAULT_SPACE_LIMITS = (500, 500, 150)  # X, Y, Z 最大边界

    # 调度任务队列参数（P2-1: 后台任务化）
    DISPATCH_QUEUE_MAX_WORKERS = _int_env("FLIGHTGRID_DISPATCH_QUEUE_MAX_WORKERS", 2, 1, 16)
    DISPATCH_JOB_RETENTION_SECONDS = _int_env("FLIGHTGRID_DISPATCH_JOB_RETENTION_SECONDS", 900, 60, 86400)
    DISPATCH_JOB_MAX_RECORDS = _int_env("FLIGHTGRID_DISPATCH_JOB_MAX_RECORDS", 400, 50, 5000)
    DISPATCH_EXECUTION_MODE = (os.getenv("FLIGHTGRID_DISPATCH_EXECUTION_MODE", "fast") or "fast").strip().lower()
    DISPATCH_ENABLE_FALLBACK = _bool_env(
        "FLIGHTGRID_DISPATCH_ENABLE_FALLBACK",
        DISPATCH_EXECUTION_MODE != "fast",
    )
    DISPATCH_ENABLE_STRICT_FINAL_VALIDATION = _bool_env(
        "FLIGHTGRID_DISPATCH_ENABLE_STRICT_FINAL_VALIDATION",
        DISPATCH_EXECUTION_MODE != "fast",
    )
    DISPATCH_ENABLE_COLLISION_ANALYSIS = _bool_env(
        "FLIGHTGRID_DISPATCH_ENABLE_COLLISION_ANALYSIS",
        DISPATCH_EXECUTION_MODE != "fast",
    )
    DISPATCH_ENABLE_DETAILED_RESPONSE = _bool_env(
        "FLIGHTGRID_DISPATCH_ENABLE_DETAILED_RESPONSE",
        DISPATCH_EXECUTION_MODE != "fast",
    )
    DISPATCH_ENABLE_PLANNING_TRACE = _bool_env(
        "FLIGHTGRID_DISPATCH_ENABLE_PLANNING_TRACE",
        DISPATCH_EXECUTION_MODE != "fast",
    )
    DISPATCH_ENABLE_COARSE_BLOCKING_ZONE_GATE = _bool_env(
        "FLIGHTGRID_DISPATCH_ENABLE_COARSE_BLOCKING_ZONE_GATE",
        DISPATCH_EXECUTION_MODE == "fast",
    )
    DISPATCH_ENABLE_FEASIBLE_FIRST_MODE = _bool_env(
        "FLIGHTGRID_DISPATCH_ENABLE_FEASIBLE_FIRST_MODE",
        DISPATCH_EXECUTION_MODE == "fast",
    )


settings = Settings()
