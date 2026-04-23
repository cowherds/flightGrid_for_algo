"""
POCAS 后端核心数据模型模块

本模块定义了无人机调度系统的核心数据结构，包括地理位置、无人机、任务、
空间约束和路径点等数据类。这些数据类作为系统内部的标准数据格式，
用于算法处理和业务逻辑计算。
"""

from dataclasses import dataclass
from math import isclose, sqrt
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GeoPoint:
    """
    地理坐标点类

    表示三维空间中的一个点，用于表示无人机位置、任务位置等。

    属性:
        x (float): X 轴坐标（米）
        y (float): Y 轴坐标（米）
        z (float): Z 轴坐标（米，高度）
    """
    x: float
    y: float
    z: float

    def as_tuple(self):
        """
        将坐标点转换为元组格式

        返回:
            Tuple[float, float, float]: (x, y, z) 坐标元组
        """
        return (self.x, self.y, self.z)


@dataclass
class Drone:
    """
    无人机类

    表示系统中的一架无人机，包含其基本属性、当前状态和能源信息。

    属性:
        id (str): 无人机唯一标识符
        depotId (str): 所属站点的 ID
        maxRange (float): 最大航程（米）
        speed (float): 飞行速度（米/秒）
        capacity (float): 最大载荷容量（千克）
        returnToDepotRequired (bool): 是否必须返回站点
        currentLocation (GeoPoint): 当前位置坐标
        remainingRange (float): 剩余航程（米），默认 0.0
        currentLoad (float): 当前载荷（千克），默认 0.0
        currentTime (float): 当前时间（秒），默认 0.0
        battery (float): 电池电量百分比（0-100），默认 100.0
        maxFlightTime (Optional[float]): 最大飞行时间（秒），可选
        energyPerMeter (float): 每米能耗（百分比/米），默认 0.01
    """
    id: str
    depotId: str
    maxRange: float
    speed: float
    capacity: float
    returnToDepotRequired: bool
    currentLocation: GeoPoint
    remainingRange: float = 0.0
    currentLoad: float = 0.0
    currentTime: float = 0.0
    battery: float = 100.0           # 电池百分比 (0-100)
    maxFlightTime: Optional[float] = None  # 最大飞行时间 (秒)
    energyPerMeter: float = 0.01     # 每米能耗 (百分比/米)


@dataclass
class Task:
    """
    任务类

    表示需要由无人机执行的任务，包含任务的位置、时间窗口、优先级等信息。

    属性:
        id (str): 任务唯一标识符
        type (str): 任务类型（delivery/pickup/inspection）
        location (GeoPoint): 任务执行位置
        weight (float): 任务载荷（千克）
        timeWindow (Tuple[float, float]): 时间窗口 (开始时间, 结束时间)，单位秒
        serviceDuration (float): 服务时长（秒）
        is_sudden (bool): 是否为突发任务，默认 False
        priority (int): 优先级（0=低, 1=中, 2=高），默认 0
        groupId (Optional[str]): 任务所属的任务组 ID，用于表示订单中的多个任务
        sequence (int): 在任务组中的序列号，默认 0
        dependencies (list): 依赖的任务 ID 列表，表示任务间的依赖关系
    """
    id: str
    type: str
    location: GeoPoint
    weight: float
    timeWindow: Tuple[float, float]
    serviceDuration: float
    is_sudden: bool = False
    priority: int = 0  # 优先级 (0=低, 1=中, 2=高)
    groupId: Optional[str] = None  # 任务组 ID（用于任务-订单关系）
    sequence: int = 0  # 在任务组中的序列号
    dependencies: list = None  # 依赖的任务 ID 列表
    metadata: Optional[Dict[str, Any]] = None  # 附加调度提示，如 preferredDroneId / distanceToDepot


@dataclass
class SpatialConstraint:
    """
    空间约束类

    表示系统中的空间限制，如障碍物或禁飞区。

    属性:
        id (str): 约束唯一标识符
        kind (str): 约束类型（obstacle=障碍物, no_fly=禁飞区）
        shape (str): 形状类型（目前仅支持 box）
        box (Tuple[float, float, float, float, float, float]):
            边界框坐标 (x_min, y_min, z_min, x_max, y_max, z_max)
    """
    id: str
    kind: str
    shape: str
    box: Optional[Tuple[float, float, float, float, float, float]] = None
    polygon: Optional[List[Tuple[float, float]]] = None
    cylinder: Optional[Tuple[float, float, float, float, float]] = None
    startActiveTime: Optional[float] = None
    endActiveTime: Optional[float] = None
    zoneKind: str = "airspace_block"
    name: Optional[str] = None
    weatherType: Optional[str] = None
    speedFactor: Optional[float] = None
    allowPassThrough: bool = False

    def is_active_at(self, current_time: Optional[float]) -> bool:
        """
        判断约束在指定时刻是否生效。
        文档中的 startActiveTime / endActiveTime 允许暂时为空；
        为空时视为对应方向上无限制。
        """
        if current_time is None:
            return True

        if self.startActiveTime is not None and current_time < self.startActiveTime:
            return False

        if self.endActiveTime is not None and current_time > self.endActiveTime:
            return False

        return True

    def contains_point(self, point: GeoPoint, current_time: Optional[float] = None) -> bool:
        if not self.is_active_at(current_time):
            return False

        if self.shape == "box" and self.box:
            min_x, min_y, min_z, max_x, max_y, max_z = self.box
            return min_x <= point.x <= max_x and min_y <= point.y <= max_y and min_z <= point.z <= max_z

        if self.shape == "polygon" and self.polygon:
            min_z, max_z = self._vertical_limits()
            if not (min_z <= point.z <= max_z):
                return False
            return self._point_in_polygon(point.x, point.y)

        if self.shape == "cylinder" and self.cylinder:
            center_x, center_y, radius, min_z, max_z = self.cylinder
            if not (min_z <= point.z <= max_z):
                return False
            return sqrt((point.x - center_x) ** 2 + (point.y - center_y) ** 2) <= radius

        return False

    def blocks_flight_at(self, current_time: Optional[float] = None) -> bool:
        if not self.is_active_at(current_time):
            return False
        if self.zoneKind == "weather_slow" and self.allowPassThrough:
            return False
        return True

    def get_speed_factor(self, current_time: Optional[float] = None) -> float:
        if not self.is_active_at(current_time):
            return 1.0
        if self.zoneKind != "weather_slow":
            return 1.0
        if self.speedFactor is None:
            return 0.85
        try:
            return max(0.05, min(1.0, float(self.speedFactor)))
        except (TypeError, ValueError):
            return 0.85

    def _vertical_limits(self) -> Tuple[float, float]:
        if self.shape == "box" and self.box:
            return self.box[2], self.box[5]
        if self.shape == "polygon":
            if self.box:
                return self.box[2], self.box[5]
            return (0.0, 0.0)
        if self.shape == "cylinder" and self.cylinder:
            return self.cylinder[3], self.cylinder[4]
        return (0.0, 0.0)

    def _point_in_polygon(self, x: float, y: float) -> bool:
        if not self.polygon or len(self.polygon) < 3:
            return False

        if self._point_on_polygon_edge(x, y):
            return True

        inside = False
        p1x, p1y = self.polygon[0]
        vertex_count = len(self.polygon)
        for index in range(1, vertex_count + 1):
            p2x, p2y = self.polygon[index % vertex_count]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        else:
                            xinters = p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _point_on_polygon_edge(self, x: float, y: float) -> bool:
        if not self.polygon:
            return False

        epsilon = 1e-6
        for index, start in enumerate(self.polygon):
            end = self.polygon[(index + 1) % len(self.polygon)]
            if self._point_on_segment(x, y, start, end, epsilon):
                return True
        return False

    def _point_on_segment(
        self,
        x: float,
        y: float,
        start: Tuple[float, float],
        end: Tuple[float, float],
        epsilon: float,
    ) -> bool:
        x1, y1 = start
        x2, y2 = end
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if not isclose(cross, 0.0, abs_tol=epsilon):
            return False

        dot = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
        if dot < -epsilon:
            return False

        squared_length = (x2 - x1) ** 2 + (y2 - y1) ** 2
        return dot <= squared_length + epsilon


@dataclass
class PathPoint:
    """
    路径点类

    表示无人机飞行路径上的一个点，包含位置、时间、执行的动作等信息。

    属性:
        location (GeoPoint): 路径点的地理位置
        time (float): 到达该点的时间（秒）
        action (str): 在该点执行的动作（fly/pickup/delivery/wait 等）
        taskId (Optional[str]): 关联的任务 ID，如果有的话
        remainingRange (float): 到达该点时的剩余航程（米）
    """
    location: GeoPoint
    time: float
    action: str
    taskId: Optional[str]
    remainingRange: float
