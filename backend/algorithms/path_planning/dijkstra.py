"""
Dijkstra路径规划算法 - 3D环境中的最优路径搜索

该模块实现了Dijkstra算法在3D城市环境中的应用，用于为无人机规划避障的最优路径。

算法原理：
- 使用Dijkstra算法在3D网格中寻找最短路径
- 不使用启发函数，仅基于实际距离进行搜索
- 支持26邻域移动（3D空间中的所有方向）
- 集成路径缓存机制，避免重复计算

主要特性：
- 支持3D环境中的路径规划
- 支持直线视线检测（LOS），快速返回直线路径
- 支持碰撞检测，避开禁飞区和障碍物
- 集成路径缓存，提高性能
- 支持自定义网格分辨率

算法复杂度：
- 时间复杂度：O(n*log(n)) 其中n为搜索空间大小
- 空间复杂度：O(n) 用于存储开放集和闭合集

适用场景：
- 3D城市环境中的无人机路径规划
- 需要避开禁飞区和障碍物的场景
- 对路径质量有要求的场景

限制条件：
- 网格分辨率影响路径精度和计算速度
- 大规模环境中计算复杂度较高
- 假设环境静态不变
- 相比A*算法，Dijkstra算法搜索效率较低（无启发函数）
"""

import heapq
import numpy as np
from typing import List, Tuple
from backend.models.domain import GeoPoint
from backend.algorithms.base import BasePathPlanner
from backend.algorithms.registry import AlgorithmRegistry
from backend.config.settings import settings


@AlgorithmRegistry.register_path_planner("dijkstra_3d")
class DijkstraPlanner(BasePathPlanner):
    """
    Dijkstra路径规划器 - 3D环境中的最优路径搜索

    该类实现了Dijkstra算法在3D城市环境中的应用。Dijkstra算法是一种经典的
    最短路径算法，通过逐步扩展已知最短距离的节点来找到最优路径。

    属性：
        env: 城市环境模型，包含地形、禁飞区等约束信息
        res: 网格分辨率，用于将连续坐标离散化为网格
        cache: 路径缓存字典，key为(起点, 终点)元组，value为(路径, 距离)

    适用场景：
        - 3D城市环境中的无人机路径规划
        - 需要避开禁飞区和障碍物的场景
        - 对路径质量有要求的场景

    限制条件：
        - 网格分辨率影响路径精度和计算速度
        - 大规模环境中计算复杂度较高
        - 假设环境静态不变
        - 相比A*算法，搜索效率较低
    """

    def __init__(self, env, grid_resolution=None):
        """
        初始化Dijkstra路径规划器

        参数：
            env: 城市环境模型，包含地形、禁飞区等约束信息
        """
        super().__init__(env)
        # 从配置中获取网格分辨率（单位：米）
        self.res = grid_resolution or settings.DEFAULT_GRID_RESOLUTION
        self.max_expansions = 40000
        # 初始化路径缓存，用于存储已计算的路径
        self.cache = {}

    def get_route(
        self,
        start: GeoPoint,
        goal: GeoPoint,
        current_time=None,
        speed=None,
    ) -> Tuple[List[GeoPoint], float]:
        """
        规划从起点到终点的最优路径

        该方法是Dijkstra路径规划器的主入口。算法流程：
        1. 检查缓存中是否已有该路径
        2. 检查起点和终点间是否有直线视线（LOS）
        3. 如果有LOS，直接返回直线路径
        4. 否则，使用Dijkstra算法在网格中搜索最优路径
        5. 将结果缓存以供后续使用

        参数：
            start (GeoPoint): 起点坐标
            goal (GeoPoint): 终点坐标

        返回值：
            Tuple[List[GeoPoint], float]: (路径点列表, 总距离)
                - 路径点列表：从起点到终点的所有路点（包含起点和终点）
                - 总距离：路径的总长度（米）

        时间复杂度：O(n*log(n)) 其中n为搜索空间大小
        空间复杂度：O(n)

        示例：
            >>> planner = DijkstraPlanner(env)
            >>> path, distance = planner.get_route(start, goal)
            >>> print(f"路径长度: {distance} 米, 路点数: {len(path)}")
        """
        timer_token = self.start_route_timer()
        try:
            self.check_cancelled()
            self.record_route_request()
            key = self._build_cache_key(start, goal, current_time, speed)

            # 检查缓存中是否已有该路径
            if key in self.cache:
                self.record_cache_hit()
                return self.cache[key]
            self.record_cache_miss()

            # 检查起点和终点间是否有直线视线（LOS）
            estimated_arrival = self._estimate_arrival_time(
                start,
                goal,
                current_time=current_time,
                speed=speed,
            )

            if self.env.is_collision(start, current_time=current_time) or self.env.is_collision(goal, current_time=estimated_arrival):
                self.cache[key] = ([], float("inf"))
                return self.cache[key]

            if self.env.line_of_sight(
                start,
                goal,
                start_time=current_time,
                end_time=estimated_arrival,
            ):
                # 计算直线距离
                dist = np.linalg.norm(np.array(start.as_tuple()) - np.array(goal.as_tuple()))
                # 缓存直线路径
                self.cache[key] = ([start, goal], dist)
                return self.cache[key]

            # 将连续坐标离散化为网格坐标
            start_g = (round(start.x/self.res)*self.res, round(start.y/self.res)*self.res, round(start.z/self.res)*self.res)
            goal_g = (round(goal.x/self.res)*self.res, round(goal.y/self.res)*self.res, round(goal.z/self.res)*self.res)

            # 初始化Dijkstra算法的数据结构
            open_set = []  # 优先队列：(距离值, 网格坐标)
            heapq.heappush(open_set, (0, start_g))
            came_from = {}  # 记录每个节点的前驱节点，用于路径回溯
            g_score = {start_g: 0}  # 记录从起点到每个节点的实际距离
            visited = set()
            expansions = 0

            # 生成3D网格的26邻域方向（包括对角线移动）
            # 3D空间中每个节点有26个邻域（3^3 - 1）
            dirs = [(dx, dy, dz) for dx in [-self.res, 0, self.res]
                    for dy in [-self.res, 0, self.res]
                    for dz in [-self.res, 0, self.res]
                    if dx!=0 or dy!=0 or dz!=0]

            # Dijkstra主循环
            while open_set:
                self.check_cancelled()
                # 从优先队列中取出距离最小的节点
                curr = heapq.heappop(open_set)[1]
                if curr in visited:
                    continue
                visited.add(curr)
                expansions += 1

                if expansions > self.max_expansions:
                    break
                current_point = GeoPoint(*curr)

                current_node_time = None
                if current_time is not None and speed and speed > 0:
                    current_node_time = current_time + g_score[curr] / speed
                goal_arrival_time = None
                if current_node_time is not None:
                    goal_arrival_time = self.env.estimate_arrival_time(
                        current_point,
                        goal,
                        current_node_time,
                        speed,
                    )

                if self._strict_line_of_sight(
                    current_point,
                    goal,
                    start_time=current_node_time,
                    end_time=goal_arrival_time,
                ):
                    path = []
                    trace = curr
                    while trace in came_from:
                        path.append(GeoPoint(*trace))
                        trace = came_from[trace]
                    path.reverse()
                    full_path = [start] + path + [goal]
                    dist = self._path_distance(full_path)
                    self.cache[key] = (full_path, dist)
                    return self.cache[key]

                # 探索当前节点的所有邻域
                for dx, dy, dz in dirs:
                    # 计算邻域节点坐标
                    nb = (curr[0]+dx, curr[1]+dy, curr[2]+dz)

                    # 检查邻域节点是否与障碍物碰撞
                    neighbor_time = None
                    if current_time is not None and speed and speed > 0:
                        edge_start_time = current_time + g_score[curr] / speed
                        neighbor_time = self.env.estimate_arrival_time(
                            current_point,
                            GeoPoint(*nb),
                            edge_start_time,
                            speed,
                        )

                    if self.env.is_collision(GeoPoint(*nb), current_time=neighbor_time):
                        continue

                    if current_time is not None and speed and speed > 0:
                        edge_start_time = current_time + g_score[curr] / speed
                    else:
                        edge_start_time = None

                    if not self.env.line_of_sight(
                        current_point,
                        GeoPoint(*nb),
                        start_time=edge_start_time,
                        end_time=neighbor_time,
                    ):
                        continue

                    # 计算移动成本（欧几里得距离）
                    cost = np.linalg.norm([dx, dy, dz])

                    # 计算从起点经过当前节点到邻域节点的距离
                    tentative_g = g_score[curr] + cost

                    # 如果找到更短的路径，更新邻域节点的信息
                    # Dijkstra核心特点：仅使用实际距离g_score，不使用启发函数h
                    if nb not in g_score or tentative_g < g_score[nb]:
                        came_from[nb] = curr
                        g_score[nb] = tentative_g
                        # 直接使用g_score作为优先队列的排序键（不加启发函数）
                        heapq.heappush(open_set, (g_score[nb], nb))

            # 如果无法找到路径，返回直线路径并标记为不可达
            self.cache[key] = ([start, goal], float('inf'))
            return self.cache[key]
        finally:
            self.finish_route_timer(timer_token)
