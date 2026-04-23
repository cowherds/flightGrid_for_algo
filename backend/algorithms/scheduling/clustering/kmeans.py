"""
K-Means 聚类调度算法

本模块实现了基于 K-Means 聚类的无人机调度算法。
该算法将任务聚类到不同的无人机，使得每架无人机负责一个聚类中的所有任务。

算法流程：
1. 初始化聚类中心为无人机的当前位置
2. 迭代多次：
   - 将每个任务分配给最近的聚类中心（无人机）
   - 重新计算聚类中心为已分配任务的几何中心
3. 返回最终的任务分配结果

优点：
- 能够自动聚类任务，减少无人机间的干扰
- 聚类中心会逐步优化，提高分配效率

缺点：
- 需要多次迭代，计算量较大
- 可能陷入局部最优
"""

import numpy as np
from typing import List, Dict, Tuple
from backend.models.domain import Drone, Task, PathPoint, GeoPoint
from backend.algorithms.registry import AlgorithmRegistry
from backend.algorithms.scheduling.optimization.insertion import InsertionScheduler


@AlgorithmRegistry.register_scheduler("kmeans")
class KMeansScheduler(InsertionScheduler):
    """
    K-Means 聚类调度器

    功能说明：
    - 使用 K-Means 聚类算法进行任务分配
    - 每架无人机作为一个聚类中心
    - 任务被分配给最近的聚类中心

    属性：
    - planner: 路径规划器
    - depots: 站点字典
    - depot_mgr: 站点管理器
    """

    def plan(self, current_drones: List[Drone], unassigned_tasks: List[Task]) -> Dict[str, List[Task]]:
        """
        执行 K-Means 聚类调度

        参数：
        - current_drones: 当前可用的无人机列表
        - unassigned_tasks: 未分配的任务列表

        返回值：
        - 调度结果字典，key 为无人机 ID，value 为分配给该无人机的任务列表

        功能说明：
        - 初始化聚类中心为无人机的当前位置
        - 迭代进行任务分配和聚类中心更新
        - 最多迭代 3 次以保证计算速度
        """
        self.reset_route_profile_cache()
        self.reset_runtime_stats()
        routes = {d.id: [] for d in current_drones}
        tasks = list(unassigned_tasks)

        # 1. 初始化聚类中心 (初始为无人机当前位置)
        centroids = {d.id: np.array(d.currentLocation.as_tuple()) for d in current_drones}

        max_iterations = 3  # 控制迭代次数，保证计算速度
        previous_signature: tuple[tuple[str, tuple[str, ...]], ...] | None = None
        for _iteration in range(max_iterations):
            iteration_routes = {d.id: [] for d in current_drones}

            # 为每个任务寻找最近且满足约束的聚类中心
            for task in sorted(tasks, key=lambda item: (-item.priority, item.timeWindow[1], item.id)):
                best_drone_id = None
                min_dist = float('inf')
                task_arr = np.array(task.location.as_tuple())

                # 按距离聚类中心的距离排序无人机
                sorted_drones = sorted(current_drones, key=lambda d: np.linalg.norm(task_arr - centroids[d.id]))

                for d in sorted_drones:
                    test_route = iteration_routes[d.id] + [task]
                    _profile, valid, _ = self.evaluate_route_candidate(d, test_route)
                    if valid:
                        dist = np.linalg.norm(task_arr - centroids[d.id])
                        if dist < min_dist:
                            min_dist = dist
                            best_drone_id = d.id
                            break  # 找到最近且合法的就分配

                if best_drone_id is not None:
                    iteration_routes[best_drone_id].append(task)

            # 2. 重新计算聚类中心 (基于已分配任务的几何中心)
            for d in current_drones:
                if iteration_routes[d.id]:
                    pts = [np.array(t.location.as_tuple()) for t in iteration_routes[d.id]]
                    centroids[d.id] = np.mean(pts, axis=0)
                else:
                    centroids[d.id] = np.array(d.currentLocation.as_tuple())

            routes = iteration_routes
            current_signature = tuple(
                (drone.id, tuple(task.id for task in routes[drone.id]))
                for drone in current_drones
            )
            if current_signature == previous_signature:
                break
            previous_signature = current_signature

        for drone in current_drones:
            ordered_cluster_tasks = sorted(
                routes[drone.id],
                key=lambda task: (-task.priority, task.timeWindow[1], task.location.x, task.location.y),
            )
            routes[drone.id] = self._build_route_by_greedy_inserts(drone, ordered_cluster_tasks)

        return routes

    # ============================================================================
    # 约束评估和路径生成方法
    # ============================================================================

    def _evaluate_route(self, drone: Drone, route: List[Task]) -> Tuple[bool, float]:
        """
        评估路由的有效性和成本

        参数：
        - drone: 无人机对象
        - route: 任务列表

        返回值：
        - (是否有效, 成本)
        - 有效时返回 (True, 完成时间)
        - 无效时返回 (False, inf)

        功能说明：
        - 检查载重约束：总载重不能超过无人机容量
        - 检查时间窗口约束：到达时间必须在任务的时间窗口内
        - 检查航程约束：总飞行距离不能超过剩余航程
        - 返回路由的完成时间作为成本
        """
        return self.evaluate_route_with_completion_cost(drone, route)

    def generate_path_points(self, drone: Drone, route: List[Task]) -> List[PathPoint]:
        """
        为无人机生成路径点

        参数：
        - drone: 无人机对象
        - route: 任务列表（已排序）

        返回值：
        - 路径点列表

        功能说明：
        - 生成无人机从起点到终点的完整路径
        - 包括起飞、飞行、服务、返航、着陆等动作
        - 每个路径点记录位置、时间、动作和剩余航程
        """
        pts = []
        current_time, current_range, current_loc = drone.currentTime, drone.remainingRange, drone.currentLocation
        if drone.currentTime == 0:
            actual_takeoff_time = self.depot_mgr.get_available_time(drone.depotId, current_time)
            if actual_takeoff_time > current_time:
                pts.append(PathPoint(current_loc, current_time, "wait", None, current_range))
                current_time = actual_takeoff_time
            pts.append(PathPoint(current_loc, current_time, "takeoff", None, current_range))
        else:
            pts.append(PathPoint(current_loc, current_time, "fly", None, current_range))
        for task in route:
            self.planner.check_cancelled()
            path, dist = self._require_planned_route(
                current_loc,
                task.location,
                current_time=current_time,
                speed=drone.speed,
                label=f"{drone.id}->{task.id}",
            )
            for pt in path[1:-1]:
                segment_distance = np.linalg.norm(np.array(current_loc.as_tuple()) - np.array(pt.as_tuple()))
                current_time += self._segment_travel_time(
                    current_loc,
                    pt,
                    speed=drone.speed,
                    current_time=current_time,
                )
                current_range -= segment_distance
                current_loc = pt
                pts.append(PathPoint(pt, current_time, "fly", None, current_range))
            final_segment_dist = np.linalg.norm(
                np.array(current_loc.as_tuple()) - np.array(task.location.as_tuple())
            )
            current_time += self._segment_travel_time(
                current_loc,
                task.location,
                speed=drone.speed,
                current_time=current_time,
            )
            current_range -= final_segment_dist
            current_loc = task.location
            wait_time = max(0, task.timeWindow[0] - current_time)
            if wait_time > 0:
                pts.append(PathPoint(current_loc, current_time, "wait", task.id, current_range))
                current_time += wait_time
                current_range -= wait_time * 2
            pts.append(PathPoint(current_loc, current_time, "serve", task.id, current_range))
            current_time += task.serviceDuration
        return_depot_id = None
        depot_loc = None
        if drone.returnToDepotRequired:
            return_choice = self._select_return_depot(
                drone,
                current_loc,
                current_time=current_time,
                speed=drone.speed,
                exact=True,
            )
            if return_choice is None:
                raise RuntimeError(f"{drone.id} 无法返回任何站点")
            return_depot_id, depot_loc, _return_dist = return_choice

        if drone.returnToDepotRequired and current_loc.as_tuple() != depot_loc.as_tuple():
            home_path, dist = self._require_planned_route(
                current_loc,
                depot_loc,
                current_time=current_time,
                speed=drone.speed,
                label=f"{drone.id} 返航",
            )
            for pt in home_path[1:]:
                segment_distance = np.linalg.norm(np.array(current_loc.as_tuple()) - np.array(pt.as_tuple()))
                current_range -= segment_distance
                current_time += self._segment_travel_time(
                    current_loc,
                    pt,
                    speed=drone.speed,
                    current_time=current_time,
                )
                current_loc = pt
                pts.append(PathPoint(pt, current_time, "fly", None, current_range))
            actual_land_time = self.depot_mgr.get_available_time(return_depot_id, current_time)
            if actual_land_time > current_time:
                pts.append(PathPoint(depot_loc, current_time, "wait", "HOVER_FOR_LANDING", current_range))
                wait_duration = actual_land_time - current_time
                current_time = actual_land_time
                current_range -= wait_duration * 2
            pts.append(PathPoint(depot_loc, current_time, "land", None, current_range))
        return pts
