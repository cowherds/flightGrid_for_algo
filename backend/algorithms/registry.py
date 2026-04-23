"""
POCAS 后端算法注册表模块

本模块提供了一个中央注册表，用于管理所有可用的路径规划算法和调度算法。
通过装饰器模式，算法实现类可以自动注册到系统中，便于动态加载和选择。

主要功能：
- 注册路径规划算法
- 注册调度算法
- 获取已注册的算法类
"""

from typing import Dict, Type


class AlgorithmRegistry:
    """
    算法注册表类

    提供了一个中央注册表，用于管理系统中所有可用的算法。
    支持通过装饰器动态注册新算法，以及通过名称获取算法类。

    类属性:
        _path_planners (Dict[str, Type]): 路径规划算法注册表，key 为算法名称，value 为算法类
        _schedulers (Dict[str, Type]): 调度算法注册表，key 为算法名称，value 为算法类
    """

    _path_planners: Dict[str, Type] = {}
    _schedulers: Dict[str, Type] = {}

    @classmethod
    def register_path_planner(cls, name: str):
        """
        路径规划算法注册装饰器

        用于将路径规划算法类注册到系统中。通过装饰器模式，
        使得算法类可以自动注册而无需手动添加到注册表。

        参数:
            name (str): 算法的唯一名称，用于后续查询和使用

        返回:
            function: 装饰器函数

        使用示例:
            @AlgorithmRegistry.register_path_planner("a_star")
            class AStarPlanner(BasePathPlanner):
                pass

        注册后可通过 AlgorithmRegistry.get_path_planner("a_star") 获取该类
        """
        def wrapper(algorithm_cls):
            """
            装饰器包装函数

            参数:
                algorithm_cls (Type): 要注册的算法类

            返回:
                Type: 原始的算法类（装饰器不修改类）
            """
            cls._path_planners[name] = algorithm_cls
            return algorithm_cls
        return wrapper

    @classmethod
    def register_scheduler(cls, name: str):
        """
        调度算法注册装饰器

        用于将调度算法类注册到系统中。通过装饰器模式，
        使得算法类可以自动注册而无需手动添加到注册表。

        参数:
            name (str): 算法的唯一名称，用于后续查询和使用

        返回:
            function: 装饰器函数

        使用示例:
            @AlgorithmRegistry.register_scheduler("insertion_heuristic")
            class InsertionHeuristicScheduler(BaseScheduler):
                pass

        注册后可通过 AlgorithmRegistry.get_scheduler("insertion_heuristic") 获取该类
        """
        def wrapper(algorithm_cls):
            """
            装饰器包装函数

            参数:
                algorithm_cls (Type): 要注册的算法类

            返回:
                Type: 原始的算法类（装饰器不修改类）
            """
            cls._schedulers[name] = algorithm_cls
            return algorithm_cls
        return wrapper

    @classmethod
    def get_path_planner(cls, name: str):
        """
        获取已注册的路径规划算法类

        根据算法名称从注册表中获取对应的算法类。

        参数:
            name (str): 算法名称

        返回:
            Type: 算法类

        异常:
            ValueError: 如果指定名称的算法未注册

        使用示例:
            planner_cls = AlgorithmRegistry.get_path_planner("a_star")
            planner = planner_cls(env)
        """
        if name not in cls._path_planners:
            raise ValueError(f"Path planner '{name}' not found.")
        return cls._path_planners[name]

    @classmethod
    def get_scheduler(cls, name: str):
        """
        获取已注册的调度算法类

        根据算法名称从注册表中获取对应的算法类。

        参数:
            name (str): 算法名称

        返回:
            Type: 算法类

        异常:
            ValueError: 如果指定名称的算法未注册

        使用示例:
            scheduler_cls = AlgorithmRegistry.get_scheduler("insertion_heuristic")
            scheduler = scheduler_cls(planner, depots, depot_mgr)
        """
        if name not in cls._schedulers:
            raise ValueError(f"Scheduler '{name}' not found.")
        return cls._schedulers[name]

    @classmethod
    def get_available_path_planners(cls) -> list[str]:
        """返回当前已注册的路径规划算法名称列表。"""

        return sorted(cls._path_planners.keys())

    @classmethod
    def get_available_schedulers(cls) -> list[str]:
        """返回当前已注册的调度算法名称列表。"""

        return sorted(cls._schedulers.keys())
