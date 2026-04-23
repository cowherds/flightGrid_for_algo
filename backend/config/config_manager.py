"""
配置管理器 - 读取、管理和应用 YAML 配置
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器 - 读取和应用 YAML 配置"""

    _instance = None  # 单例模式
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_file: str = None):
        """初始化配置管理器"""
        if not config_file:
            # 默认配置文件路径
            config_file = os.path.join(
                os.path.dirname(__file__),
                "algorithm_config.yaml"
            )

        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """从 YAML 文件加载配置"""
        try:
            if not os.path.exists(self.config_file):
                logger.warning(f"配置文件不存在: {self.config_file}")
                return self._get_default_config()

            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"成功加载配置文件: {self.config_file}")
                return config or self._get_default_config()

        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self._get_default_config()

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "drones": {
                "default": {
                    "max_range": 10000,
                    "speed": 15,
                    "capacity": 10,
                    "min_altitude": 10,
                    "max_altitude": 150,
                    "safe_radius": 20,
                    "return_to_depot_required": True,
                    "battery_capacity": 5000,
                    "battery_consumption_rate": 50,
                }
            },
            "tasks": {
                "default": {
                    "weight": 1,
                    "type": "delivery",
                    "service_duration": 10,
                    "priority": 1,
                }
            },
            "path_planning": {
                "a_star": {
                    "grid_resolution": 20,
                    "max_iterations": 100000,
                    "timeout": 30,
                    "heuristic": "euclidean",
                    "smooth_path": True,
                    "smoothing_iterations": 5,
                },
                "dijkstra": {
                    "grid_resolution": 20,
                    "max_iterations": 100000,
                    "timeout": 30,
                    "smooth_path": True,
                    "smoothing_iterations": 5,
                },
            },
            "scheduling": {
                "insertion_heuristic": {
                    "max_iterations": 1000,
                    "insertion_cost_weight": 1.0,
                    "time_penalty": 1.0,
                    "distance_penalty": 1.0,
                },
                "nearest_neighbor": {
                    "greedy_factor": 1.0,
                    "randomization": 0.0,
                },
            },
            "collision_detection": {
                "safe_radius": 20,
                "collision_distance": 40,
                "time_step": 1.0,
                "line_of_sight_samples": 50,
                "warning_threshold": 50,
                "critical_threshold": 20,
                "collision_event_aggregation": 5,
            },
            "environment": {
                "space_limits": {
                    "x_max": 500,
                    "y_max": 500,
                    "z_max": 150,
                },
                "wind_speed": 0,
                "wind_direction": 0,
                "temperature": 20,
            },
            "global": {
                "log_level": "INFO",
                "log_file": "logs/pocas.log",
                "enable_caching": True,
                "cache_ttl": 3600,
                "debug_mode": False,
                "visualize_search": False,
            },
        }

    def get_drone_config(self, drone_id: str = None) -> Dict[str, Any]:
        """获取无人机配置"""
        if drone_id and drone_id in self.config.get("drones", {}):
            return self.config["drones"][drone_id]
        return self.config.get("drones", {}).get("default", {})

    def get_task_config(self, task_id: str = None) -> Dict[str, Any]:
        """获取任务配置"""
        if task_id and task_id in self.config.get("tasks", {}):
            return self.config["tasks"][task_id]
        return self.config.get("tasks", {}).get("default", {})

    def get_path_planning_config(self, algorithm: str = "a_star") -> Dict[str, Any]:
        """获取路径规划配置"""
        return self.config.get("path_planning", {}).get(algorithm, {})

    def get_scheduling_config(self, algorithm: str = "insertion_heuristic") -> Dict[str, Any]:
        """获取调度算法配置"""
        return self.config.get("scheduling", {}).get(algorithm, {})

    def get_collision_detection_config(self) -> Dict[str, Any]:
        """获取碰撞检测配置"""
        return self.config.get("collision_detection", {})

    def get_environment_config(self) -> Dict[str, Any]:
        """获取环境配置"""
        return self.config.get("environment", {})

    def get_global_config(self) -> Dict[str, Any]:
        """获取全局配置"""
        return self.config.get("global", {})

    def get_config(self, section: str = None, key: str = None) -> Any:
        """获取配置值"""
        if section is None:
            return self.config

        if section not in self.config:
            return None

        if key is None:
            return self.config[section]

        if isinstance(self.config[section], dict):
            return self.config[section].get(key)

        return None

    def update_config(self, section: str, key: str, value: Any) -> bool:
        """更新配置值"""
        try:
            if section not in self.config:
                self.config[section] = {}

            if isinstance(self.config[section], dict):
                self.config[section][key] = value
                logger.info(f"更新配置: {section}.{key} = {value}")
                return True
            else:
                logger.error(f"无法更新配置: {section} 不是字典类型")
                return False

        except Exception as e:
            logger.error(f"更新配置失败: {e}")
            return False

    def update_nested_config(self, path: str, value: Any) -> bool:
        """更新嵌套配置值 (支持点号分隔的路径)"""
        try:
            keys = path.split(".")
            config = self.config

            # 导航到父节点
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]

            # 设置值
            config[keys[-1]] = value
            logger.info(f"更新配置: {path} = {value}")
            return True

        except Exception as e:
            logger.error(f"更新嵌套配置失败: {e}")
            return False

    def save_config(self) -> bool:
        """保存配置到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                logger.info(f"配置已保存到: {self.config_file}")
                return True

        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False

    def reload_config(self) -> bool:
        """重新加载配置"""
        try:
            self.config = self._load_config()
            logger.info("配置已重新加载")
            return True

        except Exception as e:
            logger.error(f"重新加载配置失败: {e}")
            return False

    def export_config(self, export_file: str) -> bool:
        """导出配置到文件"""
        try:
            os.makedirs(os.path.dirname(export_file), exist_ok=True)

            with open(export_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                logger.info(f"配置已导出到: {export_file}")
                return True

        except Exception as e:
            logger.error(f"导出配置失败: {e}")
            return False

    def import_config(self, import_file: str) -> bool:
        """从文件导入配置"""
        try:
            if not os.path.exists(import_file):
                logger.error(f"导入文件不存在: {import_file}")
                return False

            with open(import_file, 'r', encoding='utf-8') as f:
                imported_config = yaml.safe_load(f)
                if imported_config:
                    self.config.update(imported_config)
                    logger.info(f"配置已从 {import_file} 导入")
                    return True
                else:
                    logger.error("导入的配置为空")
                    return False

        except Exception as e:
            logger.error(f"导入配置失败: {e}")
            return False

    def reset_to_default(self) -> bool:
        """重置为默认配置"""
        try:
            self.config = self._get_default_config()
            logger.info("配置已重置为默认值")
            return True

        except Exception as e:
            logger.error(f"重置配置失败: {e}")
            return False

    def validate_config(self) -> tuple[bool, list]:
        """验证配置的有效性"""
        errors = []

        # 验证路径规划参数
        for algo in ["a_star", "dijkstra"]:
            config = self.get_path_planning_config(algo)
            if config.get("grid_resolution", 0) <= 0:
                errors.append(f"{algo}.grid_resolution 必须大于 0")
            if config.get("max_iterations", 0) <= 0:
                errors.append(f"{algo}.max_iterations 必须大于 0")

        # 验证碰撞检测参数
        collision_config = self.get_collision_detection_config()
        if collision_config.get("safe_radius", 0) <= 0:
            errors.append("collision_detection.safe_radius 必须大于 0")
        if collision_config.get("time_step", 0) <= 0:
            errors.append("collision_detection.time_step 必须大于 0")

        # 验证环境参数
        env_config = self.get_environment_config()
        space_limits = env_config.get("space_limits", {})
        if space_limits.get("x_max", 0) <= 0:
            errors.append("environment.space_limits.x_max 必须大于 0")
        if space_limits.get("y_max", 0) <= 0:
            errors.append("environment.space_limits.y_max 必须大于 0")
        if space_limits.get("z_max", 0) <= 0:
            errors.append("environment.space_limits.z_max 必须大于 0")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return self.config.copy()


# 全局配置管理器实例
config_manager = ConfigManager()
