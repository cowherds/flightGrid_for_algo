"""算法模块自动发现与加载。"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_LOADED = False


def _iter_algorithm_modules() -> list[str]:
    """扫描算法目录并返回可导入模块路径。"""

    algorithms_dir = Path(__file__).resolve().parent
    backend_dir = algorithms_dir.parent
    module_names: list[str] = []

    for subdir in ("path_planning", "scheduling"):
        root = algorithms_dir / subdir
        if not root.exists():
            continue

        for file_path in root.rglob("*.py"):
            if file_path.name == "__init__.py" or file_path.name.startswith("_"):
                continue

            relative_path = file_path.relative_to(backend_dir).with_suffix("")
            module_name = "backend." + ".".join(relative_path.parts)
            module_names.append(module_name)

    return sorted(set(module_names))


def load_algorithms_once() -> None:
    """自动加载算法模块（幂等）。"""

    global _LOADED
    if _LOADED:
        return

    for module_name in _iter_algorithm_modules():
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover
            logger.warning("导入算法模块失败: %s (%s)", module_name, exc)

    _LOADED = True
