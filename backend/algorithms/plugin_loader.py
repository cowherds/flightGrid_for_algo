"""
algo_run 无装饰器插件加载：通过 ``algo_plugins.json`` 将「外部 ID」映射到 ``module:class``。

与 ``AlgorithmRegistry`` 装饰器注册解耦：新算法只需在 ``backend/algorithms/`` 下实现类，
并在 ``algo_plugins.json`` 中增加条目即可被 ``algo_run.py`` / ``scene_dispatch_runner`` 加载。

离线脚本优先查本清单，未命中再回退注册表。
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Type

_MANIFEST_NAME = "algo_plugins.json"


def manifest_path() -> Path:
    return Path(__file__).resolve().parent / _MANIFEST_NAME


def load_manifest() -> dict[str, Any]:
    path = manifest_path()
    if not path.is_file():
        return {"planners": {}, "schedulers": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"planners": {}, "schedulers": {}}
    data.setdefault("planners", {})
    data.setdefault("schedulers", {})
    return data


def _import_class(module: str, class_name: str) -> Type[Any]:
    mod = importlib.import_module(module.strip())
    cls = getattr(mod, class_name.strip())
    if not isinstance(cls, type):
        raise TypeError(f"{module}:{class_name} 不是类")
    return cls


def _try_kind(manifest: dict[str, Any], kind: str, key: str) -> Type[Any] | None:
    bucket = manifest.get(kind) or {}
    if not isinstance(bucket, dict):
        return None
    entry = bucket.get(key.strip())
    if not entry or not isinstance(entry, dict):
        return None
    mod = entry.get("module")
    cls_name = entry.get("class")
    if not mod or not cls_name:
        return None
    return _import_class(str(mod), str(cls_name))


def try_load_planner_class(algorithm_id: str, *, alternate_key: str | None = None) -> Type[Any] | None:
    """按请求中的 algorithmId 尝试加载；可选第二键（如 catalog 解析后的内部名）。"""
    manifest = load_manifest()
    for key in (algorithm_id, alternate_key):
        if not key:
            continue
        cls = _try_kind(manifest, "planners", key)
        if cls is not None:
            return cls
    return None


def try_load_scheduler_class(scheduling_id: str, *, alternate_key: str | None = None) -> Type[Any] | None:
    manifest = load_manifest()
    for key in (scheduling_id, alternate_key):
        if not key:
            continue
        cls = _try_kind(manifest, "schedulers", key)
        if cls is not None:
            return cls
    return None


def list_plugin_ids() -> dict[str, list[str]]:
    m = load_manifest()
    pl = sorted((m.get("planners") or {}).keys()) if isinstance(m.get("planners"), dict) else []
    sc = sorted((m.get("schedulers") or {}).keys()) if isinstance(m.get("schedulers"), dict) else []
    return {"planners": pl, "schedulers": sc}
