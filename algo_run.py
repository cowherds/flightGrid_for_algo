#!/usr/bin/env python3
"""
离线场景调度：读取场景 JSON（默认 ``data/simple.json``），运行 DispatchEngine。

任意「新建分配算法 + 新建路径规划算法」只要已注册到 ``AlgorithmRegistry``，并（可选）在
``backend/algorithm_catalog.py`` 里配置外部 ID 映射，即可用本脚本验证::

  python3 algo_run.py -s <你的调度器外部ID> -p <你的规划器外部ID> 场景.json

接入新算法简要检查单
--------------------
1. 调度器：``@AlgorithmRegistry.register_scheduler("internal_id")`` 装饰器类。
2. 规划器：``@AlgorithmRegistry.register_path_planner("internal_id")`` 装饰器类；模块需被
   ``backend/algorithms/discovery`` 链式导入（与同目录其它算法一致）。
3. 若希望使用简短外部名（如 ``my_scheduler``）：在 ``backend/algorithm_catalog.py`` 的
   ``SCHEDULER_EXTERNAL_TO_INTERNAL`` / ``PLANNER_EXTERNAL_TO_INTERNAL`` 与
   ``CANONICAL_*`` 中增加映射后，``python3 algo_run.py --list`` 应能看到该 ID。
4. 默认组合仍为 **lmta + ovs**（与 ``-s lmta -p ovs`` / ``--preset lmta+ovs`` 等价）。

模式
----
1. 解耦（默认）：``-s`` / ``-p`` 分别指定任务分配与路径规划（外部 ID，与 ``algorithm_catalog`` 一致）。
2. 耦合到场景：``--from-scene-algorithms`` 使用 JSON 里 scene 已保存的 ``schedulingId`` + ``algorithmId``。
3. 预设：``--preset lmta+ovs`` 等；显式传入 ``-s``/``-p`` 时覆盖预设里对应项。

环境变量（可选）
----------------
  FLIGHTGRID_ALGO_RUN_SCENE   默认场景路径
  FLIGHTGRID_SCHEDULER        默认调度外部 ID
  FLIGHTGRID_PLANNER          默认规划外部 ID
  FLIGHTGRID_EVALUATION_PLANNER  默认估价规划器外部 ID（与主规划器可不同）

运行结束后默认将评价指标写入场景同目录 ``<场景 stem>_algo_metrics.txt``（UTF-8）；
可用 ``--metrics-out PATH`` 指定路径，``--no-metrics-file`` 关闭写入。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 预设名 -> (schedulingId, algorithmId)  均为 catalog 外部 ID
ALGO_PRESETS: dict[str, tuple[str, str]] = {
    "lmta+ovs": ("lmta", "ovs"),
    "lmta-ovs": ("lmta", "ovs"),
    "nearest+astar": ("nearest_neighbor", "astar_v2"),
    "nearest+fast": ("nearest_neighbor", "weighted_astar_fast"),
    "nearest+quality": ("nearest_neighbor", "weighted_astar_quality"),
    "nearest+geometric_speed": ("nearest_neighbor", "geometric_heuristic_speed"),
    "nearest+geometric_quality": ("nearest_neighbor", "geometric_heuristic_quality"),
    "nearest+orthogonal": ("nearest_neighbor", "orthogonal_safe"),
    "nearest+direct_safe": ("nearest_neighbor", "direct_safe"),
    "insertion+astar": ("insertion_heuristic", "astar_v2"),
    "insertion+fast": ("insertion_heuristic", "weighted_astar_fast"),
    "insertion+geometric_speed": ("insertion_heuristic", "geometric_heuristic_speed"),
    "insertion+geometric_quality": ("insertion_heuristic", "geometric_heuristic_quality"),
    "insertion+ovs": ("insertion_heuristic", "ovs"),
    "kmeans+fast": ("kmeans", "weighted_astar_fast"),
    "two_level+fast": ("two_level_scheduler", "weighted_astar_fast"),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_scene_path(root: Path, scene_arg: str) -> Path | None:
    """解析场景文件路径：绝对/相对路径，或仓库根下 ``data/<文件名>``。"""
    p = Path(scene_arg)
    if p.is_file():
        return p.resolve()
    name = Path(scene_arg).name
    for base in (root /"data", root / "data"):
        cand = base / name
        if cand.is_file():
            return cand.resolve()
    return None


def _resolve_scheduler_planner(args: argparse.Namespace) -> tuple[str, str, bool]:
    """
    返回 (scheduler_external, planner_external, use_scene_algorithms)。
    """
    if getattr(args, "from_scene_algorithms", False):
        return ("", "", True)

    preset_key = (args.preset or "").strip().lower()
    preset_pair = ALGO_PRESETS.get(preset_key) if preset_key else None
    if preset_key and not preset_pair:
        raise SystemExit(f"未知预设 '{args.preset}'。使用 --list-presets 查看可用预设。")

    env_s = os.environ.get("FLIGHTGRID_SCHEDULER", "").strip()
    env_p = os.environ.get("FLIGHTGRID_PLANNER", "").strip()

    sch = (args.scheduler or (preset_pair[0] if preset_pair else None) or env_s or "lmta").strip()
    pln = (args.planner or (preset_pair[1] if preset_pair else None) or env_p or "ovs").strip()
    return (sch, pln, False)


def _cmd_list_presets() -> None:
    print("预设 (--preset <名称>)：")
    for name, (s, p) in sorted(ALGO_PRESETS.items()):
        print(f"  {name:32}  schedulingId={s}  algorithmId={p}")


def _cmd_list_algorithms() -> int:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from backend.algorithm_catalog import get_available_algorithms_payload

    from backend.algorithms.plugin_loader import list_plugin_ids

    payload = get_available_algorithms_payload()
    print("调度器 schedulingId（注册表，节选）：")
    for x in payload.get("schedulers", []):
        print(f"  {x}")
    print("\n路径规划 algorithmId（注册表，节选）：")
    for x in payload.get("planners", []):
        print(f"  {x}")
    plug = list_plugin_ids()
    if plug.get("schedulers") or plug.get("planners"):
        print("\nalgo_plugins.json 插件 ID（无需 @AlgorithmRegistry 装饰器）：")
        for x in plug.get("schedulers", []):
            print(f"  [scheduler] {x}")
        for x in plug.get("planners", []):
            print(f"  [planner]   {x}")
    return 0


def main() -> int:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    default_scene = os.environ.get("FLIGHTGRID_ALGO_RUN_SCENE", "").strip()
    if not default_scene:
        default_scene = str(root / "data" / "simple.json")

    parser = argparse.ArgumentParser(
        description="离线运行场景调度：解耦指定 -s/-p，或 --from-scene-algorithms 使用场景内算法对。",
    )
    parser.add_argument(
        "scene_json",
        nargs="?",
        default=default_scene,
        help="场景 bundle JSON（默认见环境变量 FLIGHTGRID_ALGO_RUN_SCENE）",
    )
    parser.add_argument(
        "-s",
        "--scheduler",
        default=None,
        metavar="ID",
        help="任务分配 / 调度器外部 ID，如 lmta、insertion_heuristic、nearest_neighbor",
    )
    parser.add_argument(
        "-p",
        "--planner",
        default=None,
        metavar="ID",
        help="路径规划器外部 ID，如 ovs、astar_v2、weighted_astar_fast",
    )
    parser.add_argument(
        "-e",
        "--evaluation-planner",
        default=None,
        metavar="ID",
        help="估价用规划器外部 ID（省略则引擎按主规划器自动选；可用环境变量 FLIGHTGRID_EVALUATION_PLANNER）",
    )
    parser.add_argument(
        "--preset",
        default=None,
        metavar="NAME",
        help="算法对预设，如 lmta+ovs、nearest+fast；与 -s/-p 组合时显式参数优先",
    )
    parser.add_argument(
        "--from-scene-algorithms",
        action="store_true",
        help="耦合模式：使用场景 JSON 内 scene.schedulingId 与 scene.algorithmId（忽略 -s/-p/--preset）",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="列出内置预设后退出",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出当前已注册的调度器与规划器外部 ID 后退出",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="开启规划轨迹（更慢、更占内存）",
    )
    parser.add_argument(
        "--lmta-route-backend",
        choices=("thread", "process", "auto"),
        default=None,
        help="仅在使用 LMTA 时有效：覆盖路径预计算后端",
    )
    parser.add_argument(
        "--metrics-out",
        default=None,
        metavar="PATH",
        help="评价指标 txt 输出路径；省略则写入场景同目录：<场景 stem>_algo_metrics.txt",
    )
    parser.add_argument(
        "--no-metrics-file",
        action="store_true",
        help="不写入评价指标文件",
    )
    args = parser.parse_args()

    if args.list_presets:
        _cmd_list_presets()
        return 0
    if args.list:
        return _cmd_list_algorithms()

    try:
        scheduler_ext, planner_ext, use_scene = _resolve_scheduler_planner(args)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2

    path = _resolve_scene_path(root, args.scene_json)
    if path is None:
        print(
            f"错误: 找不到场景文件: {args.scene_json}（已尝试给定路径、仓库根下 data/ 中的同名文件）",
            file=sys.stderr,
        )
        return 2

    from backend.algorithms.discovery import load_algorithms_once
    from backend.algorithm_catalog import resolve_external_to_internal
    from backend.scene_dispatch_runner import run_scene_dispatch_on_json_file, write_dispatch_metrics_txt

    load_algorithms_once()

    extra: dict | None = None
    eval_ext = (args.evaluation_planner or os.environ.get("FLIGHTGRID_EVALUATION_PLANNER", "") or "").strip() or None

    if not use_scene:
        sched_internal = resolve_external_to_internal("scheduler", scheduler_ext)
        if sched_internal == "lmta":
            if args.lmta_route_backend:
                extra = {"schedulerParameters": {"lmta_route_backend": args.lmta_route_backend}}
            elif not os.environ.get("FLIGHTGRID_LMTA_SERIAL_ONLY"):
                extra = {"schedulerParameters": {"lmta_route_backend": "thread"}}

    print(f"场景: {path.resolve()}")
    if use_scene:
        print("算法: 耦合场景 — 将使用 JSON 内 scene.schedulingId + scene.algorithmId")
    else:
        print(f"算法: 解耦 — schedulingId={scheduler_ext}  algorithmId={planner_ext}")
        if args.preset:
            print(f"      （预设: {args.preset}）")
    if eval_ext:
        print(f"估价规划器(-e): {eval_ext}")

    out = run_scene_dispatch_on_json_file(
        path,
        planner_external_id=planner_ext or "ovs",
        scheduler_external_id=scheduler_ext or "lmta",
        use_scene_algorithms=use_scene,
        extra_planning=extra,
        enable_planning_trace=args.trace,
        evaluation_planner_external_id=eval_ext,
    )

    print(
        f"本次请求外部 ID: algorithmId={out['algorithm_id_external']}  schedulingId={out['scheduling_id_external']}"
    )
    print(f"内部注册名: planner={out['planner_name_internal']}  scheduler={out['scheduler_name_internal']}")
    if out.get("evaluation_planner_name_internal"):
        print(f"估价规划器(内部): {out['evaluation_planner_name_internal']}")
    print(f"空间边界 limits: {out['space_limits']}")
    print(f"预处理: {out['preprocess_time_s']:.3f}s, 引擎求解: {out['engine_run_time_s']:.3f}s, 合计: {out['total_elapsed_s']:.3f}s")
    print(f"任务分配: {out['targets_assigned']}/{out['targets_total']}")
    if out["unassigned_target_ids"]:
        nu = len(out["unassigned_target_ids"])
        preview = out["unassigned_target_ids"][:12]
        print(f"未分配任务数: {nu}（示例 id: {preview}{'...' if nu > 12 else ''}）")
    print(f"规划器统计: {out['planner_stats']}")
    pt = out.get("phase_timings") or {}
    if pt:
        print(f"阶段耗时摘要: schedulerTime={pt.get('schedulerTime', 0):.3f}s, pathPlanningTime={pt.get('pathPlanningTime', 0):.3f}s")

    if not args.no_metrics_file:
        metrics_path = Path(args.metrics_out) if args.metrics_out else (path.parent / f"{path.stem}_algo_metrics.txt")
        write_dispatch_metrics_txt(
            metrics_path,
            scene_path=path,
            dispatch_out=out,
            preset_name=args.preset,
            evaluation_planner_external=eval_ext,
        )
        print(f"评价指标已写入: {metrics_path.resolve()}")

    return 0 if not out["unassigned_target_ids"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
