# 离线算法验证：`algo_run.py` 使用说明

在仓库根目录用场景 JSON 调用 `DispatchEngine`，验证「任务分配（调度器）+ 路径规划」组合；无需任何 Web 服务。适用于回归、对比实验与论文复现。

---

## 1. 整体流程

```
场景 JSON
    ↓
algo_run.py（CLI：-s / -p、预设、或 --from-scene-algorithms）
    ↓
scene_dispatch_runner.py（bundle → AlgorithmRequest → 领域模型）
    ↓
plugin_loader（algo_plugins.json）或 AlgorithmRegistry（装饰器注册）
    ↓
DispatchEngine
    ↓
终端摘要 + 可选 <场景stem>_algo_metrics.txt
```

| 组件 | 路径 |
|------|------|
| 入口 | [`algo_run.py`](algo_run.py) |
| 场景与引擎胶水 | [`backend/scene_dispatch_runner.py`](backend/scene_dispatch_runner.py) |
| 引擎 | [`backend/algorithms/core/engine.py`](backend/algorithms/core/engine.py) |
| 外部 ID ↔ 内部名 | [`backend/algorithm_catalog.py`](backend/algorithm_catalog.py) |
| 无装饰器插件清单 | [`backend/algorithms/algo_plugins.json`](backend/algorithms/algo_plugins.json) + [`plugin_loader.py`](backend/algorithms/plugin_loader.py) |
| 自动加载算法模块 | [`backend/algorithms/discovery.py`](backend/algorithms/discovery.py) |

**解析顺序（与线上一致思路）**：对请求里的 `algorithmId` / `schedulingId`，先用 [`resolve_external_to_internal`](backend/algorithm_catalog.py) 得到内部名，再在 `algo_plugins.json` 中按「原始 ID → 内部名」依次查找；未命中则交给注册表按 `DispatchEngine` 传入的名称解析。

---

## 2. 环境与依赖

```bash
pip install -r backend/requirements.txt
```

在**仓库根目录**执行（脚本会把根目录加入 `sys.path`）：

```bash
python3 algo_run.py --help
```

**可选加速**：编译 [`backend/cpp_engine`](backend/cpp_engine) 生成 `libdistance.so`（或设置 `FLIGHTGRID_DISTANCE_LIB`），部分线段与距离矩阵快路径会启用 C++；失败时引擎会回退 Python 逻辑。

**可选算法依赖**（未安装时对应模块可能无法加载或仅在运行到该调度器时报错）：

| 能力 | 典型用途 |
|------|----------|
| **Pyomo** | `milp_alns`（`milp_alns_impl.py` 在 import 阶段需要 pyomo；未安装时 discovery 会打警告并跳过该模块） |
| **OR-Tools** | `ortools_vrp*`、`hybrid_large_scale*` 中依赖 OR-Tools 的路径（包内多为延迟 import，按需安装 `ortools`） |

---

## 3. 场景文件

### 3.1 路径怎么解析

1. 若参数是**已存在的文件路径**（绝对或相对），直接使用该文件。  
2. 否则按**文件名**在仓库根下的 [`data/`](data/) 目录查找同名文件。

未传场景参数时：使用环境变量 `FLIGHTGRID_ALGO_RUN_SCENE`；若未设置，默认为 `data/simple.json`。

### 3.2 JSON 形态

- **`pocas_scene_bundle`**：`packageType == "pocas_scene_bundle"` 且含 `scene` 对象时，使用其中的 `scene` 作为调度输入。  
- 若根上另有 **`scene`** 字典，也会取 `scene`。  
- 否则把整个 JSON 当作场景根对象。

### 3.3 字段约定

需与 [`AlgorithmRequest`](backend/models/algorithm_api.py) 一致，至少包括：`drones`、`orders`、`targets`、`depots`、`executionStates`、`planningConfig`；`no_fly_zones` 等按场景与模型定义选填。

---

## 4. 命令行参数

| 参数 | 说明 |
|------|------|
| `scene_json` | 位置参数；场景 JSON 路径或 `data/` 下文件名。省略时用 `FLIGHTGRID_ALGO_RUN_SCENE` 或默认 `data/simple.json`。 |
| `-s` / `--scheduler` | 调度器外部 ID（如 `lmta`、`nearest_neighbor`）。 |
| `-p` / `--planner` | 路径规划器外部 ID（如 `ovs`、`astar_v2`）。 |
| `-e` / `--evaluation-planner` | 估价阶段规划器外部 ID；也可用环境变量 `FLIGHTGRID_EVALUATION_PLANNER`。 |
| `--preset NAME` | 使用内置算法对；与 `-s`/`-p` 同时出现时，**显式传入的 -s/-p 优先**于预设中的对应项。 |
| `--from-scene-algorithms` | 使用场景 JSON 内的 `schedulingId` + `algorithmId`，忽略 `-s`/`-p`/`--preset`。 |
| `--list` | 打印注册表给出的外部 ID，并附加 `algo_plugins.json` 中的插件 ID。 |
| `--list-presets` | 打印所有 `--preset` 名称与对应的 `(schedulingId, algorithmId)`。 |
| `--trace` | 打开规划轨迹（更慢、更占内存）。 |
| `--lmta-route-backend` | 仅当调度解析为 LMTA 时生效：`thread` \| `process` \| `auto`，写入 `planningConfig.schedulerParameters.lmta_route_backend`。 |
| `--metrics-out PATH` | 指标文本输出路径；默认 `<场景文件同目录>/<stem>_algo_metrics.txt`。 |
| `--no-metrics-file` | 不写指标文件。 |

---

## 5. 环境变量

| 变量 | 含义 |
|------|------|
| `FLIGHTGRID_ALGO_RUN_SCENE` | 默认场景路径（文件路径字符串）。 |
| `FLIGHTGRID_SCHEDULER` | 默认调度外部 ID（当未指定 `-s` 且未用预设覆盖时参与默认解析）。 |
| `FLIGHTGRID_PLANNER` | 默认规划外部 ID。 |
| `FLIGHTGRID_EVALUATION_PLANNER` | 默认估价规划器外部 ID。 |
| `FLIGHTGRID_LMTA_SERIAL_ONLY` | 若设为真值，LMTA 不再自动注入 `lmta_route_backend: "thread"`（适合受限环境避免多线程/进程池）。 |

默认算法对：**lmta + ovs**（与 `-s lmta -p ovs` 或 `--preset lmta+ovs` 一致）。在未使用 `--from-scene-algorithms` 且调度器解析为 **LMTA** 时，若未传 `--lmta-route-backend` 且未设置 `FLIGHTGRID_LMTA_SERIAL_ONLY`，会向 `schedulerParameters` 注入 `lmta_route_backend: "thread"`。

---

## 6. 预设（`--preset`）

内置名称与 `algo_run.py` 中 `ALGO_PRESETS` 一致，例如：`lmta+ovs`、`nearest+astar`、`insertion+fast`、`kmeans+fast` 等。完整列表请执行：

```bash
python3 algo_run.py --list-presets
```

---

## 7. 接入新算法

### 7.1 检查单（装饰器 + catalog）

1. **调度器**：`@AlgorithmRegistry.register_scheduler("internal_id")`。  
2. **规划器**：`@AlgorithmRegistry.register_path_planner("internal_id")`；模块须落在 `discovery` 会扫描的 `path_planning/` 或 `scheduling/` 树内（见 `discovery.py`）。  
3. **短外部名**：在 [`backend/algorithm_catalog.py`](backend/algorithm_catalog.py) 的 `SCHEDULER_EXTERNAL_TO_INTERNAL` / `PLANNER_EXTERNAL_TO_INTERNAL` 与 `CANONICAL_*` 中维护；然后 `python3 algo_run.py --list` 应能看到对应 ID。

### 7.2 仅插件清单（可无装饰器）

1. 在 `backend/algorithms/` 下实现类。  
2. 在 [`algo_plugins.json`](backend/algorithms/algo_plugins.json) 的 `planners` / `schedulers` 中为**每一个**可能传入的键（含 catalog 外部名与内部名）配置 `module` 与 `class`。  
3. 清单命中时，该侧类由 `import` 直接交给引擎；**估价规划器**若未指定 `-e`，引擎仍可能使用注册表中的快速规划器，因此 `load_algorithms_once()` 仍会加载内置模块。

`algo_plugins.json` 的维护说明见文件内 `_meta`。

---

## 8. 输出与退出码

- **终端**：外部/内部算法名、空间 `limits`、预处理与引擎耗时、分配率、未分配任务示例、`planner_stats`、可选 `phase_timings`。  
- **指标文件**：UTF-8 文本；可用 `--no-metrics-file` 关闭。仓库 `.gitignore` 已忽略 `**/*_algo_metrics.txt`，避免误提交生成物。

| 退出码 | 含义 |
|--------|------|
| `0` | 全部目标已分配。 |
| `1` | 存在未分配任务（便于批跑脚本区分）。 |
| `2` | 参数/场景错误（如未知预设、找不到场景文件）。 |

---

## 9. 常用示例

```bash
python3 algo_run.py
python3 algo_run.py -s nearest_neighbor -p astar_v2 data/simple.json
python3 algo_run.py --preset lmta+ovs data/50-120.json
python3 algo_run.py --from-scene-algorithms data/simple.json
python3 algo_run.py -s lmta -p ovs --no-metrics-file
```

更多仓库级说明见根目录 [`README.md`](README.md)。
