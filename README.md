# FlightGrid_for_algo

FlightGrid 的离线算法验证仓库，用于在本地直接运行「任务分配（调度）」+「路径规划」组合，不依赖 Web 服务。  
适用于算法回归、组合对比、参数调优与论文复现。

## 项目定位

- 这是一个 **算法内核与离线 runner** 仓库，不是完整线上系统。
- 入口脚本是 `algo_run.py`，核心引擎是 `backend/algorithms/core/engine.py`。
- 主要输入是场景 JSON，输出为终端摘要与可选指标文本文件（`*_algo_metrics.txt`）。

## 主要能力

- 支持调度器与路径规划器解耦组合（`-s` / `-p`）。
- 支持按场景内置算法对运行（`--from-scene-algorithms`）。
- 支持预设算法组合（`--preset`）。
- 支持插件清单方式加载算法（`backend/algorithms/algo_plugins.json`），不强制依赖装饰器注册。
- 支持输出分配率、耗时、规划器统计与阶段耗时。
- 支持可选 C++ 距离库加速（`backend/cpp_engine`）。

## 代码结构

```text
.
├── algo_run.py                      # 离线入口 CLI
├── ALGO_RUN.md                      # algo_run 详细说明（中文）
├── data/                            # 示例场景与样例输出
└── backend/
    ├── scene_dispatch_runner.py     # 场景 JSON -> AlgorithmRequest -> DispatchEngine
    ├── algorithm_catalog.py         # 外部算法 ID 与内部实现名映射
    ├── algorithms/
    │   ├── core/engine.py           # 调度与路径规划主引擎
    │   ├── discovery.py             # 自动发现并导入算法模块
    │   ├── registry.py              # AlgorithmRegistry
    │   ├── plugin_loader.py         # algo_plugins.json 插件加载
    │   ├── scheduling/              # 调度算法
    │   └── path_planning/           # 路径规划算法
    ├── models/algorithm_api.py      # Pydantic 协议模型 + 转换逻辑
    ├── config/                      # 环境与算法配置
    └── cpp_engine/                  # 可选 C++ 距离计算库
```

## 环境要求

- Python 3.10+（建议）
- pip
- Linux/macOS（Windows 需自行适配 C++ 共享库构建与加载）

## 安装依赖

先安装基础依赖：

```bash
pip install -r backend/requirements.txt
```

`backend/requirements.txt` 当前包含最小依赖：

- `numpy`
- `pydantic`
- `PyYAML`

### 可选依赖

部分算法是按需依赖，未安装时对应模块可能无法导入或运行时报错：

- `ortools`：OR-Tools VRP 系列与部分大规模混合调度算法
- `pyomo`：`milp_alns` 相关实现

可按需安装：

```bash
pip install ortools pyomo
```

## 快速开始

在仓库根目录执行：

```bash
python3 algo_run.py --help
```

常用示例：

```bash
# 1) 使用默认场景与默认算法对（lmta + ovs）
python3 algo_run.py

# 2) 显式指定算法组合
python3 algo_run.py -s nearest_neighbor -p astar_v2 data/simple.json

# 3) 使用预设
python3 algo_run.py --preset lmta+ovs data/50-120.json

# 4) 使用场景内配置的 schedulingId + algorithmId
python3 algo_run.py --from-scene-algorithms data/simple.json

# 5) 不生成指标文件
python3 algo_run.py --no-metrics-file
```

## 运行流程

1. `algo_run.py` 解析 CLI 参数、场景路径与算法来源（解耦/耦合/预设）。
2. `scene_dispatch_runner.py` 读取场景 JSON 并构建 `AlgorithmRequest`。
3. `algorithm_catalog.py` 将外部 ID 解析为内部实现名。
4. `plugin_loader.py` 优先按 `algo_plugins.json` 加载插件类，未命中时回退注册表。
5. `DispatchEngine` 执行调度与路径规划，返回航迹与统计。
6. 终端打印摘要，并可写入 `<scene_stem>_algo_metrics.txt`。

## 场景文件说明

- 如果参数是有效文件路径，直接使用。
- 否则按文件名在仓库 `data/` 目录查找。
- 未传场景参数时，读取环境变量 `FLIGHTGRID_ALGO_RUN_SCENE`；若未设置则使用 `data/simple.json`。

支持 JSON 形态：

- 根对象为 `pocas_scene_bundle` 且包含 `scene`：使用 `scene`。
- 根对象直接包含 `scene`：使用 `scene`。
- 否则：将根对象视为场景。

核心字段需与 `backend/models/algorithm_api.py` 中 `AlgorithmRequest` 保持一致，常见包括：

- `drones`
- `orders`
- `targets`
- `depots`
- `executionStates`
- `planningConfig`
- `no_fly_zones`（可选）

## 常用环境变量

- `FLIGHTGRID_ALGO_RUN_SCENE`：默认场景路径
- `FLIGHTGRID_SCHEDULER`：默认调度器外部 ID
- `FLIGHTGRID_PLANNER`：默认规划器外部 ID
- `FLIGHTGRID_EVALUATION_PLANNER`：默认估价规划器外部 ID
- `FLIGHTGRID_LMTA_SERIAL_ONLY`：限制 LMTA 自动并行后端注入行为

此外，`backend/config/settings.py` 中还定义了 `FLIGHTGRID_DISPATCH_*` 系列参数用于运行策略控制。

## 输出与退出码

- 终端输出：算法映射、边界、耗时、分配率、未分配任务示例、规划器统计、阶段耗时
- 指标文件：默认输出到场景同目录，文件名为 `<场景stem>_algo_metrics.txt`

退出码约定：

- `0`：全部任务已分配
- `1`：存在未分配任务
- `2`：参数或场景错误

## 可选 C++ 加速

目录：`backend/cpp_engine`

示例构建流程（Linux）：

```bash
cd backend/cpp_engine
cmake -S . -B build
cmake --build build -j
```

生成 `libdistance.so` 后可通过环境变量 `FLIGHTGRID_DISTANCE_LIB` 指向该库路径。  
若未配置或加载失败，系统会回退到 Python 实现。

## 新增算法接入

可选两种方式：

1. **注册表方式（推荐）**
   - 使用 `AlgorithmRegistry.register_scheduler(...)` 或 `register_path_planner(...)`
   - 确保模块位于 `backend/algorithms/scheduling` 或 `backend/algorithms/path_planning`（被 `discovery.py` 扫描）
   - 在 `backend/algorithm_catalog.py` 维护外部 ID 映射

2. **插件清单方式**
   - 在 `backend/algorithms/algo_plugins.json` 中配置 `module` 与 `class`
   - runner 会优先通过清单加载

新增后可执行：

```bash
python3 algo_run.py --list
python3 algo_run.py --list-presets
```

## 当前仓库边界与说明

- 本仓库未包含完整线上 API 服务启动入口。
- 当前未提供标准化测试目录与 CI 配置文件（如 `tests/`、`pytest.ini`、GitHub Actions）。
- `backend/models/algorithm_api.py` 注释提及 `algorithmApi_json.md`，该文件目前不在本仓库中。

## 相关文档

- `ALGO_RUN.md`：离线运行参数与行为细节（更完整）

