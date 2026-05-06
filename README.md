# IDMEC

本仓库用于多机器人 Decay-Ergodic Search 轨迹规划、ROS bag 回放可视化和实机实验对接。请先完整检查离线轨迹和 bag 可视化结果，再接入真实无人机。

## 1. 仓库结构

```text
idmec/
├── environment.yml                                       # (legacy) Conda 环境导出文件
├── src/
│   └── multi_ergodic_search/                             # ROS2 (jazzy) 主包
│       ├── setup.py                                      # 包安装入口, 注册 idmed 控制台脚本
│       ├── package.xml                                   # ROS2 包元信息与依赖声明
│       ├── multi_ergodic_search/
│       │   ├── datas/config/config.yaml                  # 主要实验参数
│       │   ├── datas/config/random_map_history.jsonl     # 动态目标地图历史记录 (JSON Lines)
│       │   ├── scripts/idmed.py                          # 实验主入口 (离线规划 + topic/bag 输出)
│       │   ├── experiment/                               # 实验上下文、配置加载、IO 输出模块
│       │   │   ├── settings.py                           # config.yaml 读取
│       │   │   ├── config.py                             # 参数解析与初始化 (动力学、轨迹、分布)
│       │   │   └── io.py                                 # ExperimentOutput: bag 写入 & topic 发布
│       │   ├── ergodic_planning/                         # ergodic search 核心算法
│       │   │   ├── multi_solver.py                       # 增广拉格朗日 iLQR 多机器人求解器
│       │   │   ├── solver.py                             # iLQR 模板基类
│       │   │   ├── augument_lagrange_func.py             # 增广拉格朗日目标函数与约束
│       │   │   ├── target_distribution.py                # 目标分布与贝叶斯地图更新
│       │   │   ├── ergodic_metric.py                     # ergodic metric 计算
│       │   │   ├── fourier_utils.py                      # 傅里叶基函数
│       │   │   ├── tools.py                              # 通信检测、信息交换、轨迹优化
│       │   │   ├── decay_utils.py                        # beta 衰减系数更新
│       │   │   ├── metric_utils.py                       # 度量工具函数
│       │   │   ├── class_types.py                        # NamedTuple 类型定义
│       │   │   └── dynamics/                             # 动力学模型
│       │   │       ├── models.py                         # DoubleIntegrator, HomoDynamics
│       │   │       └── integrator.py                     # RK4 数值积分
│       │   ├── utils/                                    # 工具模块
│       │   │   ├── ros_messages.py                       # ROS2 消息构造 (MultiDOFJointTrajectoryPoint)
│       │   │   ├── data2bag.py                           # ROS2 bag 写入器 (轨迹/路径/地图热力图)
│       │   │   ├── data_collect.py                       # 地图 JSONL 读写与批量 bag 输出辅助函数
│       │   │   └── plot_utils.py                         # 轨迹可视化
│       │   ├── src/                                      # C++ 辅助节点
│       │   │   ├── bag2pub.cpp                           # ROS1 bag → topic 回放
│       │   │   └── distribution_2d_publisher.cpp         # 二维分布可视化发布
│       │   └── launch/pub_bag.launch                     # ROS1 bag 回放 launch 文件
```

## 2. 环境准备

本项目依赖 ROS2 (jazzy) / Ubuntu24 系统 Python，无需额外创建虚拟环境。相对关键的 Python 包是 JAX：
```bash
pip install jax[cpu]
```
如有 GPU 可安装对应版本：

```bash
pip install jax[cudaxx]
```

### 2.1 NumPy / Matplotlib 兼容问题

ROS2 运行时使用系统 Python，例如 `/usr/bin/python3`，可能导致用户目录 `~/.local/lib/python3.12/site-packages` 中安装的新版 NumPy，与 `matplotlib` 包发生 ABI 不兼容。

典型报错如下：

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
ImportError: numpy.core.multiarray failed to import
```

可先检查当前系统 Python 实际加载的 NumPy 和 Matplotlib：

```bash
/usr/bin/python3 -c "import numpy; print(numpy.__version__, numpy.__file__)"
/usr/bin/python3 -c "import matplotlib; print(matplotlib.__version__, matplotlib.__file__)"
```

我采取的方案：继续使用用户目录中的 NumPy 2.x，同时升级用户目录中的 Matplotlib，使二者来自同一套 pip 环境：

```bash
/usr/bin/python3 -m pip install --user --upgrade "matplotlib>=3.9" --break-system-packages
```

安装后再次确认 Matplotlib 路径：

```bash
/usr/bin/python3 -c "import matplotlib; print(matplotlib.__version__, matplotlib.__file__)"
```

如果出现以下警告：

```text
UserWarning: Unable to import Axes3D. This may be due to multiple versions of Matplotlib being installed
```

通常表示系统版和 pip 版 Matplotlib 的 `mpl_toolkits` 仍有混用。
但程序不使用 3D 绘图，该警告一般不影响 2D 轨迹可视化；


## 3. 实验参数

所有参数集中在：

```text
src/multi_ergodic_search/multi_ergodic_search/datas/config/config.yaml
```

文件中已包含各字段的中文注释。修改参数后直接运行即可，无需额外操作。

> 需要注意真机实验与仿真中机器人半径(膨胀半径)的差异，避免在仿真中使用过小的半径，导致真机实验中机器人碰撞。

> 同时中间部分规划可能存在约束不满足，这将导致避障失效，因此真机实验时需要提前检查效果。

### 3.1 动态目标地图与历史文件

目标分布初始值由 `config.yaml` 中的 `mapinfo_point` 定义：

```text
mapinfo_point.means  # 高斯目标中心点
mapinfo_point.covs   # 每个目标的协方差矩阵
```

运行过程中，全局目标地图会按 `update_map_freq` 周期更新。地图更新逻辑在：

```text
src/multi_ergodic_search/multi_ergodic_search/ergodic_planning/target_distribution.py
```

动态地图历史文件为：

```text
src/multi_ergodic_search/multi_ergodic_search/datas/config/random_map_history.jsonl
```

其中：每一行都是一条完整的 JSON 记录，对应一次目标地图更新后的参数。例如一行包含：

```json
{"means": [[1.2, 2.6], [0.7, 1.1]], "covs": [[[0.08, 0.0], [0.0, 0.1]], [[0.08, 0.0], [0.0, 0.1]]]}
```

其中：

| 字段 | 含义 |
| --- | --- |
| `means` | 当前阶段所有目标高斯中心点 |
| `covs` | 当前阶段所有目标高斯协方差矩阵 |

代码支持两种地图更新方式：

| 模式 | 调用方式 | 行为 | 适用场景 |
| --- | --- | --- | --- |
| 读取历史地图 | `update_map(..., w_or_r="read")` | 从 `random_map_history.jsonl` 按行读取下一条地图 | 复现实验、对比算法 |
| 写入随机地图 | `update_map(..., w_or_r="write")` | 随机生成新地图，并追加写入 `random_map_history.jsonl` | 生成新的动态地图序列 |

当前主入口 [idmed.py](/home/ubuntu24/idmec/src/multi_ergodic_search/multi_ergodic_search/scripts/idmed.py) 默认使用读取模式：

```python
context.target_distr.update_map(context.accumulated_time, "reset", "read")
```

如果需要生成新的随机地图历史，可以临时改为：

```python
context.target_distr.update_map(context.accumulated_time, "reset", "write")
```

`mode` 参数控制新地图如何产生：

| `mode` | 含义 |
| --- | --- |
| `"reset"` | 在地图边界内重新随机采样目标中心 |
| `"perturb"` | 在当前目标中心附近添加随机扰动 |

读取模式会使用 `random_map_history.jsonl` 中已有记录，因此当 `period_num` 或地图更新次数增加时，需要保证该文件中有足够的行数。写入模式会向同一个文件追加记录；如果要生成一组全新的可复现实验地图，建议先备份或清空旧的 `random_map_history.jsonl`。

## 4. 构建与运行

构建：

```bash
cd ~/idmec
source /opt/ros/jazzy/setup.bash
colcon build --packages-select multi_ergodic_search
source install/setup.bash
```
运行：

```bash
ros2 run multi_ergodic_search idmed
```
运行这个节点后，控制台没有输出，我把对应的输出写到 solver log 中。

### 4.1 运行产物位置

通过 `ros2 run multi_ergodic_search idmed` 运行时，程序执行的是 `install/` 下的已安装 Python 包。因此 log、bag 和最终轨迹图片都会写入 `install` 目录中的包路径。

默认位置如下：

| 产物 | 默认路径 |
| --- | --- |
| solver log | `~/idmec/install/multi_ergodic_search/lib/python3.12/site-packages/multi_ergodic_search/datas/logs/app_YYYY-MM-DD_HH-MM.log` |
| ROS2 bag | `~/idmec/install/multi_ergodic_search/lib/python3.12/site-packages/multi_ergodic_search/datas/bags/my_strategy/` |
| 最终轨迹图片 | `~/idmec/install/multi_ergodic_search/lib/python3.12/site-packages/multi_ergodic_search/datas/results/my_strategy/figures/my_strategy.png` |


## 5. 输出模式说明

### 5.1 模式概览

核心参数：

| 参数 | 含义 | 可选值 |
| --- | --- | --- |
| `output_mode` | 输出模式 | `none` / `bag` / `topic` / `both` |
| `output_bag_dir` | bag 文件输出目录，相对于已安装的 `multi_ergodic_search` 包根目录 | 默认 `./datas/bags/my_strategy` |
| `output_topic` | 话题名称模板，`{robot_id}` 会被替换为机器人编号 | 默认 `/trajectory/robot_{robot_id}/control_sequence` |

Topic 发布周期与仿真步长 `dt` 绑定：每隔 `dt` 秒发布一个控制点。因此默认 `dt=0.05` 时，发布频率等价于 `20 Hz`。

四种模式的含义：

- **`none`**：不输出任何数据，仅运行规划算法。
- **`bag`**：将轨迹、路径和地图数据写入 ROS2 bag 文件，供离线回放分析。
- **`topic`**：实时发布控制指令到 ROS2 话题，适合对接飞控或可视化节点。
- **`both`**：同时输出 bag 文件和发布话题。

修改方式：编辑 `config.yaml` 中的 `output_mode` 字段即可切换，详见[第 3 节](#3-实验参数)。

> **为什么需要 bag 回放？** iLQR 在线求解的 wall clock time 较长（笔记本 RTX3060 上单轮最长达 2-3 分钟），规划周期远慢于实际控制周期。因此实际工作流为：先以 `bag` 或 `both` 模式运行规划，将轨迹写入 rosbag，再通过 `ros2 bag play` 按真实时间节奏回放，对接下游飞控或可视化节点。

### 5.2 Bag 文件输出

`output_mode` 为 `bag` 或 `both` 时，会在 `output_bag_dir` 下生成以下 bag 文件：

| 文件 | 内容 | 消息类型 |
| --- | --- | --- |
| `true_robot` | 真实/全局视角：全局目标地图，以及每台机器人自己的真实轨迹和路径 | `MultiDOFJointTrajectoryPoint`, `nav_msgs/Path`, `visualization_msgs/Marker` |
| `robot0` ~ `robot3` | 各机器人自身信念视角：该机器人维护的 belief 地图，以及它视角下所有机器人的轨迹和路径 | `MultiDOFJointTrajectoryPoint`, `nav_msgs/Path`, `visualization_msgs/Marker` |

> 录制 bag 时控制台会输出大量 `INFO`，这是 ROS2 bag writer 的正常现象。

每个 bag 内部按 topic 区分机器人与数据类型。`true_robot`、`robot0` ~ `robot3` 这些 bag 内部都使用以下 topic 结构：

```text
/map_distribution
/robot_0/trajectory/control_sequence
/robot_1/trajectory/control_sequence
/robot_2/trajectory/control_sequence
/robot_3/trajectory/control_sequence
/robot_0/planned_path
/robot_1/planned_path
/robot_2/planned_path
/robot_3/planned_path
```

对应消息类型：

| Topic | 消息类型 | 内容 |
| --- | --- | --- |
| `/map_distribution` | `visualization_msgs/msg/Marker` | 当前 bag 对应视角的概率地图热力图，Marker 类型为 `POINTS` |
| `/robot_{id}/trajectory/control_sequence` | `trajectory_msgs/msg/MultiDOFJointTrajectoryPoint` | 机器人 `{id}` 的轨迹控制点，含位置、速度、控制输入 |
| `/robot_{id}/planned_path` | `nav_msgs/msg/Path` | 机器人 `{id}` 此次求解规划出的未来轨迹 |

`/robot_{id}/trajectory/control_sequence` 的字段对应关系：

```text
transforms[0].translation.x     -> 二维位置 x (m)
transforms[0].translation.y     -> 二维位置 y (m)
velocities[0].linear.x          -> 二维速度 v_x (m/s)
velocities[0].linear.y          -> 二维速度 v_y (m/s)
accelerations[0].linear.x       -> 二维控制输入 u_x (m/s²)
accelerations[0].linear.y       -> 二维控制输入 u_y (m/s²)
time_from_start                 -> bag 中该点相对当前段开始的时间
```

（`translation.z`、`rotation.w` 为固定占位值，二维规划不使用。）

Bag 输出会在一次 `emit_segment` 中写入当前段的多个控制点，但每条 message 仍然是单个控制点。rosbag 使用 `dt` 写入消息时间戳，因此 bag 时间步长与仿真步长一致。

### 5.3 Topic 发布方式

`output_mode` 为 `topic` 或 `both` 时，会为每台机器人实时发布到：

```text
/trajectory/robot_0/control_sequence
/trajectory/robot_1/control_sequence
/trajectory/robot_2/control_sequence
/trajectory/robot_3/control_sequence
```

实时 topic 每次发布一条 `trajectory_msgs/msg/MultiDOFJointTrajectoryPoint`，也就是一个控制点，而不是完整轨迹数组。字段含义与 bag 中的 `/robot_{id}/trajectory/control_sequence` 一致，但实时 topic 主要依靠 `dt` 控制发布时间间隔。

Topic 模式不是一次发布完整全局轨迹，而是随着规划循环逐段发布已确定要执行的控制序列。每个发布周期只发一个控制点。

主循环流程为：

```text
┌─ 求解所有机器人当前段轨迹 (iLQR 优化)
├─ 查找最早通信事件时间点 current_time (机器人间距 < connect_threshold)
├─ 检查地图融合 / 目标更新事件，调整 current_time
├─ emit_segment: 按 dt 周期, 从当前时刻开始计算， step=0 逐条发布到 current_time-1
├─ 机器人信息交换 (共享彼此轨迹和地图)
└─ 回到循环起点, 重新规划 ─┘
```

其中 `current_time` 由以下事件中最早的一个决定：
- 任意两台机器人的间距小于 `connect_threshold`（触发通信）
- 累计时间达到 `map_merge_freq`（触发地图融合）
- 累计时间达到 `update_map_freq`（触发目标地图更新）

因此 topic 中的数据本质是**在线滚动重规划的控制命令**：参数变化、通信事件和地图更新都会导致重规划，新规划结果会替代之前尚未发布的轨迹段。实时 topic 按 `dt` 间隔逐点发布，因此发布周期与仿真步长一致。
