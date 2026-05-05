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
│       │   │   ├── data_collect.py                       # 批量写入 bag 与指标导出
│       │   │   └── plot_utils.py                         # 轨迹可视化
│       │   ├── src/                                      # C++ 辅助节点
│       │   │   ├── bag2pub.cpp                           # ROS1 bag → topic 回放
│       │   │   └── distribution_2d_publisher.cpp         # 二维分布可视化发布
│       │   └── launch/pub_bag.launch                     # ROS1 bag 回放 launch 文件
```

## 2. 环境准备

本项目依赖 ROS2 (jazzy) 自带的系统 Python，无需额外创建虚拟环境。相对关键的 Python 包是 JAX：
```bash
pip install jax[cpu]
```
如有 GPU 可安装对应版本：

```bash
pip install jax[cudaxx]
```

### 2.1 NumPy / Matplotlib 兼容问题

ROS2 运行时通常使用系统 Python，例如 `/usr/bin/python3`。但如果用户目录 `~/.local/lib/python3.12/site-packages` 中安装了新版 NumPy，系统 Python 仍可能优先加载用户目录中的包，导致系统 `matplotlib` 与用户目录 NumPy 发生 ABI 不兼容。

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

如果 NumPy 来自 `~/.local/lib/python3.12/site-packages`，而 Matplotlib 来自 `/usr/lib/python3/dist-packages`，说明当前环境混用了用户 pip 包和系统 apt 包。

不修改现有 NumPy 的临时运行方式：

```bash
PYTHONNOUSERSITE=1 ros2 run multi_ergodic_search idmed
```

该命令只在本次运行中忽略用户目录 `site-packages`，不会卸载或降级任何包。

如果希望继续使用用户目录中的 NumPy 2.x，也可以升级用户目录中的 Matplotlib，使二者来自同一套 pip 环境：

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

通常表示系统版和 pip 版 Matplotlib 的 `mpl_toolkits` 仍有混用。若程序不使用 3D 绘图，该警告一般不影响 2D 轨迹可视化；如需修复，可强制重装 pip 版 Matplotlib：

```bash
/usr/bin/python3 -m pip install --user --force-reinstall "matplotlib>=3.9" --break-system-packages
```

## 3. 实验参数

所有参数集中在：

```text
src/multi_ergodic_search/multi_ergodic_search/datas/config/config.yaml
```

文件中已包含各字段的中文注释，按分组说明：搜索空间、iLQR 优化权重、控制/速度约束、初始/终点位置、安全与通信、优化目标权重、地图更新、输出配置及目标分布定义。修改参数后直接运行即可，无需额外操作。

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

## 5. 输出模式说明

### 5.1 模式概览

核心参数：

| 参数 | 含义 | 可选值 |
| --- | --- | --- |
| `output_mode` | 输出模式 | `none` / `bag` / `topic` / `both` |
| `output_bag_dir` | bag 文件输出目录 | 默认 `./datas/bags/my_strategy` |
| `output_topic` | 话题名称模板，`{robot_id}` 会被替换为机器人编号 | 默认 `/trajectory/robot_{robot_id}/control_sequence` |
| `output_publish_rate` | topic 发布频率 (Hz) | 默认 `20.0` |

四种模式的含义：

- **`none`**：不输出任何数据，仅运行规划算法。
- **`bag`**：将轨迹、路径和地图数据写入 ROS2 bag 文件，供离线回放分析。
- **`topic`**：实时发布控制指令到 ROS2 话题，适合对接飞控或可视化节点。
- **`both`**：同时输出 bag 文件和发布话题。

修改方式：编辑 `config.yaml` 中的 `output_mode` 字段即可切换，详见[第 3 节](#3-实验参数)。

> **为什么需要 bag 回放？** iLQR 在线求解的 wall clock time 较长（笔记本 RTX3060 上单轮最长达 2-3 分钟），规划周期远慢于实际控制周期。因此实际工作流为：先以 `bag` 或 `both` 模式运行规划，将轨迹写入 rosbag，再通过 `ros2 bag play` 按真实时间节奏回放，对接下游飞控或可视化节点。

### 5.2 话题消息格式

`output_mode` 为 `topic` 或 `both` 时，会为每台机器人发布到：

```text
/trajectory/robot_0/control_sequence
/trajectory/robot_1/control_sequence
/trajectory/robot_2/control_sequence
/trajectory/robot_3/control_sequence
```

消息类型为 `trajectory_msgs/msg/MultiDOFJointTrajectoryPoint`，字段对应关系：

```text
transforms[0].translation.x     -> 二维位置 x (m)
transforms[0].translation.y     -> 二维位置 y (m)
velocities[0].linear.x          -> 二维速度 v_x (m/s)
velocities[0].linear.y          -> 二维速度 v_y (m/s)
accelerations[0].linear.x       -> 二维控制输入 u_x (m/s²)
accelerations[0].linear.y       -> 二维控制输入 u_y (m/s²)
```

（`translation.z`、`rotation.w` 为固定占位值，二维规划不使用。）

### 5.3 发布机制

Topic 模式不是一次发布完整全局轨迹，而是随着规划循环逐段发布已确定要执行的控制序列。主循环流程为：

```text
┌─ 求解所有机器人当前段轨迹 (iLQR 优化)
├─ 查找最早通信事件时间点 current_time (机器人间距 < connect_threshold)
├─ 检查地图融合 / 目标更新事件，调整 current_time
├─ emit_segment: 按 publish_rate 频率, 从 step=0 逐条发布到 current_time-1
├─ 裁掉已执行片段 (sol_trajs 只保留 current_time 之后的部分)
├─ 机器人信息交换 (共享彼此轨迹和地图)
└─ 回到循环起点, 重新规划 ─┘
```

其中 `current_time` 由以下事件中最早的一个决定：
- 任意两台机器人的间距小于 `connect_threshold`（触发通信）
- 累计时间达到 `map_merge_freq`（触发地图融合）
- 累计时间达到 `update_map_freq`（触发目标地图更新）

因此 topic 中的数据本质是**在线滚动重规划的控制命令**：参数变化、通信事件和地图更新都会导致重规划，新规划结果会替代之前尚未发布的轨迹段。

### 5.4 Bag 文件输出

`output_mode` 为 `bag` 或 `both` 时，会在 `output_bag_dir` 下生成以下 bag 文件：

| 文件 | 内容 | 消息类型 |
| --- | --- | --- |
| `traj0.bag` ~ `traj3.bag` | 各机器人从自身视角的联合控制序列 | `MultiDOFJointTrajectoryPoint` |
| `traj0-0.bag` ~ `traj3-3.bag` | 每轮规划中机器人 i 视角下的机器人 j 轨迹 | `MultiDOFJointTrajectoryPoint` |
| `path0.bag` ~ `path3.bag` | 各机器人已执行的历史路径 | `nav_msgs/Path` |
| `path0-0.bag` ~ `path3-3.bag` | 每轮规划中机器人 i 视角下的机器人 j 路径 | `nav_msgs/Path` |
| `map_r0.bag` ~ `map_r3.bag` | 各机器人维护的belief概率地图 (热力图 Marker) | `visualization_msgs/Marker` |
| `map.bag` | 全局目标分布热力图 | `visualization_msgs/Marker` |
