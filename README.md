# IDMEC

本仓库用于多机器人 Decay-Ergodic Search 轨迹规划、ROS bag 回放可视化和实机实验对接。请先完整检查离线轨迹和 bag 可视化结果，再接入真实无人机。

## 1. 仓库结构

```text
idmec/
├── environment.yml                         # Python/JAX/绘图环境
├── src/
│   ├── ergodic_search/                     # 多机器人 ergodic search 规划代码
│   │   ├── datas/config/config.yaml        # 主要实验参数
│   │   ├── multiRobots_lib/                # 优化器、动力学、地图、bag 工具
│   │   ├── scripts/idmed.py                # 当前推荐的模块化实验入口
│   │   ├── launch/pub_bag.launch           # 将轨迹 bag 发布成控制话题
│   │   └── figures/                        # 轨迹图输出目录
│   ├── fly_order/                          # 接收轨迹命令并转发给可视化/飞控接口
│   │   ├── launch/simple_run.launch        # 四机 ROS 可视化入口
│   │   ├── launch/swarm.launch             # 四机启动配置
│   │   └── src/offboard.cpp                # PositionCommand 转发节点
│   └── uav_simulator/                      # 四旋翼模型、地图、控制器等 ROS 包
```

## 2. 环境准备

Conda 环境已经导出到仓库根目录的 `environment.yml`。

创建环境：

```bash
cd /home/cyc/idmec
conda env create -f environment.yml
conda activate uavPlan
```

如果环境已经存在：

```bash
conda activate uavPlan
conda env update -f environment.yml --prune
```

## 3. 修改实验参数

主要参数在：

```text
src/ergodic_search/datas/config/config.yaml
```

常用字段：

| 字段 | 含义 |
| --- | --- |
| `workspace` | 搜索区域边长，当前默认 4.0 |
| `dt` | 规划时间步长 |
| `tsteps` | 总规划步数 |
| `robot_number` | 机器人数量，当前默认 4 |
| `x0` | 所有机器人初始二维位置，格式 `[x0,y0,x1,y1,...]` |
| `xf` | 所有机器人终点二维位置 |
| `U_max/U_min` | 控制输入上/下限 |
| `V_max/V_min` | 速度上/下限 |
| `avoidance_radius` | 避障安全半径 |
| `com_radius` | 通信半径相关参数 |
| `connect_threshold` | 判定机器人相遇/可交换信息的距离阈值 |
| `update_map_freq` | 目标地图更新周期 |
| `map_merge_freq` | 机器人地图融合周期 |
| `mapinfo_point` | 目标分布的高斯点和协方差 |

注意：`idmed.py` 依赖相对路径，运行时请进入 `src/ergodic_search/scripts` 目录。

首次运行前创建日志目录：

```bash
mkdir -p /home/cyc/idmec/src/ergodic_search/scripts/datas/logs
```

## 4. 离线规划

当前推荐入口是模块化后的：

```bash
cd /home/cyc/idmec/src/ergodic_search/scripts
python idmed.py
```

运行结束后会输出轨迹图：

```text
src/ergodic_search/figures/my_strategy.png
```

日志默认写入：

```text
src/ergodic_search/scripts/datas/logs/app.log
```

## 5. 回放轨迹 bag

如果已经生成了 `traj0.bag` 到 `traj3.bag`，可以用 `pub_bag.launch` 发布控制序列。

回放：

```bash
cd /home/cyc/idmec
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch ergodic_search pub_bag.launch bag_dir:=/home/cyc/idmec/src/ergodic_search/scripts publish_rate:=20.0 frame_id:=world
```

该 launch 会读取每个机器人的 bag 文件，并发布到：

```text
/trajectory/robot_0/control_sequence
/trajectory/robot_1/control_sequence
/trajectory/robot_2/control_sequence
/trajectory/robot_3/control_sequence
```

如果需要接入 `fly_order/offboard`，请确认话题名和 `src/fly_order/launch/run_in_sim.launch` 中的 remap 一致。

## 5.1 ROS2 Topic 模式

当前 ROS2 版本可以通过配置直接发布轨迹控制话题。相关参数在：

```text
src/multi_ergodic_search/multi_ergodic_search/datas/config/config.yaml
```

典型配置：

```yaml
output_mode: topic  # none | bag | topic | both
output_topic: /trajectory/robot_{robot_id}/control_sequence
output_publish_rate: 20.0
```

`output_mode: topic` 会发布：

```text
/trajectory/robot_0/control_sequence
/trajectory/robot_1/control_sequence
/trajectory/robot_2/control_sequence
/trajectory/robot_3/control_sequence
```

消息类型为 ROS2 官方消息：

```text
trajectory_msgs/msg/MultiDOFJointTrajectoryPoint
```

字段含义：

```text
transforms[0].translation.x/y  -> 二维位置 x, y
velocities[0].linear.x/y       -> 二维速度 vx, vy
accelerations[0].linear.x/y    -> 二维控制输入 ux, uy
```

Topic 模式不是一次发布完整全局轨迹，而是发布当前规划周期内实际要执行的事件片段。每轮流程可以理解为：

```text
规划一段轨迹 -> 发布 step=0 到 current_time-1 -> 裁掉已执行片段 -> 地图/通信更新 -> 重新规划下一段
```

其中 `current_time` 由当前触发事件决定，例如机器人通信、地图更新或地图融合。因此 topic 中的数据更接近在线滚动重规划的控制命令，而不是离线完整轨迹文件。

ROS2 下运行：

```bash
cd ~/idmec
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run multi_ergodic_search idmed
```

另开终端查看：

```bash
source /opt/ros/jazzy/setup.bash
source ~/idmec/install/setup.bash
ros2 node list
ros2 topic list
ros2 topic info /trajectory/robot_0/control_sequence
ros2 topic echo /trajectory/robot_0/control_sequence
```

## 6. 启动 ROS bag 可视化功能包

另开一个终端：

```bash
cd /home/cyc/idmec
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch fly_order simple_run.launch
```

这个 launch 用于启动 ROS 可视化和命令转发相关节点：

- RViz
- `fly_order/offboard` 节点
- 地图/无人机状态可视化相关节点

`fly_order/offboard` 订阅每个机器人的轨迹命令，并发布给对应的可视化/控制接口。话题映射在：

```text
src/fly_order/launch/run_in_sim.launch
```

`fly_order/offboard` 侧使用的关键话题：

```text
/robot_0/trajectory/control_sequence
/robot_1/trajectory/control_sequence
/robot_2/trajectory/control_sequence
/robot_3/trajectory/control_sequence

/drone_0_planning/pos_cmd
/drone_1_planning/pos_cmd
/drone_2_planning/pos_cmd
/drone_3_planning/pos_cmd
```

## 7. 实机实验流程建议

实机前请按顺序完成：

1. 离线运行 `idmed.py`，确认轨迹图合理。
2. 用 `pub_bag.launch` 回放轨迹 bag，并用 `simple_run.launch` 启动 ROS 可视化，确认四机轨迹、速度和加速度没有异常。
3. 检查 `config.yaml` 中的 `x0/xf`、速度限制、控制限制、安全半径是否与真实场地一致。
4. 确认真实定位话题与 `run_in_sim.launch` 中的 `odom_topic` 对应。
5. 实机只先测试单机或低速短轨迹，再逐步切换到四机。

实机相关注意事项：

- 当前规划是二维轨迹，飞行高度由 `fly_order` 的 `fly_height` 参数固定。
- `offboard.cpp` 会将输入 `PositionCommand` 的 `x/y` 和速度/加速度转发出去，`z` 使用 `fly_height`。
- 起飞点必须与 `config.yaml` 的 `x0` 以及 launch 中的 `init_x/init_y/init_z` 保持一致。
- 实机前务必确认急停、遥控接管、保护网/安全区域和电池状态。
- 不要直接在实机上第一次运行新参数；先检查 bag 回放可视化，再小范围低速测试。
