#!/usr/bin/env python3
"""
save_full_trajectory_to_rosbag.py
存储完整状态和控制信息到ROSbag
"""
import rosbag
import rospy
from datetime import datetime
import os
import numpy as np
from quadrotor_msgs.msg import PositionCommand
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from scipy.ndimage import gaussian_filter
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler


def hex_to_rgb_float(hex_color):
    """
    将 '#rrggbb' 转为 (r, g, b) 浮点元组，范围 [0.0, 1.0]
    示例: '#ff0000' → (1.0, 0.0, 0.0)
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)

class CommandToRosbag:
    def __init__(self, bag_dir="~/ros_data/trajectories"):
        """初始化轨迹保存器"""
        self.bag_dir = os.path.expanduser(bag_dir)
        os.makedirs(self.bag_dir, exist_ok=True)
        
    def save_traj2bag(self, trajectory_dict, bag_filename=None, 
                            dt=0.1, robot_id=0, mode='w'):
        self.robot_id = robot_id
        x_data = np.array(trajectory_dict['x'])
        u_data = np.array(trajectory_dict['u'])
        # traj_len = len(x_data)
        # rospy.loginfo(f"轨迹点数量: {traj_len}")
        # rospy.loginfo(f"x形状: {x_data.shape}, u形状: {u_data.shape}")
        # 生成文件名
        if bag_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bag_filename = f"trajectory_{timestamp}.bag"
        bag_path = os.path.join(self.bag_dir, bag_filename)
        try:
            if mode == 'a' or os.path.exists(bag_path):
                with rosbag.Bag(bag_path, 'r') as existing_bag:
                    # 找到最大的时间戳
                    time_offset = 0.0
                    for _, _, t in existing_bag.read_messages():
                        time_offset = max(time_offset, t.to_sec())
                    time_offset += dt 
                with rosbag.Bag(bag_path, 'a') as bag:  # 'a'表示追加
                    self._save_as_state_control_sequence(bag, x_data, u_data, dt, time_offset)
            else:
                with rosbag.Bag(bag_path, 'w') as bag:
                    # 保存完整状态和控制序列
                    self._save_as_state_control_sequence(bag, x_data, u_data, dt, time_offset = 0.0)
        except Exception as e:
            rospy.logerr(f"保存ROSbag失败: {e}")
            return None
        return bag_path
    def _save_as_state_control_sequence(self, bag, x_data, u_data, dt, time_offset):
        """
        保存完整状态和控制序列
        状态: [x, y, vx, vy, ...]
        控制: [ux, uy, ...]
        """
        for i in range(len(x_data)):
            timestamp = rospy.Time.from_sec(time_offset + i * dt)
            # timestamp = rospy.Time.from_sec(i * dt)
            # 保存位置 (PoseStamped)
            controlCmd = PositionCommand()
            controlCmd.position.x = x_data[i][0]
            controlCmd.position.y = x_data[i][1]
            controlCmd.velocity.x = x_data[i][2]
            controlCmd.velocity.y = x_data[i][3]
            controlCmd.acceleration.x = u_data[i, 0]
            controlCmd.acceleration.y = u_data[i, 1]
            bag.write(f'robot_{self.robot_id}/trajectory/control_sequence', controlCmd, timestamp)


class PathToRosbag:
    def __init__(self, bag_dir="~/ros_data/paths"):
        self.bag_dir = os.path.abspath(os.path.expanduser(bag_dir))
        os.makedirs(self.bag_dir, exist_ok=True)

    def _get_latest_timestamp(self, bag_path):
        """安全读取 bag 中最大时间戳（秒），失败或文件不存在返回 -1.0"""
        if not os.path.exists(bag_path):
            return -1.0
        try:
            max_time_sec = -1.0
            with rosbag.Bag(bag_path, 'r') as bag:
                for _, _, t in bag.read_messages():
                    max_time_sec = max(max_time_sec, t.to_sec())
            return max_time_sec
        except Exception as e:
            rospy.logwarn(f"读取 bag 时间戳失败（视为新文件）: {e}")
            return -1.0

    def save_path_to_bag(
        self,
        x_traj,
        bag_filename="path.bag",
        dt=0.1,
        robot_id=0,
        frame_id="world"
    ):
        self.robot_id = robot_id
        x_traj = np.asarray(x_traj)
        bag_path = os.path.join(self.bag_dir, bag_filename)
        os.makedirs(self.bag_dir, exist_ok=True)

        # 获取上次最大时间戳
        last_time_sec = self._get_latest_timestamp(bag_path)

        if last_time_sec < 0:
            # 首次写入：使用当前 ROS 时间，若不可用则用 0.0
            time_offset_sec = rospy.Time.from_sec(0.0).to_sec()
            mode = 'w'
        else:
            # 追加：从上次结束 + dt 开始
            time_offset_sec = last_time_sec + dt
            mode = 'a'

        try:
            path_msg = Path()
            path_msg.header.frame_id = frame_id
            path_msg.header.stamp = rospy.Time.from_sec(time_offset_sec)

            poses = []
            for i in range(x_traj.shape[0]):
                pose = PoseStamped()
                pose.header.frame_id = frame_id
                pose_time_sec = time_offset_sec + i * dt
                pose.header.stamp = rospy.Time.from_sec(pose_time_sec)

                pose.pose.position.x = float(x_traj[i, 0])
                pose.pose.position.y = float(x_traj[i, 1])
                pose.pose.position.z = 0.1
                pose.pose.orientation.w = 1.0
                poses.append(pose)

            path_msg.poses = poses

            # 写入 bag —— write_time 必须是 rospy.Time
            write_time = path_msg.header.stamp
            with rosbag.Bag(bag_path, mode) as bag:
                bag.write(f'robot_{self.robot_id}/planned_path', path_msg, write_time)

            # action = "新建" if mode == 'w' else "追加"
            # rospy.loginfo(f"{action}路径 ({x_traj.shape[0]} 点) 到: {bag_path} | 起始时间: {time_offset_sec:.3f}s")
            return bag_path

        except Exception as e:
            rospy.logerr(f"保存路径失败: {e}")
            return None

class MapINfoToMarkers:
    def __init__(self, bag_dir, frame_id="world"):
        self.bag_dir = os.path.expanduser(bag_dir)
        os.makedirs(self.bag_dir, exist_ok=True)
        self.frame_id = frame_id
        self.topic_name = "/map_distribution"
    def generate_probmap_marker(self, points, probs, sigma=0.8, stamp=None, colors_hex = ["#ffffff", "#000000"]):
        if len(probs) != len(points):
            raise ValueError("probs and points length mismatch")

        # 推断网格大小
        x_vals = np.unique(np.round(points[:, 0], decimals=6))
        y_vals = np.unique(np.round(points[:, 1], decimals=6))
        nx, ny = len(x_vals), len(y_vals)

        try:
            prob_grid = probs.reshape((ny, nx))
        except ValueError:
            from scipy.interpolate import griddata
            Xi, Yi = np.meshgrid(
                np.linspace(x_vals.min(), x_vals.max(), nx),
                np.linspace(y_vals.min(), y_vals.max(), ny)
            )
            prob_grid = griddata(points, probs, (Xi, Yi), method='nearest')
            prob_grid = np.nan_to_num(prob_grid)

        prob_grid = gaussian_filter(prob_grid, sigma=sigma)

        # 归一化到 [0, 1]
        p_min, p_max = prob_grid.min(), prob_grid.max()
        norm = (prob_grid - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(prob_grid)

        # 自定义 colormap: white (low) → dark blue (high)
        from matplotlib.colors import LinearSegmentedColormap
        colors = [hex_to_rgb_float(c) for c in colors_hex]
        custom_cmap = LinearSegmentedColormap.from_list("white_to_darkblue", colors)
        rgba = custom_cmap(norm)  # (H, W, 4)
        marker = Marker()
        marker.header.frame_id = self.frame_id
        # 处理时间戳
        if stamp is None:
            stamp = rospy.Time.from_sec(0.0)
        elif isinstance(stamp, (int, float)):
            stamp = rospy.Time.from_sec(float(stamp))
        marker.header.stamp = stamp
        marker.ns = "heatmap"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        for i in range(ny):
            for j in range(nx):
                point = Point(x=x_vals[j], y=y_vals[i], z=0.0)
                color = ColorRGBA()
                color.r, color.g, color.b, color.a = rgba[i, j, 0], rgba[i, j, 1], rgba[i, j, 2], 1.0
                marker.points.append(point)
                marker.colors.append(color)
        return marker
    
    def save_probmap_to_bag(self, points, probs, dt, timesteps, color, bag_filename="map.bag", mode='w'):
        bag_path = os.path.join(self.bag_dir, bag_filename)
        marker = self.generate_probmap_marker(points, probs, stamp=dt, colors_hex = color)
        time_offset = 0.0
        if mode == 'a' or os.path.exists(bag_path):
            with rosbag.Bag(bag_path, 'r') as existing_bag:
                # 找到最大的时间戳
                for _, _, t in existing_bag.read_messages():
                    time_offset = max(time_offset, t.to_sec())
                time_offset += dt 
            with rosbag.Bag(bag_path, 'a') as bag:  # 'a'表示追加
                for i in range(timesteps):
                    marker.header.stamp = rospy.Time.from_sec(time_offset + i * dt)
                    bag.write(self.topic_name, marker, t=marker.header.stamp)
        else:
            with rosbag.Bag(bag_path, 'w') as bag:
                for i in range(timesteps):
                    marker.header.stamp = rospy.Time.from_sec(time_offset + i * dt)
                    bag.write(self.topic_name, marker, t=marker.header.stamp)


