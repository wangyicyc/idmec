#!/usr/bin/env python3
"""ROS2 bag writers for trajectory, path, and map messages."""

from pathlib import Path

import numpy as np
import rosbag2_py
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path as PathMsg
from rclpy.serialization import serialize_message
from scipy.ndimage import gaussian_filter
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from utils.ros_messages import (
    build_control_command,
    seconds_to_nanoseconds,
    seconds_to_time_msg,
)


def message_type_name(message):
    message_class = message.__class__
    package_name = message_class.__module__.split(".")[0]
    return f"{package_name}/msg/{message_class.__name__}"


def normalize_bag_uri(bag_dir, bag_filename):
    bag_name = Path(bag_filename).stem
    return str(Path(bag_dir).expanduser().resolve() / bag_name)


def hex_to_rgb_float(hex_color):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def lerp_color(low_color, high_color, value):
    return tuple(low + (high - low) * value for low, high in zip(low_color, high_color))


class Rosbag2WriterPool:
    def __init__(self, bag_dir):
        self.bag_dir = Path(bag_dir).expanduser().resolve()
        self.bag_dir.mkdir(parents=True, exist_ok=True)
        self.writers = {}
        self.created_topics = {}
        self.next_time = {}
        self.bag_uris = {}

    def write(self, bag_filename, topic_name, message, timestamp_sec):
        bag_uri = self.bag_uri_for(bag_filename)
        writer = self._get_writer(bag_uri)
        topic_key = (bag_uri, topic_name)
        if topic_key not in self.created_topics:
            writer.create_topic(
                rosbag2_py.TopicMetadata(
                    id=0,
                    name=topic_name,
                    type=message_type_name(message),
                    serialization_format="cdr",
                )
            )
            self.created_topics[topic_key] = True

        timestamp_ns = seconds_to_nanoseconds(timestamp_sec)
        writer.write(topic_name, serialize_message(message), timestamp_ns)
        self.next_time[bag_uri] = max(
            self.next_time.get(bag_uri, 0.0),
            float(timestamp_sec),
        )
        return bag_uri

    def start_time(self, bag_filename, dt):
        bag_uri = self.bag_uri_for(bag_filename)
        if bag_uri not in self.next_time:
            return 0.0
        return self.next_time[bag_uri] + dt

    def bag_uri_for(self, bag_filename):
        if bag_filename in self.bag_uris:
            return self.bag_uris[bag_filename]

        base_uri = Path(normalize_bag_uri(self.bag_dir, bag_filename))
        bag_uri = base_uri
        suffix = 1
        while bag_uri.exists():
            bag_uri = base_uri.with_name(f"{base_uri.name}_{suffix}")
            suffix += 1
        self.bag_uris[bag_filename] = str(bag_uri)
        return self.bag_uris[bag_filename]

    def _get_writer(self, bag_uri):
        if bag_uri in self.writers:
            return self.writers[bag_uri]

        writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        writer.open(storage_options, converter_options)
        self.writers[bag_uri] = writer
        return writer


class CommandToRosbag:
    def __init__(self, bag_dir="~/ros_data/trajectories"):
        self.writer_pool = Rosbag2WriterPool(bag_dir)

    def save_traj2bag(
        self,
        trajectory_dict,
        bag_filename=None,
        dt=0.1,
        robot_id=0,
        mode="w",
    ):
        x_data = np.asarray(trajectory_dict["x"])
        u_data = np.asarray(trajectory_dict["u"])
        bag_filename = bag_filename or f"trajectory_robot_{robot_id}.bag"
        time_offset = 0.0 if mode == "w" else self.writer_pool.start_time(bag_filename, dt)

        for i in range(len(x_data)):
            command = build_control_command(x_data[i], u_data[i], time_from_start=i * dt)
            self.writer_pool.write(
                bag_filename,
                f"/robot_{robot_id}/trajectory/control_sequence",
                command,
                time_offset + i * dt,
            )
        return self.writer_pool.bag_uri_for(bag_filename)


class PathToRosbag:
    def __init__(self, bag_dir="~/ros_data/paths"):
        self.writer_pool = Rosbag2WriterPool(bag_dir)

    def save_path_to_bag(
        self,
        x_traj,
        bag_filename="path.bag",
        dt=0.1,
        robot_id=0,
        frame_id="world",
    ):
        x_traj = np.asarray(x_traj)
        time_offset = self.writer_pool.start_time(bag_filename, dt)
        path_msg = PathMsg()
        path_msg.header.frame_id = frame_id
        path_msg.header.stamp = seconds_to_time_msg(time_offset)

        poses = []
        for i in range(x_traj.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose_time = time_offset + i * dt
            pose.header.stamp = seconds_to_time_msg(pose_time)
            pose.pose.position.x = float(x_traj[i, 0])
            pose.pose.position.y = float(x_traj[i, 1])
            pose.pose.position.z = 0.1
            pose.pose.orientation.w = 1.0
            poses.append(pose)

        path_msg.poses = poses
        self.writer_pool.write(
            bag_filename,
            f"/robot_{robot_id}/planned_path",
            path_msg,
            time_offset,
        )
        return self.writer_pool.bag_uri_for(bag_filename)


class MapINfoToMarkers:
    def __init__(self, bag_dir, frame_id="world"):
        self.writer_pool = Rosbag2WriterPool(bag_dir)
        self.frame_id = frame_id
        self.topic_name = "/map_distribution"

    def generate_probmap_marker(
        self,
        points,
        probs,
        sigma=0.8,
        stamp=None,
        colors_hex=None,
    ):
        colors_hex = colors_hex or ["#ffffff", "#000000"]
        points = np.asarray(points)
        probs = np.asarray(probs)
        if len(probs) != len(points):
            raise ValueError("probs and points length mismatch")

        x_vals = np.unique(np.round(points[:, 0], decimals=6))
        y_vals = np.unique(np.round(points[:, 1], decimals=6))
        nx, ny = len(x_vals), len(y_vals)

        try:
            prob_grid = probs.reshape((ny, nx))
        except ValueError:
            from scipy.interpolate import griddata

            xi, yi = np.meshgrid(
                np.linspace(x_vals.min(), x_vals.max(), nx),
                np.linspace(y_vals.min(), y_vals.max(), ny),
            )
            prob_grid = griddata(points, probs, (xi, yi), method="nearest")
            prob_grid = np.nan_to_num(prob_grid)

        prob_grid = gaussian_filter(prob_grid, sigma=sigma)
        p_min, p_max = prob_grid.min(), prob_grid.max()
        if p_max > p_min:
            normalized_grid = (prob_grid - p_min) / (p_max - p_min)
        else:
            normalized_grid = np.zeros_like(prob_grid)

        low_color = hex_to_rgb_float(colors_hex[0])
        high_color = hex_to_rgb_float(colors_hex[-1])
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = seconds_to_time_msg(stamp or 0.0)
        marker.ns = "heatmap"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.scale.y = 0.03

        for i in range(ny):
            for j in range(nx):
                value = float(normalized_grid[i, j])
                color_tuple = lerp_color(low_color, high_color, value)
                marker.points.append(Point(x=float(x_vals[j]), y=float(y_vals[i]), z=0.0))
                marker.colors.append(
                    ColorRGBA(
                        r=float(color_tuple[0]),
                        g=float(color_tuple[1]),
                        b=float(color_tuple[2]),
                        a=1.0,
                    )
                )
        return marker

    def save_probmap_to_bag(
        self,
        points,
        probs,
        dt,
        timesteps,
        color,
        bag_filename="map.bag",
        mode="w",
    ):
        time_offset = 0.0 if mode == "w" else self.writer_pool.start_time(bag_filename, dt)
        marker = self.generate_probmap_marker(points, probs, stamp=time_offset, colors_hex=color)
        for i in range(timesteps):
            timestamp = time_offset + i * dt
            marker.header.stamp = seconds_to_time_msg(timestamp)
            self.writer_pool.write(bag_filename, self.topic_name, marker, timestamp)
        return self.writer_pool.bag_uri_for(bag_filename)


TrajectoryToRosbag = CommandToRosbag


class DistributionBagWriter(MapINfoToMarkers):
    def save_distri2bag(
        self,
        probs,
        points,
        dt,
        time_spans=1,
        bag_filename="map.bag",
        mode="w",
    ):
        return self.save_probmap_to_bag(
            points,
            probs,
            dt,
            time_spans,
            ["#ffffff", "#000000"],
            bag_filename=bag_filename,
            mode=mode,
        )
