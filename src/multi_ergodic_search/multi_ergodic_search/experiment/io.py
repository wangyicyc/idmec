from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time

import numpy as np

from utils.ros_messages import (
    CONTROL_COMMAND_MSG,
    build_control_command,
    control_topic,
)


VALID_OUTPUT_MODES = {"none", "bag", "topic", "both"}


@dataclass
class ExperimentContext:
    update_map_freq: int
    map_merge_freq: int
    state_dim: int
    robot_number: int
    tsteps: int
    init_state: Any
    sol_trajs: Any
    multi_betas: Any
    target_distr: Any
    robot_distr: Any
    start_pos: Any
    end_pos: Any
    traj_solver: list
    traj_warmup: list
    connection_threshold: float
    map_merge_cnt: int
    decay_type: str
    init_dual: bool
    save_path: str
    involved_robots: set
    accumulated_time: int
    be_num: int
    global_metric: dict
    last_exchange_time: dict
    warm_up: bool
    output: Any


class ExperimentOutput:
    def __init__(
        self,
        output_mode,
        bag_dir,
        output_topic,
        publish_rate,
        robot_number,
        state_dim,
    ):
        self.output_mode = str(output_mode).strip().lower()
        self.bag_dir = str(Path(bag_dir).expanduser())
        self.output_topic = output_topic
        self.publish_rate = publish_rate
        self.robot_number = robot_number
        self.state_dim = state_dim
        self.command_saver = None
        self.path_saver = None
        self.map_saver = None
        self.publishers = None
        self.rclpy = None
        self.node = None
        self.position_command_type = None

    @property
    def enabled(self):
        return self.output_mode != "none"

    @property
    def write_bag(self):
        return self.output_mode in {"bag", "both"}

    @property
    def publish_topic(self):
        return self.output_mode in {"topic", "both"}

    def setup(self):
        if self.output_mode not in VALID_OUTPUT_MODES:
            valid_modes = ", ".join(sorted(VALID_OUTPUT_MODES))
            raise ValueError(
                f"Invalid output_mode '{self.output_mode}'. Use one of: {valid_modes}"
            )
        if self.write_bag:
            from utils.data2bag import (
                CommandToRosbag,
                MapINfoToMarkers,
                PathToRosbag,
            )

            self.command_saver = CommandToRosbag(bag_dir=self.bag_dir)
            self.path_saver = PathToRosbag(bag_dir=self.bag_dir)
            self.map_saver = MapINfoToMarkers(bag_dir=self.bag_dir)
        if self.publish_topic:
            import rclpy

            if not rclpy.ok():
                rclpy.init(args=None)
            self.rclpy = rclpy
            self.node = rclpy.create_node("idmed_trajectory_publisher")
            self.position_command_type = CONTROL_COMMAND_MSG
            self.publishers = [
                self.node.create_publisher(
                    CONTROL_COMMAND_MSG,
                    control_topic(self.output_topic, r_id),
                    10,
                )
                for r_id in range(self.robot_number)
            ]

    def emit_segment(self, context, current_time):
        if not self.enabled or current_time <= 0:
            return
        if self.write_bag:
            from utils.data_collect import multi_path_to_rosbag, multi_traj_to_rosbag

            multi_traj_to_rosbag(context.sol_trajs, self.command_saver, current_time)
            multi_path_to_rosbag(context.sol_trajs, self.path_saver, current_time)
        if self.publish_topic:
            self.publish_control_topics(context, current_time)

    def emit_map_snapshot(self, context):
        if not self.write_bag:
            return
        from utils.data_collect import multi_map_to_rosbag

        multi_map_to_rosbag(context.robot_distr, self.map_saver, context.update_map_freq)
        self.map_saver.save_probmap_to_bag(
            context.target_distr.evals[1],
            context.target_distr.evals[0],
            0.05,
            context.update_map_freq,
            ["#ffffff", "#000000"],
        )

    def publish_control_topics(self, context, current_time):
        sleep_dt = 1.0 / self.publish_rate
        for step in range(current_time):
            if not self.rclpy.ok():
                return
            for r_id, publisher in enumerate(self.publishers):
                publisher.publish(self.build_position_command(context, r_id, step))
            self.rclpy.spin_once(self.node, timeout_sec=0.0)
            time.sleep(sleep_dt)

    def build_position_command(self, context, robot_id, step):
        x_slice = slice(
            robot_id * self.state_dim,
            robot_id * self.state_dim + self.state_dim,
        )
        u_slice = slice(robot_id * 2, robot_id * 2 + 2)
        x_data = np.asarray(context.sol_trajs[robot_id]["x"][step, x_slice])
        u_data = np.asarray(context.sol_trajs[robot_id]["u"][step, u_slice])

        return build_control_command(x_data, u_data)
