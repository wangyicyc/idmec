import jax.numpy as jnp
import json
from experiment.settings import config

dt = config["dt"]
STATE_DIM = 4
CONTROL_DIM = 2
ROBOT_COLORS = ["#1D6CD4", "#FFE100", "#01A064", "#AA00FF"]
TRUE_ROBOT_BAG = "true_robot.bag"


def export_map_to_jsonl(map_info, log_file=None):
    mapinfo_to_save = {
        "means": [m.tolist() for m in map_info['means']],
        "covs": [c.tolist() for c in map_info['covs']],
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(mapinfo_to_save) + "\n")

def load_map_history_jsonl(file_path="map_history.jsonl"):
    history = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # 转换 means 和 covs 为 jnp.array 列表
            converted_record = {
                "means": [jnp.array(m) for m in record["means"]],
                "covs": [jnp.array(c) for c in record["covs"]],
            }
            history.append(converted_record)
    return history


def belief_bag_name(robot_id):
    return f"robot{robot_id}.bag"


def state_slice(robot_id):
    return slice(robot_id * STATE_DIM, (robot_id + 1) * STATE_DIM)


def control_slice(robot_id):
    return slice(robot_id * CONTROL_DIM, (robot_id + 1) * CONTROL_DIM)


def save_robot_traj_and_path(
    sol_traj,
    robot_id,
    bag_filename,
    command_saver,
    path_saver,
    current_time,
):
    x_data = sol_traj["x"][:current_time, state_slice(robot_id)]
    u_data = sol_traj["u"][:current_time, control_slice(robot_id)]
    command_saver.save_traj2bag(
        trajectory_dict={"x": x_data, "u": u_data},
        bag_filename=bag_filename,
        dt=dt,
        robot_id=robot_id,
        mode="a",
    )
    path_saver.save_path_to_bag(
        x_data[:, :2],
        bag_filename=bag_filename,
        dt=dt,
        robot_id=robot_id,
        frame_id="world",
    )


def save_consolidated_segments(sol_trajs, command_saver, path_saver, current_time):
    robot_number = config["robot_number"]

    for robot_id in range(robot_number):
        save_robot_traj_and_path(
            sol_trajs[robot_id],
            robot_id,
            TRUE_ROBOT_BAG,
            command_saver,
            path_saver,
            current_time,
        )

    for belief_id in range(robot_number):
        bag_filename = belief_bag_name(belief_id)
        for robot_id in range(robot_number):
            save_robot_traj_and_path(
                sol_trajs[belief_id],
                robot_id,
                bag_filename,
                command_saver,
                path_saver,
                current_time,
            )


def save_consolidated_maps(target_distr, robot_distr, map_saver, update_map_freq):
    map_saver.save_probmap_to_bag(
        target_distr.evals[1],
        target_distr.evals[0],
        dt=dt,
        timesteps=update_map_freq,
        bag_filename=TRUE_ROBOT_BAG,
        color=["#ffffff", "#000000"],
        mode="a",
    )

    for robot_id, distribution in enumerate(robot_distr):
        color = ["#ffffff", ROBOT_COLORS[robot_id % len(ROBOT_COLORS)]]
        map_saver.save_probmap_to_bag(
            distribution.evals[1],
            distribution.evals[0],
            dt=dt,
            timesteps=update_map_freq,
            bag_filename=belief_bag_name(robot_id),
            color=color,
            mode="a",
        )
