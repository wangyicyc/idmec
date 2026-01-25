import jax.numpy as jnp
from jax import vmap, jit
import numpy as np
import yaml
import copy
import sys 
from functools import partial
sys.path.append('..')
from libs.target_distribution import TargetDistribution # 构建目标分
from libs.plot_utils import plot_trajs

with open("../libs/config.yaml", "r") as f:
    loaded = yaml.safe_load(f)
# 提取 opt_args 子字典
config = loaded["opt_args"]
workspace_size = config["workspace"]
workspace_bnds = jnp.array([[0.0, workspace_size], [0.0, workspace_size]])         # 工作空间边界,1x1米
# 按需还原为 array / diag 矩阵等
opt_args = {
    "workspace": workspace_size,
    "dt": config["dt"],
    "tsteps": config["tsteps"],
    "R": jnp.array(config["R"]),
    "Q_z": jnp.array(config["Q_z"]),
    "R_v": jnp.array(config["R_v"]),
    "U_max": jnp.array(config["U_max"]),
    "U_min": jnp.array(config["U_min"]),
    "V_max": jnp.array(config["V_max"]),
    "V_min": jnp.array(config["V_min"]),
    "x0": jnp.array(config["x0"]),
    "xf": jnp.array(config["xf"]),
    "avoidance_radius": config["avoidance_radius"],
    "weight_erg": config["weight_erg"],
    "robot_number": config["robot_number"],
    "com_radius": config["com_radius"],
    "minimum_probability": config["minimum_probability"],
    "weight_probability": config["weight_probability"],
    "weight_avoidance": config["weight_avoidance"],
    "weight_barrierCost": config["weight_barrierCost"],
    "period_num": config["period_num"],
    "update_map_freq": config["update_map_freq"],
    "map_merge_freq": config["map_merge_freq"],
    "connect_threshold": config["connect_threshold"],
    "robot_pair": [tuple(pair) for pair in config["robot_pair"]],
}
tsteps = opt_args["tsteps"]
raw = loaded["mapinfo_point"]
# 还原为 jnp.array
mapinfo_point1 = {
    "means": [jnp.array(m) for m in raw["means"]],
    "covs": [jnp.array(c) for c in raw["covs"]],
}
mapinfo_point2 = {
    "means": [jnp.array([1.4, 2.5]), jnp.array([2.5, 1.0]), jnp.array([2.6, 3.0])],
    "covs": [jnp.array(c) for c in raw["covs"]],
}
target_distr = TargetDistribution(workspace_size, mapinfo_point1)          # 目标分布对

robot_distr = TargetDistribution(workspace_size, mapinfo_point2)
plot_trajs('Blues', robot_distr, save_path='belief_map.png')

# target_distr._dist_params['means'] = [jnp.array([1.4, 3.2]), 
#                                       jnp.array([1.6, 0.8])]

#   - [2.0, 2.9]
#   - [1.2, 1.1]
#   - [3.2, 1.5]
# update_map(update_times = 1)

plot_trajs('Greys', target_distr, save_path='real_map.png')

robot_distr.bayes_filter_reset(target_distr.evals[0], jnp.array([2.0, 2.0]))

plot_trajs('Blues', robot_distr, save_path='belief_map_after_filter.png')

