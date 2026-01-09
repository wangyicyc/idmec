import jax.numpy as jnp
from jax import vmap, jit
import numpy as np
import yaml
import copy
import sys 
import math
from functools import partial
sys.path.append('..')
from libs.target_distribution import TargetDistribution # 构建目标分
from libs.plot_utils import plot_trajs, plot_weight

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
mapinfo_point = {
    "means": [jnp.array(m) for m in raw["means"]],
    "covs": [jnp.array(c) for c in raw["covs"]],
}

target_distr = TargetDistribution(workspace_size, mapinfo_point)          # 目标分布对

# 空间定义
n = 2
size = 4.0
sigma_d = 0.5
grids_points = math.ceil(100 * size)
        # 创建空间网格
domain = jnp.meshgrid(
    *[jnp.linspace(0, size, grids_points)]*n
)
s = jnp.stack([X.ravel() for X in domain]).T
history_index = 0 
dist_params = {
            'means': [],
            'covs': [],
        }



diff_to_uav = s - jnp.array([1.5, 2.5])  # (N, 2)
d = jnp.sum(diff_to_uav**2, axis=1)  # (N,)
# 2. 计算可信度权重（高斯衰减）
weight = jnp.exp(-d / (2*sigma_d**2))  # (N,)
weight_norm = weight / jnp.sum(weight)  # 归一化权重
# 观察到的分布（假设为
# 4. 更新
target_distr.evals = (weight_norm, s)
plot_weight(target_distr, save_path='weight_map.png')


