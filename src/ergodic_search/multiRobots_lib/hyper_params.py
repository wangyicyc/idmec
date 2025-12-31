import jax.numpy as jnp
from jax import vmap, jit
import numpy as np
import yaml
import copy
import sys 
from functools import partial
sys.path.append('..')
from multiRobots_lib.target_distribution import TargetDistribution # 构建目标分布
from multiRobots_lib.dynamics import DoubleIntegrator, SingleIntegrator, HomoDynamics
from multiRobots_lib.metric_utils import gaussian_function

with open("../datas/config/config.yaml", "r") as f:
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
robot_distr = []
robot_number = config["robot_number"] 
for robot_id in range(robot_number):
    map_copy = copy.deepcopy(mapinfo_point)
    robot_distr.append(TargetDistribution(workspace_size, map_copy))          # 目标分布对
    # robot_distr.append(TargetDistribution(workspace_size))          # 目标分布对

start_pos = opt_args["x0"]
end_pos = opt_args["xf"]
dt = opt_args["dt"]
def init_traj_single(start_pos, end_pos, tsteps, dt):
    _x_traj = jnp.linspace(start_pos, end_pos, tsteps + 1)
    _x_dot = jnp.vstack([
        ((_x_traj[1:, :] - _x_traj[:-1, :]) / dt),
        jnp.zeros((1, start_pos.shape[0])),
    ])
    _init_x_traj = jnp.concatenate([_x_traj, _x_dot], axis=1)
    init_state = _init_x_traj[0]
    return _init_x_traj, init_state


def init_traj_multi(start_pos, end_pos, tsteps, dt):
    _x_traj = jnp.linspace(
        start_pos,
        end_pos,
        tsteps + 1,
    )
    _x_dot = jnp.vstack(
        [
            ((_x_traj[1:, :] - _x_traj[:-1, :]) / dt),
            np.zeros(shape=(2 * robot_number)),
        ]
    )
    _init_x_traj = jnp.column_stack(
        [_x_traj.reshape(-1, 2), _x_dot.reshape(-1, 2)]
    ).reshape(-1, _x_traj.shape[1] + _x_dot.shape[1])
    _init_u_traj = np.zeros((tsteps, robot_number * 2))

    init_state = _init_x_traj[0, :]
    init_sol = {
        "x": _init_x_traj[:-1], 
        "u": _init_u_traj, 
        "px": [jnp.zeros((0, 2))] * robot_number  # (0, 2) 形状的空轨迹
        }
    return init_sol, init_state

_init_u_traj = np.zeros((tsteps, 2))
init_state = []
sol_trajs = []
init_state_old = []
sol_trajs_old = []
for robot_id in range(robot_number):
    _init_x_traj, _init_state = init_traj_single(
        start_pos = start_pos[robot_id*2: robot_id*2+2], 
        end_pos = end_pos[robot_id*2: robot_id*2+2],
        tsteps = tsteps,
        dt = dt
        )
    sol_trajs_old.append(
        {"x": _init_x_traj[:-1], 
         "u": _init_u_traj, 
         "px": [jnp.array([]) for _ in range(robot_number)]}
    )
    init_state_old.append(_init_state)

for robot_id in range(robot_number):
    _init_sol, _init_state = init_traj_multi(
        start_pos = start_pos, 
        end_pos = end_pos,
        tsteps = tsteps,
        dt = dt
        )
    sol_trajs.append(_init_sol)
    init_state.append(_init_state)


archetype_robot_model = DoubleIntegrator()             # 单积分器机器人模型
robot_model_single = HomoDynamics(robot_number=1, dynamics=archetype_robot_model)
func_pair = partial(
    gaussian_function,
    # sigmoid_function,
    sigma=opt_args["com_radius"],
)

# 更高效的方法 - 使用列表推导式
F_beta = jnp.full(tsteps, 1)
n_small_betas = jnp.array([jnp.array([])] * robot_number)

# 使用列表推导式创建 betas
betas = [
    {
    'x': np.array(F_beta),
    'px': [np.array([]) for _ in range(robot_number)],
}
    for _ in range(robot_number)
]

F_beta_multi = jnp.full((tsteps, robot_number), 1)
# 使用列表推导式创建 betas
multi_betas = [
    {
    'x': np.array(F_beta_multi),
    'px': [np.array([]) for _ in range(robot_number)],
}
    for _ in range(robot_number)
]

def emap(x, workspace_bnds):
    """将状态归一化到工作空间坐标"""
    return jnp.array([
        (x[0]-workspace_bnds[0][0])/(workspace_bnds[0][1]-workspace_bnds[0][0]), 
        (x[1]-workspace_bnds[1][0])/(workspace_bnds[1][1]-workspace_bnds[1][0])
    ])

def barrier_cost_upper(e):
    """边界屏障函数，防止轨迹超出工作空间"""
    # return (jnp.maximum(0, e-1) + jnp.maximum(0, -e))
    return jnp.maximum(jnp.exp(e - 1), 1) - 1 - 1e-6
def barrier_cost_lower(e):
    """边界屏障函数，防止轨迹超出工作空间"""
    # return (jnp.maximum(0, e-1) + jnp.maximum(0, -e))
    return jnp.maximum(jnp.exp(-e), 1) - 1 - 1e-6 

# some function
real_ws_bnds = jnp.array([
    [0.6 * config["avoidance_radius"], workspace_size-0.6 * config["avoidance_radius"]], 
    [0.6 * config["avoidance_radius"], workspace_size-0.6 * config["avoidance_radius"]]
    ])          # 工作空间边界,1x1米
func_emap = vmap(partial(emap, workspace_bnds=real_ws_bnds))