import sys 
sys.path.append('..')
from multiRobots_lib.fourier_utils import BasisFunc, get_phik, get_ck, get_ck_avg, get_ck_sum  # 获取Fourier基函数和系数
from multiRobots_lib.ergodic_metric import ErgodicMetric # 计算遍历度指标
from multiRobots_lib.metric_utils import pair_connection_doubleInt as func_connection_value
import jax.numpy as jnp
from itertools import combinations
from jax import lax, vmap, jit
from functools import partial
from scipy.special import comb 
basis = BasisFunc(n_basis=[10,10])           # 创建二维傅里叶基函数（用于遍历性指标计算）
erg_metric = ErgodicMetric(basis)            # 遍历性评价指标Ergodic Metric
from multiRobots_lib.hyper_params import (
    opt_args,
    robot_model_single,
    func_pair,
    func_emap,
    barrier_cost_upper,
    barrier_cost_lower,
)
time_span = opt_args["update_map_freq"]
import logging
logging.basicConfig(
    filename='log/app.log',          # 日志文件名
    level=logging.INFO,          # 日志等级
    format='%(asctime)s [%(levelname)s] %(message)s',  # 格式
)
state_dim = robot_model_single.nx
control_dim = robot_model_single.nu
robot_number = opt_args["robot_number"] 
avoid_r = opt_args["avoidance_radius"]

single_R = jnp.diag(jnp.asarray(opt_args["R"]))
robot_pair = opt_args["robot_pair"]  # 机器人对列表

period_num=opt_args['period_num']
w_avoid = opt_args['weight_avoidance']
global_min_prob = opt_args["minimum_probability"]
w_prob = opt_args["weight_probability"]
tsteps = opt_args["tsteps"]
weight_erg = opt_args["weight_erg"]
w_barrierCost = opt_args["weight_barrierCost"]
num_pairs = len(robot_pair)
func_get_ck_sum = jit(
    partial(
        get_ck_sum,
        basis=basis,
    )
)
func_get_ck = jit(
    partial(
        get_ck,
        basis=basis,
    )
)
func_connection_value_jit = jit(
    partial(
        func_connection_value,
        _func_pair=func_pair,
        _nx=state_dim,
        period_num=period_num,
    )
)
V_max = jnp.asarray(opt_args['V_max'])  # (2,)
V_min = jnp.asarray(opt_args['V_min'])  # (2,)
U_max = jnp.asarray(opt_args['U_max'])  # (nu,)
U_min = jnp.asarray(opt_args['U_min'])  # (nu,)
V_max_all = jnp.tile(V_max, robot_number)  # (2 * n_involved,)
V_min_all = jnp.tile(V_min, robot_number)
U_max_all = jnp.tile(U_max, robot_number)  # (nu * n_involved,)
U_min_all = jnp.tile(U_min, robot_number)

# 定义轨迹优化的损失函数
def loss_traj_multi(sol, beta, target_distr, multi_R):
    x_traj = sol.x         # 轨迹点序列
    past_traj = sol.px
    u_traj = sol.u         # 控制输入序列
    phik = get_phik(target_distr, basis)     # 目标分布的傅里叶系数 
    ckx = func_get_ck_sum(x_traj, beta.x)   # 轨迹的傅里叶系数
    def _func_get_ckpxs(traj, beta_r):
        def compute_ck():
            return func_get_ck(traj, beta_r)
        def zero_result():
            dummy_result = func_get_ck(jnp.zeros((1, 2)), beta_r)  # 用非空轨迹获取正确形状
            return jnp.zeros_like(dummy_result)
        is_valid = jnp.logical_and(traj.shape[0] > 1, traj.size > 2)
        return lax.cond(is_valid, compute_ck, zero_result)
    all_ckpx = vmap(_func_get_ckpxs, in_axes=(0, 0))(jnp.stack(past_traj), jnp.stack(beta.px))
    ckpx = jnp.sum(all_ckpx, axis=0)  # (ck_dim,) - 假设 func_get_ck 返回固定形状
    erg_met = erg_metric(ckx + ckpx, phik)
    ctrl_cost = jnp.sum(0.5 * multi_R @ u_traj.T * u_traj.T) / (tsteps * robot_number)

    all_pos = x_traj.reshape(x_traj.shape[0], -1, state_dim)[:, :, :2]
    def compute_robot_barrier_upper(robot_pos_traj):
        emap_result = func_emap(robot_pos_traj)  # (T, 2) -> (T, 2)
        barrier_matrix = barrier_cost_upper(emap_result)  # (T, 2) -> (T, 2)
        return jnp.sum(barrier_matrix)  # (T, 2) -> 标量
    
    def compute_robot_barrier_lower(robot_pos_traj):
        emap_result = func_emap(robot_pos_traj)  # (T, 2) -> (T, 2)
        barrier_matrix = barrier_cost_lower(emap_result)  # (T, 2) -> (T, 2)
        return jnp.sum(barrier_matrix)  # (T, 2) -> 标量
    robot_barriers_upper = vmap(compute_robot_barrier_upper)(all_pos.transpose(1, 0, 2))
    robot_barriers_lower = vmap(compute_robot_barrier_lower)(all_pos.transpose(1, 0, 2))
    bar_cost_lower = jnp.sum(robot_barriers_lower) // robot_number
    bar_cost_lower =  w_barrierCost * bar_cost_lower.sum()
    bar_cost_upper = jnp.sum(robot_barriers_upper) // robot_number
    bar_cost_upper =  w_barrierCost * bar_cost_upper.sum()

    return (
        ctrl_cost
        + weight_erg * jnp.log10(erg_met)
        + bar_cost_lower
        + bar_cost_upper
    )
    

def loss_compare_multi(sol, target_distr, current_time):
    total_state_dim = robot_number * state_dim
    x_traj = jnp.zeros((time_span, total_state_dim))
    phik = get_phik(target_distr, basis)     # 目标分布的傅里叶系数 
    arr = jnp.full(time_span, 1.0)
    beta_x = arr / (jnp.sum(arr))
    for i in range(robot_number):
        x_traj = x_traj.at[-current_time:, i * state_dim : i * state_dim + 2].set(sol[i]['x'][:current_time, i * state_dim : i * state_dim + 2])
        x_traj = x_traj.at[:time_span - current_time, i * state_dim : i * state_dim + 2].set(sol[i]['px'][i][-(time_span - current_time):, :])
    ckx = get_ck_avg(trajectory=x_traj, beta=beta_x, basis=basis)
    erg_met = erg_metric(ckx, phik)
    return erg_met

def loss_traj_single(sol, beta, target_distr):
    x_traj = sol.x         # 轨迹点序列
    past_traj = sol.px
    u_traj = sol.u         # 控制输入序列
    phik = get_phik(target_distr, basis)     # 目标分布的傅里叶系数 
    ckx = func_get_ck(x_traj, beta.x)   # 轨迹的傅里叶系数
    ckpx = jnp.zeros_like(ckx)  # 先初始化为 0
    for r_id in range(robot_number):  
        if past_traj[r_id].size > 2:
            beta_r = beta.px[r_id]  # 对 r_id 的 beta 权重
            ckpx = ckpx + func_get_ck(past_traj[r_id], beta_r)
    erg_met = erg_metric(ckx + ckpx, phik)
    ctrl_cost = jnp.sum(0.5 * single_R @ u_traj.T * u_traj.T) / tsteps
    bar_cost_upper = barrier_cost_upper(func_emap(x_traj[:, 0 : 2]))
    bar_cost_upper = w_barrierCost * bar_cost_upper.sum()
    bar_cost_lower = barrier_cost_lower(func_emap(x_traj[:, 0 : 2]))
    bar_cost_lower = w_barrierCost * bar_cost_lower.sum()
    return (
        ctrl_cost
        + weight_erg * jnp.log10(erg_met)
        + bar_cost_lower
        + bar_cost_upper
    )
def loss_compare_single(sol, target_distr, current_time):
    total_state_dim = robot_number * state_dim
    x_traj = jnp.zeros((time_span, total_state_dim))
    phik = get_phik(target_distr, basis)     # 目标分布的傅里叶系数 
    arr = jnp.full(time_span, 1.0)
    beta_x = arr / jnp.sum(arr)
    for i in range(robot_number):
        x_traj = x_traj.at[-current_time:, i * state_dim : i * state_dim + 2].set(sol[i]['x'][:current_time, :2])
        x_traj = x_traj.at[:time_span - current_time, i * state_dim : i * state_dim + 2].set(sol[i]['px'][i][-(time_span - current_time):, :])
    ckx = get_ck_avg(trajectory=x_traj, beta=beta_x, basis=basis)
    erg_met = erg_metric(ckx, phik)
    return erg_met

def ineq_constr_multi(sol, beta_future, warm_up):
    x_traj = sol.x
    u_traj = sol.u 
    T = x_traj.shape[0]
    # 1. Velocity and control bounds (vectorized)
    x_reshaped = x_traj.reshape(T, robot_number, state_dim)      # (T, N, nx)
    u_reshaped = u_traj.reshape(T, robot_number, control_dim)    # (T, N, nu)
    all_vel = x_reshaped[:, :, 2:4].reshape(T, -1)               # (T, 2*N)
    all_u   = u_reshaped.reshape(T, -1)  
    upper_bound_vel = (all_vel - V_max_all).flatten() / robot_number
    lower_bound_vel = (V_min_all - all_vel).flatten() / robot_number
    upper_bound_acc = (all_u - U_max_all).flatten() / robot_number
    lower_bound_acc = (U_min_all - all_u).flatten() / robot_number

    all_pos = x_traj.reshape(x_traj.shape[0], -1, state_dim)[:, :, :2]
    def compute_robot_barrier_upper(robot_pos_traj):
        emap_result = func_emap(robot_pos_traj)  # (T, 2) -> (T, 2)
        barrier_matrix = barrier_cost_upper(emap_result)  # (T, 2) -> (T, 2)
        return jnp.sum(barrier_matrix)  # (T, 2) -> 标量
    
    def compute_robot_barrier_lower(robot_pos_traj):
        emap_result = func_emap(robot_pos_traj)  # (T, 2) -> (T, 2)
        barrier_matrix = barrier_cost_lower(emap_result)  # (T, 2) -> (T, 2)
        return jnp.sum(barrier_matrix)  # (T, 2) -> 标量
    robot_barriers_upper = vmap(compute_robot_barrier_upper)(all_pos.transpose(1, 0, 2))
    robot_barriers_lower = vmap(compute_robot_barrier_lower)(all_pos.transpose(1, 0, 2))
    bar_cost_lower = jnp.sum(robot_barriers_lower) // robot_number
    bar_cost_lower =  w_barrierCost * bar_cost_lower.sum()
    bar_cost_upper = jnp.sum(robot_barriers_upper) // robot_number
    bar_cost_upper =  w_barrierCost * bar_cost_upper.sum()

    # 2. avoiding crash Constraint
    norm_coff = beta_future.max()
    all_pos = x_traj.reshape(x_traj.shape[0], robot_number, state_dim)[:, :, :2]
    def compute_pair_dist(pair_idx):
        i, j = pair_idx
        xi = all_pos[:, i]  # (T, 2)
        xj = all_pos[:, j]  # (T, 2)
        dist_sq = jnp.sum((xi - xj)**2, axis=1)  # (T,)
        decay_scale = jnp.minimum(beta_future[0, i], beta_future[0, j]) / norm_coff
        return 1.0 - dist_sq / ((avoid_r * decay_scale) ** 2) # (T,)
    robot_pair = list(combinations(range(robot_number), 2))
    robotPair_array = jnp.array(robot_pair, dtype=jnp.int32)  # (num_pairs, 2)
    _avoidance_arr = vmap(compute_pair_dist)(robotPair_array).flatten() / comb(
    robot_number, 2) # (T * num_pairs,)
    weighted_avoidance = w_avoid * _avoidance_arr
    if warm_up == False:
        # 3. Connection Probability Constraint
        T_used = (x_traj.shape[0] // period_num) * period_num  # 向下取整到 period_num 的倍数
        connection_probability = func_connection_value(
            traj = x_traj, 
            robot_pair = robot_pair,
            _func_pair=func_pair,
            beta_future=beta_future,
            _nx=state_dim,
            period_num=period_num,
            traj_length=T_used
            )
        # Step 1: 转换 robot_pair 为 JAX 数组
        pair_array = jnp.array(opt_args["robot_pair"])  # shape (6, 2)
        # Step 2: 提取 i 和 j 索引
        i_idx = pair_array[:, 0]  # shape (6,)
        j_idx = pair_array[:, 1]  # shape (6,)
        # Step 3: 从 beta_future 中索引（只用第 0 行）
        beta = beta_future[0, :] / (norm_coff + 1e-8)  # shape (4,)
        # Step 4: 计算每对的 beta 和
        beta_sum = jnp.minimum(beta[i_idx], beta[j_idx])  # shape (6,)
        # Step 5: 计算 per-pair 最小概率
        min_prob_per_pair = global_min_prob * (T_used / period_num) * beta_sum  # shape (6,)
        min_prob_expanded = jnp.repeat(min_prob_per_pair, period_num)  # shape (6 * T,)
        prob_connection = w_prob * (-connection_probability / min_prob_expanded + 1)
        return jnp.r_[weighted_avoidance, upper_bound_acc, lower_bound_acc, 
                      upper_bound_vel, lower_bound_vel, prob_connection]
    return jnp.r_[weighted_avoidance, upper_bound_acc, lower_bound_acc, 
                  upper_bound_vel, lower_bound_vel]

def ineq_constr_single(sol):
    """ inequality constraints including control input bounds for trajectory """
    x_traj = sol.x
    _nx = robot_model_single.nx
    vel_traj = sol.x[:, 2:4]         # 状态轨迹序列，形状: (tsteps, nx)
    u_traj = sol.u         # 控制输入序列，形状: (tsteps, nu)
    upper_bound_acc = (u_traj - U_max).flatten()
    lower_bound_acc = (U_min - u_traj).flatten()
    upper_bound_vel = (vel_traj - V_max).flatten()
    lower_bound_vel = (V_min - vel_traj).flatten()
    bar_cost_upper = barrier_cost_upper(func_emap(x_traj[:, 0 : 2]))
    bar_cost_upper = w_barrierCost * bar_cost_upper.sum()
    bar_cost_lower = barrier_cost_lower(func_emap(x_traj[:, 0 : 2]))
    bar_cost_lower = w_barrierCost * bar_cost_lower.sum()
    return jnp.r_[upper_bound_acc, lower_bound_acc, 
                  upper_bound_vel, lower_bound_vel,
                  ]  # 如有其他约束，继续拼接
# 定义动力学等式约束
def eq_constr_single(sol):
    # """ dynamic equality constriants """
    # x_traj = sol.x         # 轨迹点序列
    # bar_cost = barrier_cost(func_emap(x_traj[:, 0 : 2]))
    # return w_barrierCost * bar_cost.sum()
    return 0
def eq_constr(sol):
    """ dynamic equality constriants """
    # x_traj = sol.x         # 轨迹点序列
    # all_pos = x_traj.reshape(x_traj.shape[0], -1, state_dim)[:, :, :2]
    # def compute_robot_barrier(robot_pos_traj):
    #     emap_result = func_emap(robot_pos_traj)  # (T, 2) -> (T, 2)
    #     barrier_matrix = barrier_cost(emap_result)  # (T, 2) -> (T, 2)
    #     return jnp.sum(barrier_matrix)  # (T, 2) -> 标量
    # robot_barriers = vmap(compute_robot_barrier)(all_pos.transpose(1, 0, 2))
    # bar_cost = jnp.sum(robot_barriers) // robot_number
    # return w_barrierCost * bar_cost.sum()
    return 0
