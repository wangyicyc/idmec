import numpy as np
import jax.numpy as jnp
import sys 
sys.path.append('..')
from multiRobots_lib.hyper_params import (
    robot_number,
    robot_model_single, 
)
import logging
logging.basicConfig(
    filename='../log/app.log',          # 日志文件名
    level=logging.INFO,          # 日志等级
    format='%(asctime)s [%(levelname)s] %(message)s',  # 格式
)
beta = 0.05
state_dim = robot_model_single.nx
min_val = 0.001
max_val = 1.0
def inv_decay_function(x):
    return -jnp.log(1 - x) / beta
def decay_function(x):
    return 1-jnp.exp(-beta * x)

def decay_function_sequence(min_value = 0.05, max_value = 0.95, num = 100):
    if max_value > 0.95:
        max_value = 0.95
    if min_value < 0.05:
        min_value = 0.05
    x_min = inv_decay_function(min_value)  # 约等于 1.0
    x_max = inv_decay_function(max_value)
    points = jnp.linspace(x_min, x_max, num)
    arr = decay_function(points)
    return arr 

def get_Decay_alpha(beta):
    # logging.info(f"beta['x'].max(): {beta['x'].max()}")
    if beta['x'].ndim == 1: 
        future_alpha = jnp.clip(beta['x'] / beta['x'].max(), 0.20, 1)
        past_alpha = []
        for r_id in range(robot_number):
            past_r = beta['px'][r_id]
            if past_r.shape[0] > 1:  # 非空数组
                scaled = jnp.clip(past_r / beta['x'].max(), 0.20, 1)
                # logging.info(f"past_r.max(): {past_r.max()}")
            else:
                scaled = past_r  # 保持空数组不变
                # logging.info(f"past_r.max(): {0}")
            past_alpha.append(scaled)
        # 构造新的 decayed_alpha，完全不修改输入的 beta
        decayed_alpha = {
            'x': future_alpha,
            'px': past_alpha,
        }
    else:
        future_alpha = jnp.clip(beta['x'] / beta['x'].max(), 0.20, 1)
        past_alpha = []
        for r_id in range(robot_number):
            past_r = beta['px'][r_id]
            if past_r.shape[0] > 1:  # 非空数组
                scaled = jnp.clip(past_r / beta['x'].max(), 0.20, 1)
                # logging.info(f"past_r.max(): {past_r.max()}")
            else:
                scaled = past_r  # 保持空数组不变
                # logging.info(f"past_r.max(): {0}")
            past_alpha.append(scaled)
        # 构造新的 decayed_alpha，完全不修改输入的 beta
        decayed_alpha = {
            'x': future_alpha,
            'px': past_alpha,
        }
    return decayed_alpha

def update_beta(sol_trajs, decay_type, last_exchange_time, accumulated_time):
    new_betas = []
    for r_id in range(robot_number):
        x_traj = sol_trajs[r_id]
        T_other = x_traj['px'][r_id].shape[0]
        future_arr = jnp.full((x_traj['x'].shape[0], robot_number), max_val)
        past_list = []
        for other_id in range(robot_number):
            old_len = x_traj['px'][other_id].shape[0]
            if old_len == 0:
                past_list.append(np.array([]))  # 空数组，size=0
                continue
            if decay_type == 'nodecay':
                past_list.append(jnp.full((old_len), max_val))
            elif decay_type == 'linear':
                past_list.append(np.array(jnp.linspace(min_val, max_val * old_len / T_other, num=old_len)))
            
            if other_id == r_id:
                continue
            # scale_factor = last_exchange_time.get((r_id, other_id), 1) / max(accumulated_time, 1)
            # future_arr = future_arr.at[:, other_id].multiply(scale_factor)
            # future_arr[:, other_id] = future_arr[:, other_id] * last_exchange_time.get((r_id, other_id), 1) / max(accumulated_time, 1)
            past_list[other_id] = past_list[other_id]# * last_exchange_time.get((other_id, r_id), 1) / max(accumulated_time, 1)

        new_beta = {
            'x': np.array(future_arr),
            'px': past_list,
        }
        new_betas.append(new_beta)
    return new_betas

def update_beta_single(sol_trajs, decay_type='linear'):
    new_betas = []
    for r_id in range(robot_number):
        # future 和 past
        x_traj = sol_trajs[r_id]
        T_other = x_traj['px'][r_id].shape[0]
        future_arr = jnp.full((x_traj['x'].shape[0],), max_val)
        past_list = []
        for other_id in range(robot_number):
            old_len = x_traj['px'][other_id].shape[0]
            if old_len == 0:
                past_list.append(jnp.array([]))  # 空数组，size=0
                continue
            elif decay_type == 'nodecay':
                past_list.append(jnp.full((old_len), max_val))
            elif decay_type == 'linear':
                past_list.append(jnp.linspace(min_val, max_val * old_len / T_other, num=old_len))
            
        # total_future = jnp.sum(future_arr)
        # total_past = sum(jnp.sum(arr) for arr in past_list if arr.size > 0)
        # normalization_factor = total_future + total_past + 1e-8
        new_beta = {
            'x': future_arr,
            'px': past_list,
        }
        new_betas.append(new_beta)
    return new_betas