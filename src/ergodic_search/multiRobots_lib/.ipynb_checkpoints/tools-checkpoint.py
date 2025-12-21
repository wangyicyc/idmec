import jax.numpy as jnp
import pandas as pd
import jax.numpy as jnp

def get_Decay_alpha(beta):
    # 计算归一化因子
    norm_factor = beta['future'].max()
    # 计算 future 和 past（直接基于原数组，不修改原数据）
    future_alpha = jnp.clip(beta['future'] / norm_factor, 0.05, 1)
    past_alpha = jnp.clip(beta['past'] / norm_factor, 0.05, 1)
    # 新建一个 other_alpha 列表，不修改原 beta['other']
    other_alpha = []
    for r_id in range(len(beta['other'])):
        other_r = beta['other'][r_id]
        if other_r.shape[0] > 0:  # 非空数组
            scaled = jnp.clip(other_r / norm_factor, 0.05, 1)
        else:
            scaled = other_r  # 保持空数组不变
        other_alpha.append(scaled)
    # 构造新的 decayed_alpha，完全不修改输入的 beta
    decayed_alpha = {
        'future': future_alpha,
        'other': other_alpha,   # 新建的列表
        'past': past_alpha,
    }
    return decayed_alpha

# min_value = 0.01  # 最终收敛的最小值
    # # 生成指数衰减序列
    # t = jnp.arange(2 * past_steps)
    # decay_rate = jnp.log(min_value) / (2 * past_steps - 1)  # 计算衰减率
    # beta_traj2To1 = jnp.exp(decay_rate * t)

def update_beta(sol_trajs, robot_pairs, involved_robots, betas, add_steps, decay_type='linear'):
    # TODO: 留存过去的beta值，即以前的'y'也考虑进去
    min_val = 0.001
    max_val = 1.0
    n_robots = len(sol_trajs)
    new_betas = []

    for r_id in range(n_robots):
        # future 和 past
        future_arr = jnp.full((sol_trajs[r_id]['x'].shape[0],), max_val)
        # other: 初始化为列表，每个 other_id 对应一个数组
        other_list = []
        for other_id in range(n_robots):
            # init to zero array for all others in r_id
            if other_id != r_id:  # 避免重复计算
                old_len = betas[r_id]['other'][other_id].shape[0]
                if decay_type == 'nodecay':
                    other_list.append(jnp.full((old_len,), max_val))
                elif decay_type == 'linear':
                    other_list.append(jnp.linspace(min_val, max_val * old_len / sol_trajs[r_id]['px'].shape[0], num=old_len))
            else:
                other_list.append(jnp.array([]))  # self 对应空数组
        # 如果是 involved_robots，更新为 linspace
        if r_id in involved_robots:
            for other_id in involved_robots:
                if other_id != r_id:
                    T_other = sol_trajs[r_id]['y'][other_id].shape[0]
                    if decay_type == 'nodecay':
                        other_list[other_id] = jnp.full((T_other,), max_val)
                    elif decay_type == 'linear':
                        other_list[other_id] = jnp.linspace(min_val, max_val, num=T_other)

        if decay_type == 'linear' and len(sol_trajs[r_id]['y']) > 2:
            past_arr = jnp.linspace(min_val, max_val, num=sol_trajs[r_id]['px'].shape[0])
        elif decay_type == 'nodecay' or len(sol_trajs[r_id]['y']) < 2:
            past_arr = jnp.full((sol_trajs[r_id]['px'].shape[0],), max_val)

        total_future = jnp.sum(future_arr)
        total_past = jnp.sum(past_arr)
        total_other = sum(jnp.sum(arr) for arr in other_list if arr.size > 0)
        normalization_factor = total_future + total_past + total_other + 1e-8

        # 创建 new_beta，全部使用 jnp.array 或 list of jnp.array
        new_beta = {
            'future': future_arr / normalization_factor,
            'past': past_arr / normalization_factor,
            'other': [arr / normalization_factor for arr in other_list]  # 列表，元素为 jnp.array
        }
        new_betas.append(new_beta)

    return new_betas

def exchange_info(sol_trajs, robot_pairs, timestep, betas, decay_type='linear'):
    # 1. 提取所有涉及到的机器人ID（去重）
    involved_robots = set()
    for i, j in robot_pairs:
        involved_robots.add(i)
        involved_robots.add(j)
    new_sol_trajs = []
    for r_id in range(len(sol_trajs)):
        # 提取当前时间步之前的数据
        new_data = sol_trajs[r_id]['x'][:timestep, 0:2]
        # 创建新的轨迹字典
        new_traj = {}
        # 浅拷贝
        for key in sol_trajs[r_id]:
            new_traj[key] = sol_trajs[r_id][key]
        # new_traj = {
        #     'x': sol_trajs[r_id]['x'].copy(),
        #     'u': sol_trajs[r_id]['u'].copy(),
        #     'px': sol_trajs[r_id]['px'].copy(),
        #     'y': [arr.copy() if arr.size > 0 else arr for arr in sol_trajs[r_id]['y']]  # 假设 y 是列表
        # }
        # 更新未来轨迹（截断到当前时间步之后）
        new_traj['x'] = sol_trajs[r_id]['x'][timestep:, :]
        new_traj['u'] = sol_trajs[r_id]['u'][timestep:, :]
        # 更新过去轨迹
        if sol_trajs[r_id]['px'].size == 2:  # 初始状态
            new_traj['px'] = new_data
        else:
            new_traj['px'] = jnp.concatenate(
                [sol_trajs[r_id]['px'], new_data], axis=0)
        # 添加到新的轨迹列表
        new_sol_trajs.append(new_traj)

    final_sol_trajs = []
    for r_id in range(len(sol_trajs)):
        # 创建新的轨迹字典
        final_traj = {}
        # 浅拷贝
        for key in sol_trajs[r_id]:
            final_traj[key] = new_sol_trajs[r_id][key] 
        if r_id in involved_robots:
            for other_id in involved_robots:
                if other_id != r_id and (min(r_id, other_id), max(r_id, other_id)) in robot_pairs:
                    final_traj['y'][other_id] = new_sol_trajs[other_id]['px'][:, 0:2]
                    # print(final_traj['y'][other_id].shape)
        final_sol_trajs.append(final_traj)

    # 更新 beta 值
        
    new_betas = update_beta(final_sol_trajs, robot_pairs, involved_robots, betas, timestep, decay_type=decay_type)
    return final_sol_trajs, new_betas, involved_robots

def find_first_connected(sol_trajs, threshold):
    """
    找到最早发生机器人连接的时间步及所有在该时间步连接的机器人对。
    参数:
        sol_trajs: 轨迹列表，每个元素是包含 'x' 键的字典（轨迹数据）
        threshold: 距离阈值
    
    返回:
        min_timestep: 最早满足条件的时间步（若无则为 None）
        connected_pairs: 在该时间步满足条件的所有机器人对列表（若无则为空列表）
    """
    n_robots = len(sol_trajs)
    n_timesteps = sol_trajs[0]['x'].shape[0]  # 假设所有轨迹长度相同
    
    # 提取所有机器人的轨迹坐标 (n_robots, n_timesteps, 2)
    traj_coords = jnp.array([traj['x'][:, 0:2] for traj in sol_trajs])
    
    # 计算所有机器人对的距离矩阵 (n_robots, n_robots, n_timesteps)
    diff = traj_coords[:, None, :, :] - traj_coords[None, :, :, :]
    distances = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
    
    # 创建布尔掩码：距离是否小于阈值
    mask = distances < threshold
    
    # 初始化变量
    min_timestep = None
    connected_pairs = []
    
    # 遍历所有机器人对
    for i in range(n_robots):
        for j in range(i + 1, n_robots):  # 只考虑 i < j 避免重复
            # 获取该机器人对满足条件的时间步
            timesteps = jnp.arange(n_timesteps)[mask[i, j]]
            
            if len(timesteps) > 0:
                # 获取该对的最小时间步
                pair_min_timestep = timesteps[0]
                
                # 更新全局最小时间步
                if min_timestep is None or pair_min_timestep < min_timestep:
                    min_timestep = pair_min_timestep
                    connected_pairs = [(i, j)]  # 重置列表
                elif pair_min_timestep == min_timestep:
                    connected_pairs.append((i, j))  # 添加到列表
    
    return min_timestep, connected_pairs

# def find_first_connected(sol_trajs, threshold):
#     n_robots = len(sol_trajs)
#     n_timesteps = sol_trajs[0]['x'].shape[0]  # 假设所有轨迹长度相同
    
#     # 提取所有机器人的轨迹坐标 (n_robots, n_timesteps, 2)
#     traj_coords = jnp.array([traj['x'][:, 0:2] for traj in sol_trajs])
    
#     # 计算所有机器人对的距离矩阵 (n_robots, n_robots, n_timesteps)
#     diff = traj_coords[:, None, :, :] - traj_coords[None, :, :, :]  # 广播计算差值
#     distances = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))  # 欧氏距离
    
#     # 找到全局最小时间步
#     mask = distances < threshold
#     min_timestep = None
#     robot_pair = None
    
#     for i in range(n_robots):
#         for j in range(i + 1, n_robots):
#             # 用布尔数组 mask[i, j] 对 jnp.arange(n_timesteps) 进行筛选，仅保留 mask[i, j] 为 True 的时间步。
#             timesteps = jnp.arange(n_timesteps)[mask[i, j]]
#             if len(timesteps) > 0:
#                 current_min = timesteps[0]
#                 if (min_timestep is None) or (current_min < min_timestep):
#                     min_timestep = current_min
#                     robot_pair = (i, j)
#     return min_timestep, robot_pair

# import jax.numpy as jnp