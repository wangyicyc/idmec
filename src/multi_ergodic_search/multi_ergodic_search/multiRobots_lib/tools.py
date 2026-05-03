import jax.numpy as jnp
import itertools
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import sys 
sys.path.append('..')
logging.basicConfig(
    filename='../log/app.log',          # 日志文件名
    level=logging.INFO,          # 日志等级
    format='%(asctime)s [%(levelname)s] %(message)s',  # 格式
)
from multiRobots_lib.hyper_params import (
    opt_args,
    robot_model_single,
    robot_number,
)
update_map_freq = opt_args["update_map_freq"]
state_dim = robot_model_single.nx
control_dim = robot_model_single.nu
invalid_value = -100.0
def exchange_info(sol_trajs, robot_pairs, timestep, robot_distr, last_exchange_time):
    involved_robots = {r for pair in robot_pairs for r in pair}
    new_sol_trajs = []
    reset_instructions = {}  # list of (r_id, other_eval, other_pos)
    # step1: 构建新的轨迹信息['x']
    for r_id in range(robot_number):
        traj = sol_trajs[r_id]
        # ==================== for map update ===================================
        eval_val = robot_distr[r_id].evals[0]
        pos_val = traj['x'][timestep, r_id * state_dim: r_id * state_dim + 2]
        reset_instructions[r_id] = (eval_val, pos_val)
        # ==================== for map update ===================================
        # update px, however, we need to update the past trajectory for each robot
        # === 更新 other px：同步更新所有长度匹配的机器人历史 ===
        for i in range(robot_number):
            px_i = traj['px'][i]
            new_px = traj['x'][:timestep, i * state_dim: i * state_dim + 2]  # 新的历史位置片段
            # if px_i.shape[0] == 0 and (i in involved_robots or r_id not in involved_robots):
            if px_i.shape[0] == 0:
                traj['px'][i] = new_px
            else:
                traj['px'][i] = jnp.concatenate([px_i, new_px], axis=0)
        # === 更新 other px 结束
        new_traj = {
            **traj,
            'x': traj['x'][timestep:, :],
            'u': traj['u'][timestep:, :],
            # 'px': updated_px,
        }
        new_sol_trajs.append(new_traj)
    # step2: 为每个机器人补全联合轨迹信息
    for r1, r2 in robot_pairs:
        # logging.info("exchange")
        traj1 = new_sol_trajs[r1]
        traj2 = new_sol_trajs[r2]
        traj1['x'] = traj1['x'].at[:, r2 * state_dim:(r2+1) * state_dim].set(traj2['x'][:, r2 * state_dim:(r2 + 1) * state_dim])
        traj1['u'] = traj1['u'].at[:, r2 * control_dim:(r2+1) * control_dim].set(traj2['u'][:, r2 * control_dim:(r2+1) * control_dim])
        traj2['x'] = traj2['x'].at[:, r1 * state_dim:(r1+1) * state_dim].set(traj1['x'][:, r1 * state_dim:(r1 + 1) * state_dim])
        traj2['u'] = traj2['u'].at[:, r1 * control_dim:(r1+1) * control_dim].set(traj1['u'][:, r1 * control_dim:(r1+1) * control_dim])
        traj1['px'][r2] =  traj2['px'][r2]
        traj2['px'][r1] =  traj1['px'][r1]  
        for i in range(robot_number):
            if i == r1 or i == r2:
                continue
            contact_time_r1 = last_exchange_time.get((r1, i), 0)
            contact_time_r2 = last_exchange_time.get((r2, i), 0)
            if contact_time_r1 > contact_time_r2:
                traj2['px'][i] = traj1['px'][i]
                traj2['x'] = traj2['x'].at[:, i * state_dim:(i + 1) * state_dim].set(traj1['x'][:, i * state_dim:(i + 1) * state_dim])
                last_exchange_time[(r2, i)] = contact_time_r1
                last_exchange_time[(i, r2)] = contact_time_r1
            elif contact_time_r1 < contact_time_r2:
                traj1['px'][i] = traj2['px'][i] 
                traj1['x'] = traj1['x'].at[:, i * state_dim:(i + 1) * state_dim].set(traj2['x'][:, i * state_dim:(i + 1) * state_dim])
                last_exchange_time[(r1, i)] = contact_time_r2
                last_exchange_time[(i, r1)] = contact_time_r2
        # ==================== for map update ===================================
        eval_r2, pos_r2 = reset_instructions[r2]
        robot_distr[r1].bayes_filter_reset(eval_r2, pos_r2, 4.0)
        eval_r1, pos_r1 = reset_instructions[r1]
        robot_distr[r2].bayes_filter_reset(eval_r1, pos_r1, 4.0)
        # ==================== for map update ===================================
    return new_sol_trajs, involved_robots

def exchange_info_old(sol_trajs, robot_pairs, timestep, robot_distr):
    involved_robots = {r for pair in robot_pairs for r in pair}
    new_sol_trajs = []
    # 收集每个机器人的自身状态（用于被配对伙伴重置时使用）
    reset_instructions = {}  # r_id -> (eval, pos)
    for r_id in range(robot_number):
        traj = sol_trajs[r_id]
        # ==================== for map update ===================================
        eval_val = robot_distr[r_id].evals[0]
        pos_val = traj['x'][timestep, :2]
        reset_instructions[r_id] = (eval_val, pos_val)
        # ==================== for map update ===================================
        if traj['px'][r_id].size != 0:
            traj['px'][r_id] = jnp.concatenate([traj['px'][r_id], traj['x'][:timestep, :2]], axis=0)
        else:
            traj['px'][r_id] = traj['x'][:timestep, :2]
        new_traj = {
            **traj,
            'x': traj['x'][timestep:, :],
            'u': traj['u'][timestep:, :],
        }
        new_sol_trajs.append(new_traj)
    
    for r1, r2 in robot_pairs:
        traj1 = new_sol_trajs[r1]
        traj2 = new_sol_trajs[r2]
        traj1['px'][r2] = traj2['px'][r2]
        traj2['px'][r1] = traj1['px'][r1]
        for i in range(robot_number):
            if traj1['px'][i].shape[0] > traj2['px'][i].shape[0]:
                traj2['px'][i] = traj1['px'][i]
            elif traj1['px'][i].shape[0] < traj2['px'][i].shape[0]:
                traj1['px'][i] = traj2['px'][i]
        # ==================== for map update ===================================
        eval_r2, pos_r2 = reset_instructions[r2]
        robot_distr[r1].bayes_filter_reset(eval_r2, pos_r2, 4.0)
        eval_r1, pos_r1 = reset_instructions[r1]
        robot_distr[r2].bayes_filter_reset(eval_r1, pos_r1, 4.0)
        # ==================== for map update ===================================
    return new_sol_trajs, involved_robots

def union_find(connected_pairs):
    if not connected_pairs:
        return []
    parent = list(range(robot_number))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        u = find(x)
        v = find(y)
        if u != v:  # 仅合并非自身
            parent[u] = v
    for r1, r2 in connected_pairs:
        union(r1, r2)
    components = defaultdict(list)
    for r in range(robot_number):
        components[find(r)].append(r)
    extended_pairs = []
    for comp in components.values():
        if len(comp) > 1:
            for i in range(len(comp)):
                for j in range(i + 1, len(comp)):
                    extended_pairs.append((comp[i], comp[j]))
    return extended_pairs
    
def find_first_connected(sol_trajs, threshold, last_exchange_time, 
                         exchange_cooldown=3):
    """
    高效版：考虑冷却时间、避免全量距离矩阵计算。
    """
    past_lens = sol_trajs[0]['px'][0].shape[0]
    n_timesteps = sol_trajs[0]['x'].shape[0]
    traj_coords = jnp.stack([traj['x'][:, idx * state_dim: idx * state_dim + 2] for idx, traj in enumerate(sol_trajs)])
    timesteps_arr = jnp.arange(n_timesteps)
    min_timestep = None
    connected_pairs = []
    # 仅遍历上三角组合
    for i, j in itertools.combinations(range(robot_number), 2):
        # 计算距离序列 (T,)
        diff = traj_coords[i] - traj_coords[j]  # 差值 (T, 2)
        dist = jnp.sum(diff ** 2, axis=1)    # 平方距离 (T,)
        # 找出低于阈值的时间步
        mask = (dist <= threshold)
        valid_steps = timesteps_arr[mask]
        if len(valid_steps) == 0:
            continue
        last_t = last_exchange_time.get((i, j), -jnp.inf)
        selected_t = None
        # 遍历所有有效时间步，找第一个满足冷却时间的
        for t_candidate in valid_steps:
            t_candidate = int(t_candidate)
            if t_candidate + past_lens - last_t > exchange_cooldown:
                selected_t = t_candidate
                break  # 找到最早的合法时间即可
        if selected_t is None:
            continue  # 该对在本次无合法连接时刻
        if (min_timestep is None) or (selected_t < min_timestep):
            min_timestep = selected_t
            # logging.info(f"min_timestep:{min_timestep}")
            connected_pairs = [(i, j)]
        elif selected_t == min_timestep:
            connected_pairs.append((i, j))
    connected_pairs = union_find(connected_pairs)
    return min_timestep, connected_pairs

def find_first_connected_single(sol_trajs, threshold, last_exchange_time, 
                         exchange_cooldown=3):
    """
    高效版：考虑冷却时间、避免全量距离矩阵计算。
    """
    past_lens = sol_trajs[0]['px'][0].shape[0]
    n_timesteps = sol_trajs[0]['x'].shape[0]
    traj_coords = jnp.stack([traj['x'][:, 0: 2] for traj in sol_trajs])
    timesteps_arr = jnp.arange(n_timesteps)
    min_timestep = None
    connected_pairs = []
    # 仅遍历上三角组合
    for i, j in itertools.combinations(range(robot_number), 2):
        # 计算距离序列 (T,)
        diff = traj_coords[i] - traj_coords[j]  # 差值 (T, 2)
        dist = jnp.sum(diff ** 2, axis=1)    # 平方距离 (T,)
        # 找出低于阈值的时间步
        mask = (dist <= threshold)
        valid_steps = timesteps_arr[mask]
        if len(valid_steps) == 0:
            continue
        last_t = last_exchange_time.get((i, j), -jnp.inf)
        # 冷却时间检查
        selected_t = None
        # 遍历所有有效时间步，找第一个满足冷却时间的
        for t_candidate in valid_steps:
            t_candidate = int(t_candidate)
            if t_candidate + past_lens - last_t > exchange_cooldown:
                selected_t = t_candidate
                break  # 找到最早的合法时间即可
        if selected_t is None:
            continue  # 该对在本次无合法连接时刻

        if (min_timestep is None) or (selected_t < min_timestep):
            min_timestep = selected_t
            connected_pairs = [(i, j)]
        elif selected_t == min_timestep:
            connected_pairs.append((i, j))
    connected_pairs = union_find(connected_pairs)
    return min_timestep, connected_pairs
def _solve_one_robot(args):
    rid, solver, x0, init_sol, beta, init_dual, r_eps, loss_eps = args
    # 注意：JAX DeviceArray 在线程间传递是安全的（只读）
    sol, conv = solver.solve(
        x0=x0,
        init_sol=init_sol,      # 如果 solver 会修改它，建议 deep copy
        beta=beta,
        init_dual=init_dual,
        max_iter=1000,
        if_print=False,
        r_eps = r_eps,
        loss_eps = loss_eps
    )
    return rid, sol, conv

def optimize_trajs(involved_robots, sol_trajs, betas, traj_solver, init_state, init_dual, r_eps = 0.02, loss_eps = 1e-6):
    to_remove = set()
    # 准备任务参数
    tasks = [
        (rid, traj_solver[rid], init_state[rid], sol_trajs[rid], betas[rid], init_dual, r_eps, loss_eps)
        for rid in involved_robots
    ]
    # 并行执行
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        results = executor.map(_solve_one_robot, tasks)
    # 收集结果
    for rid, sol, conv in results:
        sol_trajs[rid] = sol
        if conv:
            to_remove.add(rid)
    return sol_trajs, to_remove

def _multi_solve_one_robot(args):
    rid, solver, x0, init_sol, beta, init_dual, r_eps, loss_eps = args
    # 注意：JAX DeviceArray 在线程间传递是安全的（只读）
    sol, conv = solver.solve(
        x0=x0,
        init_sol=init_sol,      # 如果 solver 会修改它，建议 deep copy
        beta=beta,
        init_dual=init_dual,
        max_iter=1000,
        if_print=False,
        r_eps = r_eps,
        loss_eps = loss_eps
    )
    return rid, sol, conv

def multi_robot_optimize_trajs(involved_robots, sol_trajs, betas, traj_solver, init_state, init_dual, r_eps = 0.02, loss_eps = 1e-6):
    to_remove = set()
    # 准备任务参数
    tasks = [
        (rid, traj_solver[rid], init_state[rid], sol_trajs[rid], betas[rid], init_dual, r_eps, loss_eps)
        for rid in involved_robots
    ]
    # 并行执行
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        results = executor.map(_multi_solve_one_robot, tasks)
    # 收集结果
    for rid, sol, conv in results:
        sol_trajs[rid] = sol
        if conv:
            to_remove.add(rid)
    return sol_trajs, to_remove

def update_accumulated_time(connect_timestep, accumulated_time, be_num):
    if connect_timestep is None:
        return update_map_freq * be_num - accumulated_time, update_map_freq * be_num
    accumulated_time += connect_timestep
    if accumulated_time >= update_map_freq * be_num:
        connect_timestep -= accumulated_time - update_map_freq * be_num
        accumulated_time = update_map_freq * be_num
    return connect_timestep, accumulated_time
