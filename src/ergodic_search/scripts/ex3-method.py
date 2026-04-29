import os
os.environ["JAX_ENABLE_X64"] = "True"
import sys 
sys.path.append('..')
# 增广拉格朗日法优化器
from multiRobots_lib.multi_solver import al_iLQR 
from multiRobots_lib.augument_lagrange_func import loss_traj_multi, eq_constr, ineq_constr_multi, loss_compare_multi
from multiRobots_lib.data_collect import save_ergodic_metrics_to_excel, append_metric, multi_traj_to_rosbag, multi_path_to_rosbag, multi_map_to_rosbag
from multiRobots_lib.plot_utils import plot_trajs
# from multiRobots_lib.plot_utils_paper import plot_trajs
from multiRobots_lib.tools import find_first_connected, exchange_info, multi_robot_optimize_trajs, update_accumulated_time
from multiRobots_lib.decay_utils import update_beta
from multiRobots_lib.data2bag import CommandToRosbag, PathToRosbag, MapINfoToMarkers
from IPython.display import clear_output
import logging
import yaml
from jax import jit
from functools import partial
from multiRobots_lib.hyper_params import (
    opt_args,
    target_distr,
    robot_distr,
    robot_number,
    init_state,
    sol_trajs,
    multi_betas,
    robot_model_single,
)
from datetime import datetime
# 生成包含年月日_时分的日志文件名，例如：app_2026-01-16_15-06.log
log_filename = f"datas/logs/app_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"
logging.basicConfig(
    filename=log_filename,          # 日志文件名
    level=logging.INFO,          # 日志等级
    format='%(asctime)s [%(levelname)s] %(message)s',  # 格式
)
update_map_freq = opt_args["update_map_freq"]
map_merge_freq = opt_args["map_merge_freq"] 
state_dim = robot_model_single.nx
robot_number = opt_args["robot_number"]
tsteps = opt_args["tsteps"]
commandSaver = CommandToRosbag(bag_dir="./")    
pathSaver = PathToRosbag(bag_dir="./")
map_saver = MapINfoToMarkers(bag_dir="./")
# 双机器人初始化
start_pos = opt_args["x0"]
end_pos = opt_args["xf"]
# 假设你想创建包含 n 个 al_iLQR 结果的数组
n = robot_number  # 例如，创建 10 个结果
jit_ineq_constr_multi = jit(partial(ineq_constr_multi, warm_up=False))
warm_up_ineq_constr = jit(partial(ineq_constr_multi, warm_up=True))
traj_solver = []
traj_warmUp = []
for _id in range(n):
    solver = al_iLQR(
        args=opt_args,
        objective=loss_traj_multi,
        dynamics=robot_model_single,
        inequality=jit_ineq_constr_multi,
        equality=eq_constr,
        target_distr=robot_distr[_id].evals,
        robot_id=_id,
    )
    warmup = al_iLQR(
        args=opt_args,
        objective=loss_traj_multi,
        dynamics=robot_model_single,
        inequality=warm_up_ineq_constr,
        equality=eq_constr,
        target_distr=robot_distr[_id].evals,
        robot_id=_id,
    )
    solver.update_dynamics(robot_number)
    warmup.update_dynamics(robot_number)
    traj_solver.append(solver)
    traj_warmUp.append(warmup)
connection_threshold = opt_args["connect_threshold"]
connection_threshold = connection_threshold**2
map_merge_cnt = 0
decay_type = 'linear'
# decay_type = 'nodecay'
init_dual = True
save_path = '../figures/my_strategy.png'
involved_robots = list(range(robot_number))
involved_robots = set(involved_robots)
flag = True
accumulated_time = 0
be_num = 1
object_value = []
global_metric = {
        'time': [],
        'values': [],
    }
last_exchange_time = {}
# 迭代优化并动态可视化轨迹与障碍物分布
warm_up = True
# warm_up = False
logging.info('my_strategy')
plot_trajs(start_pos, end_pos, sol_trajs, multi_betas, robot_distr, save_path)  
while True:
    if warm_up == True:
        # sol_trajs, to_remove = multi_robot_optimize_trajs(involved_robots, sol_trajs, multi_betas, traj_warmUp, init_state, init_dual)
        for r_id in involved_robots:
            sol_trajs[r_id], conv = traj_warmUp[r_id].solve(
            x0=init_state[r_id],
            init_sol=sol_trajs[r_id],      # 如果 solver 会修改它，建议 deep copy
            beta=multi_betas[r_id],
            init_dual=init_dual,
            max_iter=1000,
            if_print=False,
            r_eps = 0.02,
            loss_eps = 1e-6)
    else:
        # sol_trajs, to_remove = multi_robot_optimize_trajs(involved_robots, sol_trajs, multi_betas, traj_solver, init_state, init_dual)
        for r_id in involved_robots:
            sol_trajs[r_id], conv = traj_solver[r_id].solve(
            x0=init_state[r_id],
            init_sol=sol_trajs[r_id],      # 如果 solver 会修改它，建议 deep copy
            beta=multi_betas[r_id],
            init_dual=init_dual,
            max_iter=1000,
            if_print=False,
            r_eps = 0.02,
            loss_eps = 1e-6) 
    # involved_robots -= to_remove
    clear_output(wait=True)                   # 清除上一次输出，动态刷新
    init_dual = False

    if warm_up == True:
        warm_up = False
        init_dual = True
        logging.info("have warm up")
        continue
    else:
        warm_up = True
    current_time, robot_pair = find_first_connected(sol_trajs, connection_threshold, last_exchange_time)
    current_time, accumulated_time = update_accumulated_time(current_time, accumulated_time, be_num)
    map_merge_cnt += current_time
    if map_merge_cnt >= map_merge_freq:
        accumulated_time -= (map_merge_cnt - map_merge_freq)
        current_time -= (map_merge_cnt - map_merge_freq)
        map_merge_cnt = 0
        if accumulated_time >= tsteps:
            # multi_traj_to_rosbag(sol_trajs, commandSaver, current_time)
            # multi_path_to_rosbag(sol_trajs, pathSaver, current_time)
            # multi_map_to_rosbag(robot_distr, map_saver, update_map_freq)
            # map_saver.save_probmap_to_bag(target_distr.evals[1], target_distr.evals[0], 0.12, update_map_freq)
            append_metric(global_metric, loss_compare_multi(sol_trajs, target_distr.evals, current_time))
            break
        robot_pair = []
        
        for r_id in range(robot_number):
            # print(f"current_time = {connect_timestesp}, type = {type(current_time)}")
            q_t = sol_trajs[r_id]['x'][current_time, r_id * state_dim: r_id * state_dim + 2]
            robot_distr[r_id].bayes_filter_reset(target_distr.evals[0], q_t)
            traj_solver[r_id].update_distribution(robot_distr[r_id].evals)
            traj_warmUp[r_id].update_distribution(robot_distr[r_id].evals)
            logging.info("merge map")
    # ================================ 为replan准备 ==============================================================  
    if accumulated_time == update_map_freq * be_num:
        append_metric(global_metric, loss_compare_multi(sol_trajs, target_distr.evals, current_time))
        # map_saver.save_probmap_to_bag(target_distr.evals[1], target_distr.evals[0], 0.12, update_map_freq)
        # multi_map_to_rosbag(robot_distr, map_saver, update_map_freq)
        logging.info("update map")
        robot_pair = []
        target_distr.update_map(accumulated_time, "reset", "read")
        be_num += 1
        if accumulated_time >= tsteps:
            break
    # multi_traj_to_rosbag(sol_trajs, commandSaver, current_time)
    # multi_path_to_rosbag(sol_trajs, pathSaver, current_time)

    # 只在循环外更新last_exchange_time
    for i, j in robot_pair:
        last_exchange_time[(i, j)] = accumulated_time 
        last_exchange_time[(j, i)] = accumulated_time
    sol_trajs, connected_pairs = exchange_info(sol_trajs, robot_pair, current_time, robot_distr, last_exchange_time)
    multi_betas = update_beta(sol_trajs, decay_type, last_exchange_time, accumulated_time)
    if connected_pairs:
        involved_robots = connected_pairs 
    else:
        involved_robots = set(list(range(robot_number)))
    init_state = [traj['x'][0, :] for traj in sol_trajs]     
    init_dual = True
    logging.warning(f"connect time:{current_time}, accumulated time:{accumulated_time}, and the robot pair:{robot_pair}")
    # plot_trajs(start_pos, end_pos, sol_trajs, multi_betas, robot_distr, save_path)  
plot_trajs(start_pos, end_pos, sol_trajs, multi_betas, robot_distr, save_path)  


def save_ergodic_metrics_to_yaml(global_metric, decay_type, map_id, file_path='experiment3.yaml'):
    """
    将遍历性指标保存到 YAML 文件中，每条记录包含 type、map_id 和 metrics 列表。
    
    参数:
        global_metric (dict): 包含 'values' 键的字典，值为数值列表
        decay_type (str): 衰减类型标识（如 'baseline1', 'exp' 等）
        map_id (int or str): 地图的唯一标识符（如 0, 1, 2 或 "map_A"）
        file_path (str): YAML 文件路径，默认为 'experiment2.yaml'
    """
    # 提取数值并转为 float 列表
    values_list = [float(v) for v in global_metric['values']]
    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError:
                print(f"警告：{file_path} 格式错误，将覆盖为新文件。")
                data = []
    else:
        data = []

    new_entry = {
        "type": str(decay_type),
        "metrics": values_list
    }
    
    # 查找是否存在相同的 map_id，如果存在则追加或更新，不存在则创建新的
    map_exists = False
    for item in data:
        if isinstance(item, dict) and item.get('map_id') == map_id:
            # 检查是否已存在相同 type 的记录，若存在则替换
            type_exists = False
            for t in item['types']:
                if t['type'] == decay_type:
                    t.update(new_entry)
                    type_exists = True
                    break
            if not type_exists:
                item['types'].append(new_entry)
            map_exists = True
            break
    
    if not map_exists:
        data.append({
            "map_id": map_id,
            "types": [new_entry]
        })
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, indent=2, sort_keys=False)
    
    print(f"完整数据已成功保存到 {file_path}（map_id={map_id}, type={decay_type}, 共 {len(values_list)} 个值）")

import argparse
parser = argparse.ArgumentParser(description="保存遍历性指标到 YAML 文件")
parser.add_argument("--map_id", type=int, required=True, help="地图 ID，例如 0, 1, 2...")
args = parser.parse_args()
save_ergodic_metrics_to_yaml(global_metric, "method", map_id=args.map_id)


