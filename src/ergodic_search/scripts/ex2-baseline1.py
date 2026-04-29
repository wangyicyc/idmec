
import os
os.environ["JAX_ENABLE_X64"] = "True"
import sys 
sys.path.append('..')
# 增广拉格朗日法优化器
from multiRobots_lib.solver import al_iLQR 
from multiRobots_lib.augument_lagrange_func import loss_traj_single, eq_constr_single, ineq_constr_single, loss_compare_single
from multiRobots_lib.data_collect import append_metric, traj_to_rosbag, path_to_rosbag
import logging
from datetime import datetime
import numpy as np
from multiRobots_lib.plot_utils import plot_trajs_old
from multiRobots_lib.tools import exchange_info_old, optimize_trajs, update_accumulated_time
from multiRobots_lib.decay_utils import update_beta_single
from multiRobots_lib.data2bag import CommandToRosbag, MapINfoToMarkers, PathToRosbag
from IPython.display import clear_output
import yaml
from multiRobots_lib.hyper_params import (
    opt_args,
    target_distr,
    robot_number,
    init_state_old,
    sol_trajs_old,
    robot_model_single,
    betas,
)

robot_distr = []
for robot_id in range(robot_number):
    robot_distr.append(target_distr)          # 目标分布对

# 生成包含年月日_时分的日志文件名，例如：app_2026-01-16_15-06.log
log_filename = f"datas/logs/app_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"
logging.basicConfig(
    filename=log_filename,          # 日志文件名
    level=logging.INFO,          # 日志等级
    format='%(asctime)s [%(levelname)s] %(message)s',  # 格式
)

# 创建保存器
# commandSaver = CommandToRosbag(bag_dir="../datas/bags/baseline1")   
# map_saver = MapINfoToMarkers(bag_dir="../datas/bags/baseline1", frame_id="world")
# pathSaver = PathToRosbag(bag_dir="../datas/bags/baseline1") 
# 保存完整轨迹

update_map_freq = opt_args["update_map_freq"]
map_merge_freq = opt_args["map_merge_freq"] 
init_state = init_state_old
sol_trajs = sol_trajs_old
robot_number = opt_args["robot_number"]
tsteps = opt_args["tsteps"]
# 双机器人初始化
start_pos = opt_args["x0"]
end_pos = opt_args["xf"]
# 假设你想创建包含 n 个 al_iLQR 结果的数组
n = robot_number  # 例如，创建 10 个结果
traj_solver = [al_iLQR(
    args=opt_args,
    objective=loss_traj_single,
    dynamics=robot_model_single,
    inequality=ineq_constr_single,
    equality=eq_constr_single,
    target_distr = robot_distr[_id].evals,
    robot_id = _id,
) for _id in range(n)]
decay_type = 'nodecay'
init_dual = True
save_path = '../figures/ex2-baseline1.png'
map_merge_cnt = 0
involved_robots = list(range(robot_number))
involved_robots = set(involved_robots)
accumulated_time = 0
be_num = 1
global_metric = {
        'time': [],
        'values': [],
}
current_time = None
robot_pair = []
logging.info('ex2-baseline1')

while True:
    # sol_trajs, to_remove = optimize_trajs(involved_robots, sol_trajs, betas, traj_solver, init_state, init_dual)
    to_remove = set()
    for r_id in involved_robots:
        sol_trajs[r_id], conv = traj_solver[r_id].solve(
        x0=init_state[r_id],
        init_sol=sol_trajs[r_id],      # 如果 solver 会修改它，建议 deep copy
        beta=betas[r_id],
        init_dual=init_dual,
        max_iter=1000,
        if_print=False,
        r_eps = 0.02,
        loss_eps = 1e-6)
        if conv:
            to_remove.add(r_id)
    
    involved_robots -= to_remove
    clear_output(wait=True)                   # 清除上一次输出，动态刷新
    # init_dual = False
    # =================================== 更新地图 ==============================================================
    current_time, accumulated_time = update_accumulated_time(current_time, accumulated_time, be_num)
    map_merge_cnt += current_time
    if map_merge_cnt >= map_merge_freq:
        accumulated_time -= (map_merge_cnt - map_merge_freq)
        current_time -= (map_merge_cnt - map_merge_freq)
        map_merge_cnt = 0
        if accumulated_time >= tsteps:
            append_metric(global_metric, loss_compare_single(sol_trajs, target_distr.evals, current_time))
            break
        for r_id in range(robot_number):
            u_t = sol_trajs[r_id]['x'][current_time, 0:2]
    # ================================ 为replan准备 ============================================================== 
    if accumulated_time == update_map_freq * be_num:
        append_metric(global_metric, loss_compare_single(sol_trajs, target_distr.evals, current_time))
        if accumulated_time >= tsteps:
            break
        logging.info("update map")
        # map_saver.save_heatmap_to_bag(target_distr.evals[1], target_distr.evals[0], 0.12, update_map_freq)
        # target_distr.update_map(accumulated_time, "perturb", 'write')
        target_distr.update_map(accumulated_time, "perturb", 'read')
        be_num += 1

    sol_trajs, connected_pairs = exchange_info_old(sol_trajs, robot_pair, current_time, robot_distr)
    betas = update_beta_single(sol_trajs, decay_type)
    involved_robots = set(list(range(robot_number)))
    init_state = [traj['x'][0, :] for traj in sol_trajs]  
    init_dual = True
    logging.warning(f"connect time:{current_time}, accumulated time:{accumulated_time}")
    current_time = None
plot_trajs_old(start_pos, end_pos, sol_trajs, betas, robot_distr, save_path)  

def save_ergodic_metrics_to_yaml(global_metric, decay_type, map_id, file_path='experiment2.yaml'):
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

save_ergodic_metrics_to_yaml(global_metric, "baseline1", map_id=args.map_id)






