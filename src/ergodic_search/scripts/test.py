import os
os.environ["JAX_ENABLE_X64"] = "True"
import sys 
sys.path.append('..')
# 增广拉格朗日法优化器
from multiRobots_lib.multi_solver import al_iLQR 
from multiRobots_lib.augument_lagrange_func import loss_traj_multi, eq_constr, ineq_constr_multi, loss_compare_multi
from multiRobots_lib.data_collect import save_ergodic_metrics_to_excel, append_metric, multi_traj_to_rosbag
from multiRobots_lib.plot_utils import plot_trajs
from multiRobots_lib.tools import find_first_connected, exchange_info, optimize_trajs, update_accumulated_time
from multiRobots_lib.decay_utils import update_beta
from multiRobots_lib.data2bag import CommandToRosbag, PathToRosbag
from IPython.display import clear_output
import logging
from jax import jit
from functools import partial
from multiRobots_lib.hyper_params import (
    opt_args,
    target_distr,
    robot_distr,
    robot_number,
    init_state,
    sol_trajs,
    betas,
    robot_model_single,
)
logging.basicConfig(
    filename='log/app.log',          # 日志文件名
    level=logging.INFO,          # 日志等级
    format='%(asctime)s [%(levelname)s] %(message)s',  # 格式
)
update_map_freq = opt_args["update_map_freq"]
map_merge_freq = opt_args["map_merge_freq"] 
state_dim = robot_model_single.nx
robot_number = opt_args["robot_number"]
tsteps = opt_args["tsteps"]
# saver = TrajectoryToRosbag(bag_dir="../datas/bags/prob_connect")    
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
save_path = '../figures/probabilistic_connection.png'
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
logging.info('prob connect')
# print(robot_distr[0].evals[0])



# 模拟数据（替换成你的 robot_distr）
# probs = robot_distr[0].evals[0]   # (N, 2)
# points = robot_distr[0].evals[1]   # (N,)

saver = CommandToRosbag(bag_dir="../datas/bags/prob_connect")  
saver = PathToRosbag(bag_dir="../datas/bags/prob_connect")
print("save path start")
saver.save_path_to_bag(sol_trajs[0]['x'][:,:2], bag_filename="path.bag")

import os
print("Current working directory:", os.getcwd())
print("Intended bag_dir (raw):", "../datas/bags/prob_connect")
bag_dir = os.path.abspath("../datas/bags/prob_connect")
print("Resolved bag_dir:", bag_dir)
print("Parent dir exists?", os.path.exists(os.path.dirname(bag_dir)))