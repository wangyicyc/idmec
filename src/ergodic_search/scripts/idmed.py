import os
os.environ["JAX_ENABLE_X64"] = "True"
import sys 
sys.path.append('..')
# 增广拉格朗日法优化器
from multiRobots_lib.multi_solver import al_iLQR 
from multiRobots_lib.augument_lagrange_func import loss_traj_multi, eq_constr, ineq_constr_multi, loss_compare_multi
from multiRobots_lib.data_collect import append_metric
from multiRobots_lib.plot_utils import plot_trajs
# from multiRobots_lib.plot_utils_paper import plot_trajs
from multiRobots_lib.tools import find_first_connected, exchange_info, update_accumulated_time
from multiRobots_lib.decay_utils import update_beta

from IPython.display import clear_output
import logging
from jax import jit
from functools import partial
from multiRobots_lib.hyper_params import (
    opt_args,
    target_distr,
    robot_distr,
    init_state,
    sol_trajs,
    multi_betas,
    robot_model_single,
)
def prepare_experiment():
    logging.basicConfig(
        filename='datas/logs/app.log',          # 日志文件名
        level=logging.INFO,          # 日志等级
        format='%(asctime)s [%(levelname)s] %(message)s',  # 格式
    )
    update_map_freq = opt_args["update_map_freq"]
    map_merge_freq = opt_args["map_merge_freq"] 
    state_dim = robot_model_single.nx
    robot_number = opt_args["robot_number"]
    tsteps = opt_args["tsteps"]
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
    init_dual = True
    save_path = '../figures/my_strategy.png'
    involved_robots = list(range(robot_number))
    involved_robots = set(involved_robots)
    accumulated_time = 0
    be_num = 1
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

    return {
        "update_map_freq": update_map_freq,
        "map_merge_freq": map_merge_freq,
        "state_dim": state_dim,
        "robot_number": robot_number,
        "tsteps": tsteps,
        "init_state": init_state,
        "sol_trajs": sol_trajs,
        "multi_betas": multi_betas,
        "start_pos": start_pos,
        "end_pos": end_pos,
        "traj_solver": traj_solver,
        "traj_warmUp": traj_warmUp,
        "connection_threshold": connection_threshold,
        "map_merge_cnt": map_merge_cnt,
        "decay_type": decay_type,
        "init_dual": init_dual,
        "save_path": save_path,
        "involved_robots": involved_robots,
        "accumulated_time": accumulated_time,
        "be_num": be_num,
        "global_metric": global_metric,
        "last_exchange_time": last_exchange_time,
        "warm_up": warm_up,
    }


def run_experiment(context):
    while True:
        if context["warm_up"] == True:
            for r_id in context["involved_robots"]:
                context["sol_trajs"][r_id], conv = context["traj_warmUp"][r_id].solve(
                x0=context["init_state"][r_id],
                init_sol=context["sol_trajs"][r_id],
                beta=context["multi_betas"][r_id],
                init_dual=context["init_dual"],
                max_iter=150,
                if_print=False,
                r_eps = 0.03,
                loss_eps = 1e-6)
        else:
            for r_id in context["involved_robots"]:
                context["sol_trajs"][r_id], conv = context["traj_solver"][r_id].solve(
                x0=context["init_state"][r_id],
                init_sol=context["sol_trajs"][r_id],
                beta=context["multi_betas"][r_id],
                init_dual=context["init_dual"],
                max_iter=200,
                if_print=False,
                r_eps = 0.03,
                loss_eps = 1e-6)
        clear_output(wait=True)
        context["init_dual"] = False

        if context["warm_up"] == True:
            context["warm_up"] = False
            context["init_dual"] = True
            logging.info("have warm up")
            continue
        else:
            context["warm_up"] = True
        current_time, robot_pair = find_first_connected(context["sol_trajs"], context["connection_threshold"], context["last_exchange_time"])
        current_time, context["accumulated_time"] = update_accumulated_time(current_time, context["accumulated_time"], context["be_num"])
        context["map_merge_cnt"] += current_time
        if context["map_merge_cnt"] >= context["map_merge_freq"]:
            context["accumulated_time"] -= (context["map_merge_cnt"] - context["map_merge_freq"])
            current_time -= (context["map_merge_cnt"] - context["map_merge_freq"])
            context["map_merge_cnt"] = 0
            if context["accumulated_time"] >= context["tsteps"]:
                append_metric(context["global_metric"], loss_compare_multi(context["sol_trajs"], target_distr.evals, current_time))
                break
            robot_pair = []

            for r_id in range(context["robot_number"]):
                q_t = context["sol_trajs"][r_id]['x'][current_time, r_id * context["state_dim"]: r_id * context["state_dim"] + 2]
                robot_distr[r_id].bayes_filter_reset(target_distr.evals[0], q_t)
                context["traj_solver"][r_id].update_distribution(robot_distr[r_id].evals)
                context["traj_warmUp"][r_id].update_distribution(robot_distr[r_id].evals)
                logging.info("merge map")
        if context["accumulated_time"] == context["update_map_freq"] * context["be_num"]:
            append_metric(context["global_metric"], loss_compare_multi(context["sol_trajs"], target_distr.evals, current_time))
            logging.info("update map")
            robot_pair = []
            target_distr.update_map(context["accumulated_time"], "reset", "read")
            context["be_num"] += 1
            if context["accumulated_time"] >= context["tsteps"]:
                break

        for i, j in robot_pair:
            context["last_exchange_time"][(i, j)] = context["accumulated_time"]
            context["last_exchange_time"][(j, i)] = context["accumulated_time"]
        context["sol_trajs"], connected_pairs = exchange_info(context["sol_trajs"], robot_pair, current_time, robot_distr, context["last_exchange_time"])
        context["multi_betas"] = update_beta(context["sol_trajs"], context["decay_type"], context["last_exchange_time"], context["accumulated_time"])
        if connected_pairs:
            context["involved_robots"] = connected_pairs
        else:
            context["involved_robots"] = set(list(range(context["robot_number"])))
        context["init_state"] = [traj['x'][0, :] for traj in context["sol_trajs"]]
        context["init_dual"] = True
        logging.warning(f"connect time:{current_time}, accumulated time:{context['accumulated_time']}, and the robot pair:{robot_pair}")
    plot_trajs(context["start_pos"], context["end_pos"], context["sol_trajs"], context["multi_betas"], robot_distr, context["save_path"])

def main():
    # 准备实验相关参数
    context = prepare_experiment()
    run_experiment(context)


if __name__ == "__main__":
    main()
    
