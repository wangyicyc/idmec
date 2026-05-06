import os
os.environ["JAX_ENABLE_X64"] = "True"
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
# 增广拉格朗日法优化器
from ergodic_planning.multi_solver import al_iLQR 
from ergodic_planning.augument_lagrange_func import loss_traj_multi, eq_constr, ineq_constr_multi
from utils.plot_utils import plot_trajs
from ergodic_planning.tools import find_first_connected, exchange_info, update_accumulated_time
from ergodic_planning.decay_utils import update_beta
from experiment.io import ExperimentContext, ExperimentOutput

try:
    from IPython.display import clear_output
except ModuleNotFoundError:
    def clear_output(*args, **kwargs):
        return None

import logging
from jax import jit
from functools import partial
from experiment.config import (
    opt_args,
    target_distr,
    robot_distr,
    init_state,
    sol_trajs,
    multi_betas,
    robot_model_single,
)


def prepare_experiment():
    log_dir = PACKAGE_ROOT / "datas" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / 'app.log',          # 日志文件名
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
    jit_ineq_constr_multi = jit(partial(ineq_constr_multi, warm_up=False))
    warm_up_ineq_constr = jit(partial(ineq_constr_multi, warm_up=True))
    traj_solver = []
    traj_warmup = []
    for _id in range(robot_number):
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
        traj_warmup.append(warmup)
    connection_threshold = opt_args["connect_threshold"]
    connection_threshold = connection_threshold**2
    map_merge_cnt = 0
    decay_type = 'linear'
    init_dual = True
    save_path = str(
        PACKAGE_ROOT / 'datas' / 'results' / 'my_strategy' / 'figures' / 'my_strategy.png'
    )
    involved_robots = set(range(robot_number))
    accumulated_time = 0
    be_num = 1
    last_exchange_time = {}
    output = ExperimentOutput(
        output_mode=opt_args["output_mode"],
        bag_dir=PACKAGE_ROOT / opt_args["output_bag_dir"],
        output_topic=opt_args["output_topic"],
        dt=opt_args["dt"],
        robot_number=robot_number,
        state_dim=state_dim,
    )
    output.setup()
    # 迭代优化并动态可视化轨迹与障碍物分布
    warm_up = True
    # warm_up = False
    logging.info('my_strategy')
    plot_trajs(
        start_pos,
        end_pos,
        sol_trajs,
        multi_betas,
        robot_distr,
        save_path,
    )

    return ExperimentContext(
        update_map_freq=update_map_freq,
        map_merge_freq=map_merge_freq,
        state_dim=state_dim,
        robot_number=robot_number,
        tsteps=tsteps,
        init_state=init_state,
        sol_trajs=sol_trajs,
        multi_betas=multi_betas,
        target_distr=target_distr,
        robot_distr=robot_distr,
        start_pos=start_pos,
        end_pos=end_pos,
        traj_solver=traj_solver,
        traj_warmup=traj_warmup,
        connection_threshold=connection_threshold,
        map_merge_cnt=map_merge_cnt,
        decay_type=decay_type,
        init_dual=init_dual,
        save_path=save_path,
        involved_robots=involved_robots,
        accumulated_time=accumulated_time,
        be_num=be_num,
        last_exchange_time=last_exchange_time,
        warm_up=warm_up,
        output=output,
    )


def solve_multi_traj(context):
    if context.warm_up:
        solver_key = "traj_warmup"
        max_iter = 150
    else:
        solver_key = "traj_solver"
        max_iter = 200

    solver_group = getattr(context, solver_key)
    for r_id in context.involved_robots:
        context.sol_trajs[r_id], _conv = solver_group[r_id].solve(
            x0=context.init_state[r_id],
            init_sol=context.sol_trajs[r_id],
            beta=context.multi_betas[r_id],
            init_dual=context.init_dual,
            max_iter=max_iter,
            if_print=False,
            r_eps=0.03,
            loss_eps=1e-6,
        )


def handle_warmup_transition(context):
    context.init_dual = False
    if context.warm_up:
        context.warm_up = False
        context.init_dual = True
        logging.info("have warm up")
        return True

    context.warm_up = True
    return False


def find_connection_event(context):
    current_time, robot_pair = find_first_connected(
        context.sol_trajs,
        context.connection_threshold,
        context.last_exchange_time,
    )
    current_time, context.accumulated_time = update_accumulated_time(
        current_time,
        context.accumulated_time,
        context.be_num,
    )
    context.map_merge_cnt += current_time
    return current_time, robot_pair


def apply_map_merge_if_needed(context, current_time, robot_pair):
    if context.map_merge_cnt < context.map_merge_freq:
        return current_time, robot_pair, False

    merge_overshoot = context.map_merge_cnt - context.map_merge_freq
    context.accumulated_time -= merge_overshoot
    current_time -= merge_overshoot
    context.map_merge_cnt = 0

    if context.accumulated_time >= context.tsteps:
        context.output.emit_map_snapshot(context)
        return current_time, robot_pair, True

    robot_pair = []
    for r_id in range(context.robot_number):
        q_t = context.sol_trajs[r_id]['x'][
            current_time,
            r_id * context.state_dim: r_id * context.state_dim + 2,
        ]
        context.robot_distr[r_id].bayes_filter_reset(context.target_distr.evals[0], q_t)
        context.traj_solver[r_id].update_distribution(context.robot_distr[r_id].evals)
        context.traj_warmup[r_id].update_distribution(context.robot_distr[r_id].evals)
        logging.info("merge map")

    return current_time, robot_pair, False


def apply_target_update_if_needed(context, current_time, robot_pair):
    if context.accumulated_time != context.update_map_freq * context.be_num:
        return current_time, robot_pair, False

    context.output.emit_map_snapshot(context)
    logging.info("update map")
    robot_pair = []
    # 更新目标地图
    # mode:
    #     "perturb"       —— 在当前means附近随机扰动
    #     "reset"         —— 完全随机生成新的means
    # w_or_r: 写入或读取地图参数
    context.target_distr.update_map(context.accumulated_time, mode="reset", w_or_r="read")
    context.be_num += 1
    return current_time, robot_pair, context.accumulated_time >= context.tsteps


def handle_map_events(context, current_time, robot_pair):
    current_time, robot_pair, done = apply_map_merge_if_needed(
        context,
        current_time,
        robot_pair,
    )
    if done:
        return current_time, robot_pair, True
    return apply_target_update_if_needed(context, current_time, robot_pair)


def handle_robot_exchange(context, current_time, robot_pair):
    for i, j in robot_pair:
        context.last_exchange_time[(i, j)] = context.accumulated_time
        context.last_exchange_time[(j, i)] = context.accumulated_time

    context.sol_trajs, connected_pairs = exchange_info(
        context.sol_trajs,
        robot_pair,
        current_time,
        context.robot_distr,
        context.last_exchange_time,
    )
    context.multi_betas = update_beta(
        context.sol_trajs,
        context.decay_type,
        context.last_exchange_time,
        context.accumulated_time,
    )
    if connected_pairs:
        context.involved_robots = connected_pairs
    else:
        context.involved_robots = set(range(context.robot_number))
    context.init_state = [traj['x'][0, :] for traj in context.sol_trajs]
    context.init_dual = True


def run_experiment(context):
    while True:
        # 求解所有机器人的轨迹
        solve_multi_traj(context)
        clear_output(wait=True)
        # 判断是否需要warm up
        if handle_warmup_transition(context):
            continue
        # 查找机器人通信事件的触发情况
        current_time, robot_pair = find_connection_event(context)
        # 处理地图融合与目标地图更新事件, 同时收紧current_time 和 robot_pair
        current_time, robot_pair, done = handle_map_events(
            context,
            current_time,
            robot_pair,
        )
        context.output.emit_segment(context, current_time)
        if done:
            break
        # 处理机器人信息交换事件
        handle_robot_exchange(context, current_time, robot_pair)
        logging.warning(
            f"connect time:{current_time}, accumulated time:{context.accumulated_time}, "
            f"and the robot pair:{robot_pair}"
        )

    # 绘制轨迹图
    plot_trajs(
        context.start_pos,
        context.end_pos,
        context.sol_trajs,
        context.multi_betas,
        context.robot_distr,
        context.save_path,
    )

def main():
    # 准备实验相关参数
    context = prepare_experiment()
    # 运行实验
    run_experiment(context)


if __name__ == "__main__":
    main()
    
