# %%
import sys 
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(
    filename='../log/app.log',          # 日志文件名
    level=logging.INFO,          # 日志等级
    format='%(asctime)s [%(levelname)s] %(message)s',  # 格式
)
from multiRobots_lib.hyper_params import (
    robot_number,
)
from multiRobots_lib.decay_utils import get_Decay_alpha # 获取权重显示

colors = ['purple', 'blue', 'green', 'orange', 'red', 'brown', 'pink', 'gray']
def plot_trajs(start_pos, end_pos, sol_trajs, betas, convergence, robot_distr, save_path=None):
    """
    动态绘制多个机器人的轨迹
    参数：
        start_pos: 所有机器人的起点（形状为 (n_robots*2,) 或列表）
        end_pos: 所有机器人的终点（形状为 (n_robots*2,) 或列表）
        sol_traj: 轨迹列表，每个元素是一个字典（包含 'x', 'px', 'y' 等键）
        beta: 每个机器人的 beta 参数列表
    """
    fig = plt.figure(figsize=(8 * robot_number, 6), facecolor='none')  # 宽度根据机器人数量调整
    # 动态创建子图
    axes = []
    grids_x, grids_y = robot_distr[0].get_grids()
    for i in range(robot_number):
        ax = fig.add_subplot(1, robot_number, i+1)  # 1行，robot_number列，第i+1个子图
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, robot_distr[0].domain[0].max())
        ax.set_ylim(0, robot_distr[0].domain[1].max())
        pdf_vals, _ = robot_distr[i].evals
        ax.contourf(
            grids_x, grids_y, pdf_vals.reshape(grids_x.shape), 
            levels=50, cmap='Blues', alpha=0.3
        )
        axes.append(ax) 
    # 为每个机器人绘制轨迹
    for i, (ax, sol_traj, beta) in enumerate(zip(
        axes, sol_trajs, betas)):
        decayed_alpha = get_Decay_alpha(beta)  # 假设 alpha 只依赖当前机器人的 beta
        
        # 提取当前机器人的起点和终点（假设 start_pos 和 end_pos 是长度为 n_robots*2 的数组）
        start = start_pos[i * 2 : i * 2 + 2]
        end = end_pos[i * 2 : i * 2 + 2]
        
        # 绘制起点和终点
        ax.plot(start[0], start[1], 'b^', markersize=16, 
                markerfacecolor='none', markeredgewidth=3, markeredgecolor=colors[i], label='start point')
        # ax.plot(end[0], end[1], 'bs', markersize=16, 
        #         markerfacecolor='none', markeredgewidth=3, markeredgecolor=colors[i], 
        #         label=f'target(robot{i})')
        ax.plot(sol_traj['x'][0, 4 * i], sol_traj['x'][0, 4 * i + 1], 
            'k^', markersize=8, markerfacecolor='none', markeredgewidth=4)

        # 绘制轨迹点
        for t in range(robot_number):  # 假设每个轨迹点有4个维度（x, y, vx, vy）
            if t == i:
                ax.scatter(sol_traj['x'][:, 4 * t], sol_traj['x'][:, 4 * t + 1], 
                    c = colors[t], s=50, alpha=decayed_alpha['x'][:, t], 
                    marker='^', edgecolors='none', zorder=3)
                if sol_traj['px'][t].shape[0] > 0: 
                    ax.scatter(sol_traj['px'][t][:, 0], sol_traj['px'][t][:, 1], 
                                c = colors[t], s=40, alpha=decayed_alpha['px'][t], marker='^', 
                                edgecolors='none', zorder=2)
            else:
                ax.scatter(sol_traj['x'][:, 4 * t], sol_traj['x'][:, 4 * t + 1], 
                    c = colors[t], s=10, alpha=decayed_alpha['x'][:, t], 
                    edgecolors='none', zorder=2)
                if sol_traj['px'][t].shape[0] > 0:
                    ax.scatter(sol_traj['px'][t][:, 0], sol_traj['px'][t][:, 1], 
                                c = colors[t], s=10, alpha=decayed_alpha['px'][t], 
                                edgecolors='none', zorder=2)
        ax.set_title(f'Robot {i}', fontsize=30, fontweight='bold')
        # ax.legend(loc='best')
    if convergence:
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_trajs_old(start_pos, end_pos, sol_trajs, betas, convergence, robot_distr, save_path=None):
    """
    动态绘制多个机器人的轨迹
    参数：
        start_pos: 所有机器人的起点（形状为 (n_robots*2,) 或列表）
        end_pos: 所有机器人的终点（形状为 (n_robots*2,) 或列表）
        sol_traj: 轨迹列表，每个元素是一个字典（包含 'x', 'px', 'y' 等键）
        beta: 每个机器人的 beta 参数列表
    """
    n_robots = len(sol_trajs)  # 机器人数量
    fig = plt.figure(figsize=(8 * robot_number, 6), facecolor='none')  # 宽度根据机器人数量调整
    # 动态创建子图
    axes = []
    grids_x, grids_y = robot_distr[0].get_grids()
    for i in range(robot_number):
        ax = fig.add_subplot(1, robot_number, i+1)  # 1行，robot_number列，第i+1个子图
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, robot_distr[0].domain[0].max())
        ax.set_ylim(0, robot_distr[0].domain[1].max())
        pdf_vals, _ = robot_distr[i].evals
        ax.contourf(
            grids_x, grids_y, pdf_vals.reshape(grids_x.shape), 
            levels=50, cmap='Blues', alpha=0.3
        )
        axes.append(ax) 
    # 为每个机器人绘制轨迹
    for i, (ax, sol_traj, beta) in enumerate(zip(
        axes, sol_trajs, betas)):
        decayed_alpha = get_Decay_alpha(beta)  # 假设 alpha 只依赖当前机器人的 beta
        
        # 提取当前机器人的起点和终点（假设 start_pos 和 end_pos 是长度为 n_robots*2 的数组）
        start = start_pos[i * 2 : i * 2 + 2]
        end = end_pos[i * 2 : i * 2 + 2]
        
        # 绘制起点和终点
        ax.plot(start[0], start[1], 'b^', markersize=16, 
                markerfacecolor='none', markeredgewidth=3, markeredgecolor=colors[i])
        # ax.plot(end[0], end[1], 'bs', markersize=16, 
        #         markerfacecolor='none', markeredgewidth=3, markeredgecolor=colors[i], 
        #         label=f'target(robot{i})')
        ax.plot(sol_traj['x'][0, 0], sol_traj['x'][0, 1], 
            'k^', markersize=8, markerfacecolor='none', markeredgewidth=4, label='start point')

        # 绘制轨迹点
        ax.scatter(sol_traj['x'][:, 0], sol_traj['x'][:, 1], 
                   c = colors[i], s=50, alpha=decayed_alpha['x'], 
                   marker='^', edgecolors='none', zorder=3)

        

        for t in range(robot_number):  # 假设每个轨迹点有4个维度（x, y, vx, vy）
            if t == i and sol_traj['px'][t].shape[0] > 0:
                ax.scatter(sol_traj['px'][t][:, 0], sol_traj['px'][t][:, 1], 
                    c = colors[t], s=40, alpha=decayed_alpha['px'][t], marker='^', 
                    edgecolors='none', zorder=2)
                    
            elif sol_traj['px'][t].shape[0] > 0:
                ax.scatter(sol_traj['px'][t][:, 0], sol_traj['px'][t][:, 1], 
                    c = colors[t], s=10, alpha=decayed_alpha['px'][t], 
                    edgecolors='none', zorder=2)
                 
        # 设置标题和图例
        ax.set_title(f'Robot {i}', fontsize=30, fontweight='bold')
        # ax.legend(loc='best')

    # 保存或显示
    if convergence:
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

def plot_line_graph(n_robots, metric, range_type, save_path=None):
    """
    参数:
        n_robots: 机器人数量
        metric_history: 指标历史数据列表，每个元素是一个字典 {'time': [], 'values': []}
        save_path: 图片保存路径（可选）
    """
    # 创建图形
    fig = plt.figure(figsize=(10, 6), facecolor='none')
    ax = fig.add_subplot(1, 1, 1)
    if '_linear' in save_path:
        title = 'linear decay'
    elif '_type1' in save_path:
        title = 'type1'
    elif '_noexchange' in save_path:
        title = 'no_exchange'
    elif '_nodecay' in save_path:
        title = 'no_decay'
    # 设置标题和标签
    ax.set_title(f'Ergodic Metrics({title})', fontsize=20, fontweight='bold')
    ax.set_xlabel('Time', fontsize=16)
    ax.set_ylabel('Metric Value', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3)
    
    # 定义颜色和标记样式
    colors = ['purple', 'blue', 'green', 'orange']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # 初始化变量
    max_iterations = 0
    all_values = []
    has_data = False
    
    # 为每个机器人绘制指标折线
    for i in range(n_robots):
        # 确保索引在范围内
        if i < len(metric):
            robot_data = metric[i]
            iterations = robot_data['time']
            values = robot_data['values']
            
            # 检查是否有数据
            if iterations and values:
                has_data = True
                
                # 更新最大迭代次数
                max_iterations = max(max_iterations, max(iterations) if iterations else 0)
                all_values.extend(values)
                
                # 绘制折线图
                ax.plot(
                    iterations, 
                    values, 
                    color=colors[i % len(colors)], 
                    marker=markers[i % len(markers)], 
                    markersize=3,
                    linestyle='-',
                    linewidth=2.0,
                    label=f'Robot {i}'
                )
    
    # 设置图例
    if has_data:
        # 获取所有标签和句柄
        handles, labels = ax.get_legend_handles_labels()
        
        # 去除重复标签
        unique_labels = []
        unique_handles = []
        for label, handle in zip(labels, handles):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        
        # 添加图例
        ax.legend(unique_handles, unique_labels, fontsize=12, loc='best')
    else:
        # 如果没有数据，添加提示信息
        ax.text(0.5, 0.5, 'No metric data available', 
                fontsize=16, ha='center', va='center',
                transform=ax.transAxes)
    # # 设置坐标轴范围
    if range_type == 'auto':
        if max_iterations > 0:
            ax.set_xlim(-0.5, max_iterations + 0.5)
            if all_values:
                y_min = min(all_values)
                y_max = max(all_values)
            if y_max > 20.0:
                y_max = 20.0
            y_range = y_max - y_min
            padding = 0.1 * y_range if y_range > 0 else 0.1
            ax.set_ylim(y_min - padding, y_max + padding)
    else:
        ax.set_ylim(-0.1, 12)

    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.show()
    plt.close(fig)

def plot_hist(hist_data, save_path, title, baseline=None):
    # 解包数据：每组4个值，共4组
    a1, b1, c1, d1, e1,f1,g1,h1,i1,j1 = hist_data[0]  # group 1
    a2, b2, c2, d2, e2,f2,g2,h2,i2,j2 = hist_data[1]
    a3, b3, c3, d3, e3,f3,g3,h3,i3,j3 = hist_data[2]
    fig, ax = plt.subplots(figsize=(12, 8))
    n_groups = 3  # 每组柱子的数量
    index = np.arange(10)  # case数量
    total_width = 0.8
    
    bar_width = total_width / n_groups  # 每个柱子的宽度
    spacing = 0.02  # 组间微小间隔（可选）

    # 计算每组柱子的中心位置（均匀分布在每个机器人位置周围）
    offsets = np.linspace(-total_width/2 + bar_width/2, total_width/2 - bar_width/2, n_groups)
    positions = [index + offset for offset in offsets]

    # 定义颜色和标签（请根据你的实际策略修改标签）
    colors = ['skyblue', 'salmon', 'lightgreen']
    # labels = ['no exchange', 'no decay', 'linear decay', 'type1 decay']  # 示例标签
    labels = ['no exchange info','no decay', 'linear decay']  # 示例标签
    # 绘制四组柱子
    bars = []
    data_groups = [
        [a1, b1, c1, d1, e1, f1, g1,h1,i1,j1],
        [a2, b2, c2, d2, e2,f2,g2,h2,i2,j2],
        [a3, b3, c3, d3, e3,f3,g3,h3,i3,j3],
    ]

    for i in range(n_groups):
        bar = ax.bar(positions[i], data_groups[i], bar_width,
                     label=labels[i], color=colors[i], edgecolor='black')
        bars.append(bar)

    # 设置标签
    ax.set_xlabel('Robots', fontsize=12)
    ax.set_ylabel('Ergodic Metric', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 设置x轴刻度和标签
    ax.set_xticks(index)
    ax.set_xticklabels(['case 1', 'case 2', 'case 3', 'case 4', 'case 5', 'case 6', 'case 7', 'case 8', 'case 9', 'case 10'], fontsize=10)

    # 图例
    ax.legend(loc='best', fontsize=20)  # 建议不要用20，太大了

    # 添加数值标签
    for bar_group in bars:
        ax.bar_label(bar_group, padding=3, fontsize=9, fmt='%.3f')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

