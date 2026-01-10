# %%
import sys 
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import FuncFormatter
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

# 自定义颜色列表，基于给定的RGB值
colors = [
    '#%02x%02x%02x' % (29, 108, 212),   # 根据指定的RGB值生成十六进制颜色代码
    '#%02x%02x%02x' % (255, 170, 0),
    '#%02x%02x%02x' % (1, 160, 100),   
    '#%02x%02x%02x' % (170, 0, 255)
]
def plot_trajs(start_pos, end_pos, sol_trajs, betas, convergence, target_distr, save_path_pattern=None):
    """
    为每个机器人单独绘制轨迹并保存为独立图片。
    
    参数：
        start_pos: 所有机器人的起点（形状为 (n_robots*2,) 或列表）
        end_pos: 所有机器人的终点（形状为 (n_robots*2,) 或列表）
        sol_trajs: 轨迹列表，每个元素是一个字典（包含 'x', 'px', 'y' 等键）
        betas: 每个机器人的 beta 参数列表
        robot_distr: 机器人分布列表，每个元素对应一个 robot 的分布对象
        save_path_pattern: 保存路径模板，例如 "robot_{i}.png"
                           若为 None，则显示图像（不保存）
    """
    n_robots = robot_number
    
    for i in range(n_robots + 1):
        # 创建独立 figure
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='none')
        ax.set_facecolor("#EFEFF4FF")
        
        # 获取网格和 PDF
        grids_x, grids_y = target_distr.get_grids()
        pdf_vals, _ = target_distr.evals
        pdf_vals = pdf_vals.reshape(grids_x.shape)
        
        # 设置坐标轴范围
        x_max = target_distr.domain[0].max()
        y_max = target_distr.domain[1].max()
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        
        # 设置刻度
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.tick_params(axis='both', which='major', labelsize=14)
        # ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '' if y == 0 else f'{int(y)}'))
        
        # 网格线
        ax.grid(True, which='major', linestyle=(0, (5, 5)), linewidth=1.0)
        
        # 绘制背景概率密度
        ax.contourf(
            grids_x, grids_y, pdf_vals,
            levels=200, cmap='Reds', alpha=0.3
        )
        
        # 加粗边框
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_linewidth(2)
        
        # === 绘制当前机器人轨迹 ===
        
        sol_traj = sol_trajs[0]  # 假设所有机器人共享同一个 sol_trajs[0]
        
        if i == n_robots:
            for t in range(robot_number):  # 假设每个轨迹点有4个维度（x, y, vx, vy）
                decayed_alpha = get_Decay_alpha(betas[t])
                # 绘制起点和终点
                # === 起点：LaTeX 实心三角 ▲ ===
                ax.scatter(
                    start_pos[t * 2], start_pos[t * 2 + 1], s=300,  # 与 \bigstar 的 s=200 视觉平衡
                    c=colors[t], marker=r'$\blacktriangle$',
                    zorder=3, edgecolors='white',
                    linewidth=3.0, alpha=0.9
                )
                # === 终点：LaTeX 五角星 ★ ===
                ax.scatter(
                    sol_traj['x'][-1, 4 * t], sol_traj['x'][-1, 4 * t + 1], s=600,  # 稍微调大以匹配视觉重量
                    c=colors[t], marker=r'$\bigstar$',  # LaTeX 填充五角星
                    zorder=5, edgecolors='white',
                    linewidth=3.0, alpha=0.9
                )
                # 当前位置：使用带描边的圆环（空心圆），颜色与机器人一致，但加白色高亮边框
                ax.scatter(
                    sol_traj['x'][0, 4 * t], sol_traj['x'][0, 4 * t + 1],
                    s=200,
                    facecolors='none',               # 空心（透明填充）
                    edgecolors=colors[t],            # 边框颜色 = 机器人主色
                    linewidth=4.0,
                    zorder=6,
                    alpha=1.0,
                )
                ax.scatter(sol_traj['x'][:, 4 * t], sol_traj['x'][:, 4 * t + 1], 
                    c = colors[t], s=70, alpha=decayed_alpha['x'][:, t], 
                    marker='^', edgecolors='none', zorder=5)
                
                # 过去轨迹（如果存在）
                if sol_traj['px'][t].shape[0] > 0:
                    ax.scatter(
                        sol_traj['px'][t][:, 0], sol_traj['px'][t][:, 1],
                        c=colors[t], s=60, alpha=decayed_alpha['px'][t],
                        marker='^', edgecolors='none', zorder=4
                    )
                # ax.set_title(f'Robot Joint trajectory', fontsize=30, fontweight='bold')
        
        
        else:
            decayed_alpha = get_Decay_alpha(betas[i])
            # 起点：实心三角
            ax.scatter(
                start_pos[i * 2], start_pos[i * 2 + 1], s=300,
                c=colors[i], marker=r'$\blacktriangle$',
                zorder=3, edgecolors='white', linewidth=3.0, alpha=0.9,
                label='start point'
            )
            
            # 终点：五角星
            ax.scatter(
                sol_traj['x'][-1, 4 * i], sol_traj['x'][-1, 4 * i + 1], s=600,
                c=colors[i], marker=r'$\bigstar$',
                zorder=5, edgecolors='white', linewidth=3.0, alpha=0.9,
                label='end point'
            )

            # 当前位置：使用带描边的圆环（空心圆），颜色与机器人一致，但加白色高亮边框
            ax.scatter(
                sol_traj['x'][0, 4 * i], sol_traj['x'][0, 4 * i + 1],
                s=200,
                facecolors='none',               # 空心（透明填充）
                edgecolors=colors[i],            # 边框颜色 = 机器人主色
                linewidth=4.0,
                zorder=6,
                alpha=1.0,
                label='robot position'
            )    


            # 未来轨迹
            ax.scatter(
                sol_traj['x'][:, 4 * i], sol_traj['x'][:, 4 * i + 1],
                c=colors[i], s=70, alpha=decayed_alpha['x'][:, i],
                marker='^', edgecolors='none', zorder=5
            )
            
            # 过去轨迹（如果存在）
            if sol_traj['px'][i].shape[0] > 0:
                ax.scatter(
                    sol_traj['px'][i][:, 0], sol_traj['px'][i][:, 1],
                    c=colors[i], s=60, alpha=decayed_alpha['px'][i],
                    marker='^', edgecolors='none', zorder=4
                )
        
            # 标题
            
            # ax.set_title(f'Robot {i}', fontsize=30, fontweight='bold')
            ax.legend(loc='best', fontsize=14)
        
        # 保存或显示
        if save_path_pattern is not None:
            save_path_i = save_path_pattern.format(i=i)
            plt.savefig(save_path_i, dpi=300, bbox_inches='tight')
            plt.close(fig)