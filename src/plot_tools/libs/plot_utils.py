# %%
import sys 
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap

# 自定义颜色列表，基于给定的RGB值
colors = [
    '#1D6CD4',  # 蓝色 (Blue)      —— RGB(29, 108, 212)
    '#FFAA00',  # 橙色 (Orange)    —— RGB(255, 170, 0)
    '#01A064',  # 翡翠绿 (Teal)    —— RGB(1, 160, 100)
    '#AA00FF',  # 紫色 (Purple)    —— RGB(170, 0, 255)
]
def plot_trajs(color, robot_distr, save_path=None):
    """
    动态绘制多个机器人的轨迹
    参数：
        start_pos: 所有机器人的起点（形状为 (n_robots*2,) 或列表）
        end_pos: 所有机器人的终点（形状为 (n_robots*2,) 或列表）
        sol_traj: 轨迹列表，每个元素是一个字典（包含 'x', 'px', 'y' 等键）
        beta: 每个机器人的 beta 参数列表
    """
    fig = plt.figure(figsize=(7.6 * 1, 6), facecolor='none')  # 宽度根据机器人数量调整
    # 动态创建子图
    axes = []
    grids_x, grids_y = robot_distr.get_grids()


    ax = fig.add_subplot(1, 1, 1)  # 1行，1列，第i+1个子图
    ax.set_facecolor("#FFFFFFFF")  # 使用十六进制颜色代码代表亮灰色
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='both', which='major', labelsize=14)  # 根据需要调整labelsize值
        # 如果你想要在图内部显示刻度线，可以使用grid方法
    ax.grid(True, which='major', linestyle=(0, (5, 5)), linewidth=1.0)
    ax.set_xlim(0, robot_distr.domain[0].max())
    ax.set_ylim(0, robot_distr.domain[1].max())
    pdf_vals, _ = robot_distr.evals
    ax.contourf(
        grids_x, grids_y, pdf_vals.reshape(grids_x.shape), 
        levels=200, cmap=color, alpha=0.3
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '' if y == 0 else f'{int(y)}'))
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(2)  # 加粗边框

    # === 终点：LaTeX 五角星 ★ ===
    ax.scatter(
        2.0, 2.0, s=1400,  # 稍微调大以匹配视觉重量
        c=colors[0], marker=r'$\bigstar$',  # LaTeX 填充五角星
        zorder=5, edgecolors='white',
        linewidth=3.0, alpha=0.9, label='robot 1'
        )
    # ax.scatter(
    #     1.0, 2.5, s=1400,  # 稍微调大以匹配视觉重量
    #     c=colors[2], marker=r'$\bigstar$',  # LaTeX 填充五角星
    #     zorder=5, edgecolors='white',
    #     linewidth=3.0, alpha=0.9, label='robot 2'
    #     ) 
    ax.legend(loc='lower left', fontsize=24)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

# 使用LinearSegmentedColormap生成渐变色板
# 定义起始和结束颜色（放在函数外，避免重复创建）
color_start = "white"
color_end = "#2de124"
cmap_custom = LinearSegmentedColormap.from_list("custom_single", [color_start, color_end])
color_start = "white"
color_end = "#3373e1"
cmap_custom = LinearSegmentedColormap.from_list("custom_single", [color_start, color_end])

def plot_weight(robot_distr, save_path=None):
    """
    绘制单张权重分布图（无轨迹），仅用单色渐变表示概率密度。
    """
    fig = plt.figure(figsize=(7.6 * 1, 6), facecolor='none')
    grids_x, grids_y = robot_distr.get_grids()
    
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor("#FFFFFFFF")
    
    # 设置坐标轴范围
    x_max = robot_distr.domain[0].max()
    y_max = robot_distr.domain[1].max()
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # 网格线（间断样式）
    # ax.grid(True, which='major', linestyle=(0, (5, 5)), linewidth=1.0)
    
    # 获取并重塑 PDF 值
    pdf_vals, _ = robot_distr.evals
    pdf_vals = pdf_vals.reshape(grids_x.shape)
    
    # 避免颜色过深：截断上尾 + 使用 pcolormesh
    vmax = np.percentile(pdf_vals, 95)  # 忽略最大的 5%
    
    # 使用 pcolormesh 替代 contourf（更适合纯颜色填充）
    mesh = ax.pcolormesh(
        grids_x, grids_y, pdf_vals,
        cmap=cmap_custom,
        vmin=0,
        vmax=vmax,
        alpha=0.3,
        shading='auto'  # 自动处理网格对齐
    )
    
    # 隐藏 y 轴的 0 标签，与 x 轴共用原点
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '' if y == 0 else f'{int(y)}'))
    
    # 加粗边框
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(2)
    

    # === 终点：LaTeX 五角星 ★ ===
    ax.scatter(
        2.0, 2.0, s=1400,  # 稍微调大以匹配视觉重量
        c=colors[0], marker=r'$\bigstar$',  # LaTeX 填充五角星
        zorder=5, edgecolors='white',
        linewidth=3.0, alpha=0.9, label='robot 1'
        )
    # ax.scatter(
    #     1.0, 2.5, s=1400,  # 稍微调大以匹配视觉重量
    #     c=colors[2], marker=r'$\bigstar$',  # LaTeX 填充五角星
    #     zorder=5, edgecolors='white',
    #     linewidth=3.0, alpha=0.9, label='robot 2'
    #     )
    ax.legend(loc='lower left', fontsize=24)
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)