
import os
os.environ["JAX_ENABLE_X64"] = "True"
import sys 
sys.path.append('..')
# 增广拉格朗日法优化器
from multiRobots_lib.hyper_params import (
    target_distr,
)
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'，需安装对应 GUI 库
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
white_to_black = LinearSegmentedColormap.from_list(
    "white_to_black", ["#FFFFFF", "#000000"], N=256
)

def plot_trajs(_distr, save_path=None):
    fig = plt.figure(figsize=(8 * 1, 6), facecolor='none')  # 宽度根据机器人数量调整
    # 动态创建子图
    axes = []
    grids_x, grids_y = _distr.get_grids()

    ax = fig.add_subplot(1, 1, 1)  # 1行，robot_number列，第i+1个子图
    ax.set_facecolor("#EFEFF4FF")  # 使用十六进制颜色代码代表亮灰色
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xticks(np.arange(0, _distr.domain[0].max() + 1e-8, step=1.0)) # 根据需要调整步长
    # ax.set_yticks(np.arange(0, _distr.domain[1].max() + 1e-8, step=1.0)) # 根据需要调整步长
    # 如果你想要在图内部显示刻度线，可以使用grid方法
    # ax.grid(True, which='major', linestyle=(0, (5, 5)), linewidth=1.0)
    ax.set_xlim(0, _distr.domain[0].max())
    ax.set_ylim(0, _distr.domain[1].max())
    pdf_vals, _ = _distr.evals
    ax.contourf(
        grids_x, grids_y, pdf_vals.reshape(grids_x.shape), 
        levels=200, cmap=white_to_black, alpha=0.3
    )
    # ax.margins(0)  # 禁用数据周围的默认空白
    # # 关键：防止刻度线和标签超出边界
    # ax.tick_params(axis='both', which='both', 
    #             direction='in',  # 或 'in'，避免向外延伸
    #             labelsize=16,
    #             pad=6,              # 控制标签与刻度线距离（可调）
    #             length=4)           # 刻度线长度（可选）
    # dx = 0.008 * _distr.domain[0].max()
    # dy = 0.012 * _distr.domain[1].max()

    # # 在 ( -dx, -dy ) 位置添加文本，但需确保在 axes 坐标系中可见
    # # 使用 annotation 或 text，推荐用 annotation 支持坐标变换
    # ax.annotate(
    #     '0', 
    #     xy=(0, 0),                     # 指向原点
    #     xytext=(-dx, -dy),             # 文本位置（左下方）
    #     textcoords='data',             # 使用数据坐标
    #     fontsize=16,
    #     ha='right',                    # 水平对齐：文本右对齐 → 靠近原点
    #     va='top',                      # 垂直对齐：文本顶部对齐 → 靠近原点
    #     color='black'
    # )
    # ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '' if y == 0 else f'{int(y)}'))
    # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '' if x == 0 else f'{int(x)}'))
    axes.append(ax) 
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()



plot_trajs(target_distr, save_path='../figures/real_map_0.png')

# target_distr.update_map(0, "perturb", 'write')
# plot_trajs(target_distr, save_path='../figures/real_map_1.png')
# target_distr.update_map(0, "perturb", 'write')
# plot_trajs(target_distr, save_path='../figures/real_map_2.png')
# target_distr.update_map(0, "perturb", 'write')
# plot_trajs(target_distr, save_path='../figures/real_map_3.png')
# target_distr.update_map(0, "perturb", 'write')
# plot_trajs(target_distr, save_path='../figures/real_map_4.png')


target_distr.update_map(0, "perturb", 'read')
plot_trajs(target_distr, save_path='../figures/real_map_1.png')
target_distr.update_map(1, "perturb", 'read')
plot_trajs(target_distr, save_path='../figures/real_map_2.png')
target_distr.update_map(2, "perturb", 'read')
plot_trajs(target_distr, save_path='../figures/real_map_3.png')
target_distr.update_map(3, "perturb", 'read')
plot_trajs(target_distr, save_path='../figures/real_map_4.png')