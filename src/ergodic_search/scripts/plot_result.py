import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def plot_hist(hist_data, save_path, title, baseline=None):
    # 解包数据：每组4个值，共4组
    a1, b1, c1, d1, e1 = hist_data[0]  # group 1
    a2, b2, c2, d2, e2 = hist_data[1]
    a3, b3, c3, d3, e3 = hist_data[2]
    fig, ax = plt.subplots(figsize=(12, 8))
    n_groups = 3  # 每组柱子的数量
    index = np.arange(5)  # case数量
    total_width = 0.8
    
    bar_width = total_width / n_groups  # 每个柱子的宽度
    spacing = 0.02  # 组间微小间隔（可选）

    # 计算每组柱子的中心位置（均匀分布在每个机器人位置周围）
    offsets = np.linspace(-total_width/2 + bar_width/2, total_width/2 - bar_width/2, n_groups)
    positions = [index + offset for offset in offsets]

    # 定义颜色和标签（请根据你的实际策略修改标签）
    colors = ['skyblue', 'salmon', 'lightgreen']
    # labels = ['no exchange', 'no decay', 'linear decay', 'type1 decay']  # 示例标签
    labels = ['baseline1','no prob_connect', 'prob_connect']  # 示例标签
    # 绘制四组柱子
    bars = []
    data_groups = [
        [a1, b1, c1, d1, e1],
        [a2, b2, c2, d2, e2],
        [a3, b3, c3, d3, e3],
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
    ax.set_xticklabels(['case 1', 'case 2', 'case 3', 'case 4', 'case 5'], fontsize=10)

    # 图例
    ax.legend(loc='best', fontsize=15)  # 建议不要用20，太大了

    # 添加数值标签
    for bar_group in bars:
        ax.bar_label(bar_group, padding=3, fontsize=9, fmt='%.3f')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



# 初始化 datas 为 3x10 的结构：3 行（三种类型），10 列（case1~case10）
datas = [[0.0 for _ in range(5)] for _ in range(3)]
# 定义要提取的类型顺序
types_order = ['baseline1', 'no_prob_connect', 'prob_connect']
# 遍历 case1 到 case10
for i in range(5):
    folder_name = f"../case{i+1}"
    folder_path = Path(folder_name)
    # 查找该文件夹下的 Excel 文件（假设扩展名为 .xlsx）
    excel_files = list(folder_path.glob("*.xlsx"))
    if not excel_files:
        raise FileNotFoundError(f"No Excel file found in {folder_name}")
    if len(excel_files) > 1:
        print(f"Warning: Multiple Excel files in {folder_name}, using the first one.")

    excel_file = excel_files[0]
    df = pd.read_excel(excel_file)

    # 提取三种类型的 metric_mean
    for row_idx, t in enumerate(types_order):
        matched = df[df['type'] == t]
        if not matched.empty:
            datas[row_idx][i] = matched['metric_mean'].iloc[0]
        else:
            datas[row_idx][i] = None  # 或设为 0.0 / NaN 等
            print(f"Warning: '{t}' not found in {excel_file}")

# 读取数据并准备datas
title = "4 peak and 4x4 map"
save_path = "./hist_means.png"
# 正确调用函数：传递二维的datas
plot_hist(datas, save_path, title)