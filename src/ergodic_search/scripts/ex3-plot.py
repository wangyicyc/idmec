import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1️⃣ 共同数据加载
# ==============================
file_path = 'experiment3.xlsx'
df = pd.read_excel(file_path, sheet_name='ergodic_metric')

metric_cols = sorted(
    [col for col in df.columns if col.startswith('metric_')],
    key=lambda x: int(x.split('_')[1])
)
steps = list(range(len(metric_cols)))

methods = ['baseline1', 'baseline2', 'baseline3', 'baseline4', 'baseline5', 'baseline6', 'method']
colors = {
    'baseline1': "#F221B3",
    'baseline2': "#7A3CDD",
    'baseline3': "#3F42DB",
    'baseline4': "#2DC649",
    'baseline5': "#DFF706",
    'baseline6': "#F1B900",
    'method': "#40FFE2"
}
markers = ['o', 's', '^', 'D', '*', 'X', 'P']


# ==============================
# 2️⃣ 图1：原始折线图（趋势图）
# ==============================
# 保存原始 rcParams
original_rc = plt.rcParams.copy()

# 应用第一套样式（大图、粗线、大字体）
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 0.5,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

plt.figure(figsize=(8.5, 5.2))

for i, method in enumerate(methods):
    row = df[df['type'] == method]
    if row.empty:
        print(f"Warning: '{method}' not found in data.")
        continue
    values = row[metric_cols].iloc[0].values.astype(float)
    
    # 折线（无 marker）
    plt.plot(
        steps, values,
        color=colors[method],
        linewidth=2.5,
        label=method,
        marker=None
    )
    
    # 半透明 marker
    plt.scatter(
        steps, values,
        color=colors[method],
        marker=markers[i],
        s=500,
        alpha=0.8,
        edgecolors='white',
        linewidths=0.8,
        zorder=5
    )

plt.xlabel('Step')
# plt.ylabel('Ergodic Metric')
# plt.title('Ergodic Metric Comparison Across Methods', fontweight='bold')
plt.legend(frameon=False)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(steps)
plt.tight_layout()

save_path_pdf = 'experiment3.png'
plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ 折线图已保存至: {save_path_pdf}")

# 恢复原始设置（可选，但为安全起见）
plt.rcParams.update(original_rc)


# ==============================
# 3️⃣ 图2：柱状图（均值 ± 标准差）
# ==============================
# 应用第二套紧凑出版级样式（小图、细线、Arial 风格）
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 12,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.linewidth": 0.5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,
})

# 计算每个方法的均值和标准差（跨 steps）
means, stds, valid_methods, valid_colors_list = [], [], [], []
for method in methods:
    row = df[df['type'] == method]
    if not row.empty:
        vals = row[metric_cols].iloc[0].astype(float)
        means.append(vals.mean())
        stds.append(vals.std())
        valid_methods.append(method)
        valid_colors_list.append(colors[method])

x = np.arange(len(valid_methods)) * 0.5
fig, ax = plt.subplots(figsize=(3.6, 2.3))

# 绘制柱状图 + 误差线
bars = ax.bar(
    x, means,
    yerr=stds,
    width=0.4,
    color=valid_colors_list,
    capsize=5,                     # 稍微加大帽宽，更清晰
    error_kw={
        'elinewidth': 2.0,         # 误差线稍粗
        'capthick': 2.0,           # 帽子也稍粗
        'ecolor': 'black',         # 误差线统一为黑色（更专业）
        'alpha': 0.8               # 可选：轻微透明避免压过柱子
    },
    zorder=3                       # 确保柱子在网格线上方
)

# 调整坐标轴
ax.set_xticks(x)
ax.set_xticklabels(valid_methods, rotation=30, ha='right')
# ax.set_title('Performance Stability Across Steps')

# 自动调整 y 轴范围，避免误差线上端被裁剪
y_max = max(np.array(means) + np.array(stds))
ax.set_ylim(bottom=0, top=y_max * 1.15)

plt.tight_layout(pad=0.3)

bar_save_path = 'experiment3_mean_std.png'
plt.savefig(bar_save_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ 柱状图（均值±标准差）已保存至: {bar_save_path}")