import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1️⃣ 共同数据加载
# ==============================
file_path = 'experiment2.xlsx'
df = pd.read_excel(file_path, sheet_name='ergodic_metric')

metric_cols = sorted(
    [col for col in df.columns if col.startswith('metric_')],
    key=lambda x: int(x.split('_')[1])
)
steps = list(range(len(metric_cols)))

methods = ['baseline1', 'baseline2', 'baseline3']
colors = {
    'baseline1': '#4E79A7',
    'baseline2': '#F28E2B',
    'baseline3': '#E15759',
    'baseline4': '#76B7B2',
    'baseline5': '#B07296',
    'baseline6': '#9C755F',
    'method': '#FF9F40'
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
    "axes.linewidth": 2.5,
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
plt.title('Ergodic Metric Comparison Across Methods', fontweight='bold')
plt.legend(frameon=False)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(steps)
plt.tight_layout()

save_path_pdf = 'experiment2.pdf'
plt.savefig(save_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
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
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.linewidth": 1.0,
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

x = np.arange(len(valid_methods))
plt.figure(figsize=(3.6, 2.3))
plt.bar(
    x, means, yerr=stds,
    color=valid_colors_list,
    capsize=2.5,
    error_kw={'elinewidth': 1.0, 'capthick': 1.0}
)
plt.xticks(x, valid_methods, rotation=30, ha='right')
# plt.ylabel('Ergodic Metric\n(Mean ± Std across Steps)')
plt.title('Performance Stability Across Steps')
plt.tight_layout(pad=0.3)

bar_save_path = 'experiment2_mean_std.pdf'
plt.savefig(bar_save_path, format='pdf', bbox_inches='tight')
plt.show()
print(f"✅ 柱状图（均值±标准差）已保存至: {bar_save_path}")