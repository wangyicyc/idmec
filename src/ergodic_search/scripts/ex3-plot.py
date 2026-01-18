import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 👈 必须在 pyplot 之前，避免 GUI 后端
import matplotlib.pyplot as plt
import yaml
import os

# ==============================
# 1️⃣ 加载嵌套 YAML 数据（不预过滤）
# ==============================
file_path = 'experiment3.yaml'

if not os.path.exists(file_path):
    raise FileNotFoundError(f"YAML 文件未找到: {file_path}")

with open(file_path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

if not isinstance(data, list) or len(data) == 0:
    raise ValueError("YAML 文件应包含非空列表")

# 展开嵌套结构
rows = []
max_metrics = 0
for group in data:
    if isinstance(group, dict) and 'map_id' in group and 'types' in group:
        map_id = group['map_id']
        for rec in group['types']:
            if isinstance(rec, dict) and 'type' in rec and 'metrics' in rec:
                metrics = rec['metrics']
                max_metrics = max(max_metrics, len(metrics))
                rows.append({
                    'map_id': map_id,
                    'type': rec['type'],
                    'metrics': metrics
                })

if not rows:
    raise ValueError("未找到有效的实验记录")

# 构造完整 DataFrame
metric_cols = [f"metric_{i}" for i in range(max_metrics)]
df_rows = []
for r in rows:
    padded = (r['metrics'] + [np.nan] * max_metrics)[:max_metrics]
    df_rows.append({
        'map_id': r['map_id'],
        'type': r['type'],
        **{f'metric_{i}': v for i, v in enumerate(padded)}
    })

full_df = pd.DataFrame(df_rows)

# 排序 metric 列
metric_cols = sorted(
    [col for col in full_df.columns if col.startswith('metric_')],
    key=lambda x: int(x.split('_')[1])
)
steps = list(range(len(metric_cols)))

# 要绘制的方法
methods = ['baseline1', 'baseline2', 'baseline3', 'baseline4', 'baseline5', 'baseline6', 'method']
colors = {
    'baseline1': "#D400FF",
    'baseline2': "#7A3CDD",
    'baseline3': "#349CF7",
    'baseline4': "#2DC649",
    'baseline5': "#DFF706",
    'baseline6': "#F1B900",
    'method': "#000000"
}
markers = {
    'baseline1': 'o',
    'baseline2': 'o',
    'baseline3': 'o',
    'baseline4': '^',
    'baseline5': '^',
    'baseline6': '^',
    'method': '*'
}

# 获取所有 map_id（去重并排序）
map_ids = sorted(full_df['map_id'].unique())
print(f"检测到 map_id: {map_ids}")

# ==================== 在这里添加统计代码 ====================
print("\n" + "="*80)
print("📊 每个方法在每个地图的指标统计（均值和方差）")
print("="*80)

# 为每个地图和方法计算统计
for map_id in map_ids:
    map_df = full_df[full_df['map_id'] == map_id]
    
    print(f"\n📁 Map ID: {map_id}")
    print("-" * 60)
    print(f"{'Method':<12} {'Mean':<12} {'Variance':<12} {'#Metrics':<10}")
    print("-" * 60)
    
    for method in methods:
        method_data = map_df[map_df['type'] == method]
        
        if not method_data.empty:
            # 获取该方法的指标值
            metric_values = []
            for col in metric_cols:
                val = method_data[col].values[0]
                if not pd.isna(val):
                    metric_values.append(val)
            
            if metric_values:
                mean_val = np.mean(metric_values)
                var_val = np.var(metric_values)
                print(f"{method:<12} {mean_val:<12.6f} {var_val:<12.6f} {len(metric_values):<10}")
            else:
                print(f"{method:<12} {'N/A':<12} {'N/A':<12} {'0':<10}")
        else:
            print(f"{method:<12} {'N/A':<12} {'N/A':<12} {'0':<10}")

print("\n" + "="*80)
# ==================== 统计代码结束 ====================
# ==============================
# 2️⃣ 对每个 map_id 分别绘图
# ==============================
for map_id in map_ids:
    print(f"\n📊 正在处理 map_id = {map_id}")
    
    # 过滤当前 map_id 的数据
    df = full_df[full_df['map_id'] == map_id].reset_index(drop=True)
    
    # 如果没有指定 methods 中的任何方法，跳过
    available_methods = set(methods) & set(df['type'])
    if not available_methods:
        print(f"  ⚠️  map_id={map_id} 中没有匹配的方法 {methods}，跳过。")
        continue

    # ------------------------------
    # 折线图
    # ------------------------------
    original_rc = plt.rcParams.copy()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
        "axes.linewidth": 0.5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    plt.figure(figsize=(13, 5.2))

    plotted = False
    for i, method in enumerate(methods):
        row = df[df['type'] == method]
        if row.empty:
            continue
        values = row[metric_cols].iloc[0].values.astype(float)
        plt.plot(
            steps, values,
            color=colors[method],
            linewidth=2.5,
            marker=markers[method],
            markersize=20,
            markerfacecolor=colors[method],
            markeredgecolor='white',
            markeredgewidth=0.8,
            alpha=0.9,
            label=method,
            zorder=5
        )
        plotted = True

    if not plotted:
        plt.close()
        plt.rcParams.update(original_rc)
        continue

    # plt.xlabel('Step')
    # plt.legend(frameon=False, fontsize=20)
    plt.legend(frameon=False, fontsize=18, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(-0.5, len(steps) - 1 + 0.5)
    plt.ylim(-0.05, 1.25)  # ✅ 添加y轴范围限制
    plt.xticks(steps)
    plt.tight_layout()

    save_path = f'experiment3_map{map_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 👈 显式关闭 figure 释放内存
    plt.rcParams.update(original_rc)
    print(f"✅ 折线图已保存至: {save_path}")

    # ------------------------------
    # 柱状图（均值 ± 标准差）
    # ------------------------------
    # plt.rcParams.update({
    #     "text.usetex": False,
    #     "font.family": "sans-serif",
    #     "font.size": 8,
    #     "axes.labelsize": 9,
    #     "axes.titlesize": 9,
    #     "xtick.labelsize": 8,
    #     "ytick.labelsize": 8,
    #     "legend.fontsize": 8,
    #     "axes.grid": True,
    #     "grid.alpha": 0.3,
    #     "grid.linestyle": "--",
    #     "axes.linewidth": 1.0,
    #     "xtick.major.width": 1.0,
    #     "ytick.major.width": 1.0,
    #     "xtick.direction": "in",
    #     "ytick.direction": "in",
    #     "pdf.fonttype": 42,
    # })

    # means, stds, valid_methods, valid_colors_list = [], [], [], []
    # for method in methods:
    #     row = df[df['type'] == method]
    #     if not row.empty:
    #         vals = row[metric_cols].iloc[0].astype(float)
    #         means.append(vals.mean())
    #         stds.append(vals.std())
    #         valid_methods.append(method)
    #         valid_colors_list.append(colors[method])

    # if not valid_methods:
    #     plt.rcParams.update(original_rc)
    #     continue

    # x = np.arange(len(valid_methods))
    # plt.figure(figsize=(3.6, 2.3))
    # plt.bar(
    #     x, means, yerr=stds,
    #     color=valid_colors_list,
    #     capsize=2.5,
    #     error_kw={'elinewidth': 1.0, 'capthick': 1.0}
    # )
    # plt.xticks(x, valid_methods, rotation=30, ha='right')
    # plt.tight_layout(pad=0.3)

    # bar_save_path = f'experiment2_map{map_id}_mean_std.png'
    # plt.savefig(bar_save_path, dpi=300, bbox_inches='tight')
    # plt.close()
    # print(f"✅ 柱状图已保存至: {bar_save_path}")



# ==============================
# 3 跨map绘制每个method的boxplot
# ==============================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10.5,
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
})

global_box_data = []
global_labels = []
global_colors = []

for method in methods:
    all_values = []
    for _, row in full_df.iterrows():
        if row['type'] == method:
            vals = [row[col] for col in metric_cols if not pd.isna(row[col])]
            all_values.extend(vals)
    if all_values:
        global_box_data.append(all_values)
        global_labels.append(method)
        global_colors.append(colors[method])

if global_box_data:
    fig, ax = plt.subplots(figsize=(7.1, 4.5))
    
    # 绘制箱线图（无填充）
    bp = ax.boxplot(
        global_box_data,
        # tick_labels=global_labels,
        patch_artist=False,
        medianprops=dict(linewidth=3.0),
        boxprops=dict(linewidth=2.0),
        whiskerprops=dict(linewidth=2.0),
        capprops=dict(linewidth=2.0),
        flierprops=dict(
            marker='o',
            markersize=9,
            markerfacecolor='black',  # 改为实心
            markeredgecolor='black',  # 边缘颜色
            alpha=0.6
        )
    )
    # 为每个方法设置统一颜色（包括异常值）
    for i, color in enumerate(global_colors):
        bp['boxes'][i].set_color(color)
        bp['whiskers'][2*i].set_color(color)
        bp['whiskers'][2*i + 1].set_color(color)
        bp['caps'][2*i].set_color(color)
        bp['caps'][2*i + 1].set_color(color)
        bp['medians'][i].set_color(color)
        # ✅ 关键：设置异常值的颜色
        bp['fliers'][i].set_markeredgecolor(color)
        bp['fliers'][i].set_markerfacecolor(color)  # 添加这一行使异常点变为实心
    ax.set_ylim(-0.05, 1.25)
    ax.set_xticklabels([])
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(ha='center')
    plt.tight_layout()
    plt.savefig('experiment3_box.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 全局箱线图已保存至: experiment3_box.png")

print("\n🎉 所有地图可视化完成！")