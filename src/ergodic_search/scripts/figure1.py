import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

# ============================
# 1. Grid map 尺寸
# ============================
W, H = 150, 200
grid = np.zeros((H, W))

# ============================
# 2. 定义多个火焰源
# ============================
fire_sources = [
    {
        "pos": (70, 70),
        "theta": np.deg2rad(60),
        "sigma": 11.0,
        "anisotropy": 2.0,
        "weight": 0.9
    },
    {
        "pos": (110, 150),
        "theta": np.deg2rad(60),
        "sigma": 11.0,
        "anisotropy": 1.5,
        "weight": 0.9
    },
]
    # {"pos": (110, 70), "theta": np.deg2rad(65), "sigma": 11.0, "anisotropy": 1.5, "weight": 0.9},
    # {"pos": (50, 140), "theta": np.deg2rad(65), "sigma": 11.0, "anisotropy": 1.5, "weight": 0.9},
# ============================
# 3. 生成火焰信息场
# ============================
for fire in fire_sources:
    cx, cy = fire["pos"]
    theta = fire["theta"]
    sigma = fire["sigma"]
    a = fire["anisotropy"]
    w = fire["weight"]

    c, s = np.cos(theta), np.sin(theta)

    for i in range(H):
        for j in range(W):
            dx = j - cx
            dy = i - cy

            x_p =  c * dx + s * dy
            y_p = -s * dx + c * dy

            d_eff2 = (x_p / a) ** 2 + y_p ** 2
            grid[i, j] += w * np.exp(-d_eff2 / (2 * sigma ** 2))

grid = np.clip(grid / grid.max(), 0.0, 1.0)

# ============================
# 4. 颜色映射
# ============================
cmap = mcolors.LinearSegmentedColormap.from_list(
    "forest_to_fire",
    ["#2E8034", "#5AB95D", "#FFEB3B", "#FF9800", "#B71C1C"]
)
norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
colors = cmap(norm(grid))  # shape: (H, W, 4)

# ============================
# 5. 设置柱体参数
# ============================
height_scale = 10
base_height = 0.5
Z_height = base_height + height_scale * grid  # 每个柱子总高度

# 准备 bar3d 所需的一维数组
x = np.arange(W)
y = np.arange(H)
X, Y = np.meshgrid(x, y)
X_flat = X.ravel()
Y_flat = Y.ravel()
Z_bottom = np.zeros_like(Z_height).ravel()      # 从 Z=0 开始
Z_height_flat = Z_height.ravel()

# 颜色展平（去掉透明度或保留）
facecolors_flat = colors.reshape(-1, 4)

# 柱体尺寸
dx = dy = 1.0

# ============================
# 6. 创建无边框 3D 图
# ============================
fig = plt.figure(figsize=(8, 6), frameon=False)
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
# 绘制柱体（无边框）
ax.bar3d(
    X_flat, Y_flat, Z_bottom,
    dx, dy, Z_height_flat,
    color=facecolors_flat,
    shade=True,
    linewidth=0,      # 关键：柱体自身无边框
    edgecolor='none'  # 确保无边缘线
)

# --- 彻底移除所有 3D 边框和装饰 ---
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_zlim(0, 30)

# 隐藏 ticks 和 labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# 关闭 panes（背景面）的填充和边框
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.pane.set_edgecolor('none')
ax.yaxis.pane.set_edgecolor('none')
ax.zaxis.pane.set_edgecolor('none')

# 隐藏坐标轴的“脊柱”（3D 中通过 pane 边框控制）
ax.xaxis.line.set_color((1, 1, 1, 0))
ax.yaxis.line.set_color((1, 1, 1, 0))
ax.zaxis.line.set_color((1, 1, 1, 0))

ax.set_axis_off() # 移除了坐标轴线和刻度 

# 去除整个图的空白边距
ax.margins(0)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.view_init(elev=10, azim=-10)

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ============================
# 8. 保存
# ============================
plt.savefig(
    "figure1.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0
)

# plt.show()