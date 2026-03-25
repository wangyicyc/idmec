import numpy as np
import matplotlib
matplotlib.use('Agg')  # 避免 Qt/Wayland 问题
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from tqdm import tqdm
import imageio.v2 as imageio

# ============================
# 参数设置
# ============================
W, H = 150, 200
TOTAL_FRAMES = 300
FPS = 20
OUTPUT_VIDEO = "animated_flame_smooth.mp4"

# ============================
# 起始与目标火焰源
# ============================
fire_sources_start = [
    {"pos": (70.0, 70.0), "theta": np.deg2rad(60), "sigma": 11.0, "anisotropy": 2.0, "weight": 0.9},
    {"pos": (110.0, 150.0), "theta": np.deg2rad(60), "sigma": 11.0, "anisotropy": 1.5, "weight": 0.9}
]

fire_sources_target = [
    {"pos": (110.0, 70.0), "theta": np.deg2rad(65), "sigma": 11.0, "anisotropy": 1.5, "weight": 0.9},
    {"pos": (60.0, 130.0), "theta": np.deg2rad(65), "sigma": 11.0, "anisotropy": 1.5, "weight": 0.9}
]

# ============================
# 缓动函数：平滑起停
# ============================
def ease_in_out(t):
    """S形缓动：0→0, 0.5→0.5, 1→1，导数在两端为0"""
    return 3 * t**2 - 2 * t**3

# ============================
# 插值更新火焰源（关键改进）
# ============================
def update_fire_sources(frame, total_frames=TOTAL_FRAMES):
    t = frame / total_frames
    progress = ease_in_out(t)  # ← 使用缓动函数
    
    fire_sources = []
    for i in range(len(fire_sources_start)):
        start = fire_sources_start[i]
        target = fire_sources_target[i]
        
        # 位置：保留浮点精度（不再 int()！）
        new_pos = (
            start["pos"][0] + (target["pos"][0] - start["pos"][0]) * progress,
            start["pos"][1] + (target["pos"][1] - start["pos"][1]) * progress
        )
        
        # 其他参数线性插值（角度变化小，可接受）
        new_theta = start["theta"] + (target["theta"] - start["theta"]) * progress
        new_sigma = start["sigma"] + (target["sigma"] - start["sigma"]) * progress
        new_anisotropy = start["anisotropy"] + (target["anisotropy"] - start["anisotropy"]) * progress
        new_weight = start["weight"] + (target["weight"] - start["weight"]) * progress
        
        fire_sources.append({
            "pos": new_pos,
            "theta": new_theta,
            "sigma": new_sigma,
            "anisotropy": new_anisotropy,
            "weight": new_weight
        })
    return fire_sources

# ============================
# 生成火焰热力图（支持浮点中心）
# ============================
def generate_grid(fire_sources):
    grid = np.zeros((H, W))
    for fire in fire_sources:
        cx, cy = fire["pos"]  # ← 现在是 float，完全合法
        theta = fire["theta"]
        sigma = fire["sigma"]
        a = fire["anisotropy"]
        w = fire["weight"]
        c, s = np.cos(theta), np.sin(theta)
        for i in range(H):
            for j in range(W):
                dx = j - cx
                dy = i - cy
                x_p = c * dx + s * dy
                y_p = -s * dx + c * dy
                d_eff2 = (x_p / a) ** 2 + y_p ** 2
                grid[i, j] += w * np.exp(-d_eff2 / (2 * sigma ** 2))
    return np.clip(grid / (grid.max() + 1e-8), 0.0, 1.0)

# ============================
# 颜色与绘图设置
# ============================
cmap = mcolors.LinearSegmentedColormap.from_list(
    "forest_to_fire",
    ["#2E8034", "#5AB95D", "#FFEB3B", "#FF9800", "#B71C1C"]
)
norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

height_scale = 10
base_height = 0.5

x = np.arange(W)
y = np.arange(H)
X, Y = np.meshgrid(x, y)
X_flat = X.ravel()
Y_flat = Y.ravel()
Z_bottom = np.zeros_like(X_flat)
dx = dy = 1.0

print("🎥 正在渲染火焰动画帧...")

# 使用 imageio 流式写入，避免内存爆炸
with imageio.get_writer(
    OUTPUT_VIDEO,
    fps=FPS,
    codec='libx264',
    ffmpeg_params=['-crf', '15', '-preset', 'fast', '-pix_fmt', 'yuv420p']
) as writer:

    for frame in tqdm(range(TOTAL_FRAMES), desc="渲染进度"):
        fire_sources = update_fire_sources(frame, TOTAL_FRAMES)
        grid = generate_grid(fire_sources)
        colors = cmap(norm(grid))
        Z_height = base_height + height_scale * grid
        Z_height_flat = Z_height.ravel()
        facecolors_flat = colors.reshape(-1, 4)
        
        fig = plt.figure(figsize=(6, 8), dpi=450)
        ax = fig.add_subplot(111, projection='3d')
        
        ax.bar3d(
            X_flat, Y_flat, Z_bottom,
            dx, dy, Z_height_flat,
            color=facecolors_flat,
            shade=True,
            linewidth=0,
            edgecolor='none'
        )
        
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_zlim(0, 35)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_axis_off()
        ax.view_init(elev=10, azim=-10)
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]  # RGB
        
        writer.append_data(img)   # ← 直接写入，不存列表！
        plt.close(fig)            # ← 立即释放 figure

print(f"✅ 视频已生成: {OUTPUT_VIDEO}")