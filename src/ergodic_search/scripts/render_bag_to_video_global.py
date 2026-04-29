#!/usr/bin/env python3
"""
支持三重动态数据的时间对齐可视化（平等渲染所有轨迹）：
  - /map_distribution                → 动态热力图
  - /robot_X/planned_path            → 动态规划路径（每次重规划更新）
  - /robot_X/trajectory/control_sequence → 真实轨迹（逐点发布）

所有机器人轨迹均以相同样式绘制：
  - 无发光效果
  - 规划路径线宽统一
  - 真实轨迹线宽统一 + 渐变衰减
"""

import rosbag
import cv2
import numpy as np
import os
from quadrotor_msgs.msg import PositionCommand
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker

def hex_to_bgr(hex_color):
    """
    将 '#rrggbb' 转为 OpenCV 的 BGR 元组 (b, g, r)，每个值为 0～255 的 int
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV is BGR

ROBOT_COLORS = {
    0: hex_to_bgr('#1d6cd4'),
    1: hex_to_bgr('#facf21'),
    2: hex_to_bgr('#01a064'),
    3: hex_to_bgr('#aa00ff')
}
PAST_TRAJ_LINE_WIDTH = 6      # 所有机器人历史轨迹线宽
PLANNED_PATH_LINE_WIDTH = 8   # 所有机器人规划路径线宽

def build_heatmap_image(marker_msg, target_size=(2460, 2460), pixel_block=9):
    if len(marker_msg.points) == 0:
        img = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255
        return img, None
    points = np.array([[p.x, p.y] for p in marker_msg.points])
    colors = np.array([[c.r, c.g, c.b] for c in marker_msg.colors])
    # 处理归一化颜色 [0,1] → [0,255]
    if colors.size > 0 and colors.max() <= 1.0 + 1e-5:
        colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    else:
        colors = np.clip(colors, 0, 255).astype(np.uint8)

    xmin, xmax = points[:, 0].min(), points[:, 0].max()
    ymin, ymax = points[:, 1].min(), points[:, 1].max()
    # 安全扩展边界
    dx = (xmax - xmin) * 0.05 if (xmax - xmin) > 1e-8 else 1.0
    dy = (ymax - ymin) * 0.05 if (ymax - ymin) > 1e-8 else 1.0
    xmin, xmax = xmin - dx, xmax + dx
    ymin, ymax = ymin - dy, ymax + dy

    width, height = target_size
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白底

    for (x, y), (r, g, b) in zip(points, colors):
        u = int((x - xmin) / (xmax - xmin) * (width - 1))
        v = int((ymax - y) / (ymax - ymin) * (height - 1))

        if 0 <= u < width and 0 <= v < height:
            half = pixel_block // 2
            u1, u2 = max(0, u - half), min(width, u + half + 1)
            v1, v2 = max(0, v - half), min(height, v + half + 1)
            img[v1:v2, u1:u2] = [b, g, r]  # BGR

    return img, (xmin, xmax, ymin, ymax)

def world_to_pixel(x, y, bounds, img_shape):
    if bounds is None:
        h, w = img_shape[:3]
        return w // 2, h // 2
    h, w = img_shape[:2]
    xmin, xmax, ymin, ymax = bounds
    u = int((x - xmin) / (xmax - xmin) * (w - 1))
    v = int((ymax - y) / (ymax - ymin) * (h - 1))
    return u, v

def extract_path_points(path_msg):
    """从 nav_msgs/Path 提取 (x, y) 列表"""
    pts = []
    for pose in path_msg.poses:
        pts.append([pose.pose.position.x, pose.pose.position.y])
    return pts


def draw_quadrotor(frame, cx, cy, yaw, color, is_execute=False, scale=1.0):
    """
    参数:
        frame: OpenCV 图像 (BGR)
        cx, cy: 中心像素坐标
        yaw: 朝向角（弧度，+x 为 0）
        color: BGR 主色（用于旋翼填充颜色）
        is_execute: 是否是重点机器人（会更大、更醒目）
        scale: 整体缩放因子（默认 1.0）
    """
    # 缩放参数
    arm_len = int(60 * scale)
    rotor_r = int(30 * scale)
    line_width_arm = 10 if not is_execute else 15  # 更粗的机臂线条
    rotor_border = 5

    # 四个旋翼基础角度（NW, NE, SW, SE）
    base_angles = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
    
    # 计算旋翼位置（考虑 yaw）
    rotors = []
    for base_ang in base_angles:
        ang = base_ang + yaw
        rx = int(cx + arm_len * np.cos(ang))
        ry = int(cy - arm_len * np.sin(ang))  # y 轴向下，需取反
        rotors.append((rx, ry))

    # 绘制十字臂（主色，更粗的线条）
    for (rx, ry) in rotors:
        cv2.line(frame, (cx, cy), (rx, ry), (0, 0, 0), line_width_arm, cv2.LINE_AA)

    # 绘制旋翼：原色填充 + 黑色边框
    for (rx, ry) in rotors:
        cv2.circle(frame, (rx, ry), rotor_r, color, -1)  # 原色填充
        cv2.circle(frame, (rx, ry), rotor_r, (0, 0, 0), rotor_border)   # 黑色边框

    # 绘制朝向箭头（黑色，抗锯齿）
    arrow_len = int(arm_len * 1.4)
    dx = int(arrow_len * np.cos(yaw))
    dy = -int(arrow_len * np.sin(yaw))
    cv2.arrowedLine(
        frame, 
        (cx, cy), 
        (cx + dx, cy + dy), 
        (0, 0, 0), 
        thickness=line_width_arm, 
        tipLength=0.25,
        line_type=cv2.LINE_AA
    )

    # 如果是 EXECUTE_ID，加微弱高亮背景（提升视觉层级）
    if is_execute:
        glow_r = int((arm_len + rotor_r) * 1.4)
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), glow_r, color, -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)


def main():
    BAG_PATH = "../datas/bags/global.bag"
    OUTPUT_VIDEO = "global.mp4"
    ROBOT_IDS = [0, 1, 2, 3]
    FPS = 10

    print(f"📂 正在读取 {BAG_PATH} ...")
    bag = rosbag.Bag(BAG_PATH, 'r')

    # === 1. 加载动态地图 ===
    map_msgs = []
    for _, msg, t in bag.read_messages(topics=["/map_distribution"]):
        map_msgs.append((t.to_sec(), *build_heatmap_image(msg)))
    map_msgs.sort(key=lambda x: x[0])
    if not map_msgs:
        default_img = np.ones((720, 960, 3), dtype=np.uint8) * 255
        map_msgs = [(-float('inf'), default_img, (-10,10,-10,10))]
    print(f"✅ 地图帧数: {len(map_msgs)}")

    # === 2. 加载规划路径（每个机器人独立）===
    planned_paths = {rid: [] for rid in ROBOT_IDS}
    for rid in ROBOT_IDS:
        topic = f'robot_{rid}/planned_path'
        for _, msg, t in bag.read_messages(topics=[topic]):
            pts = extract_path_points(msg)
            planned_paths[rid].append((t.to_sec(), pts))
        planned_paths[rid].sort(key=lambda x: x[0])
        if not planned_paths[rid]:
            planned_paths[rid] = [(-float('inf'), [])]
    print(f"✅ 规划路径已加载（每个机器人）")

    # === 3. 加载真实轨迹（带时间戳和速度）===
    real_trajs = {rid: [] for rid in ROBOT_IDS}
    all_times = set()
    for rid in ROBOT_IDS:
        topic = f'robot_{rid}/trajectory/control_sequence'
        for _, msg, t in bag.read_messages(topics=[topic]):
            t_sec = t.to_sec()
            real_trajs[rid].append((
                t_sec,
                msg.position.x,
                msg.position.y,
                msg.velocity.x,
                msg.velocity.y
            ))
            all_times.add(t_sec)
        real_trajs[rid].sort(key=lambda x: x[0])

    start_positions = {}
    for rid in ROBOT_IDS:
        if real_trajs[rid]:
            _, x0, y0, _, _ = real_trajs[rid][0]
            start_positions[rid] = (x0, y0)
        else:
            start_positions[rid] = None

    all_times = sorted(all_times)
    if not all_times:
        print("❌ 无真实轨迹数据")
        return

    MAX_FRAMES = 3000
    if len(all_times) > MAX_FRAMES:
        step = len(all_times) // MAX_FRAMES
        all_times = all_times[::step]
    print(f"🎬 总帧数: {len(all_times)}")

    # === 4. 渲染每一帧 ===
    frames = []
    map_idx = 0
    path_indices = {rid: 0 for rid in ROBOT_IDS}

    for frame_t in all_times:
        # 更新地图
        while map_idx + 1 < len(map_msgs) and map_msgs[map_idx + 1][0] <= frame_t:
            map_idx += 1
        bg_img, bounds = map_msgs[map_idx][1], map_msgs[map_idx][2]
        h, w = bg_img.shape[:2]
        frame = bg_img.copy()

        # === 统一绘制所有机器人（平等对待）===
        for rid in ROBOT_IDS:
            color = ROBOT_COLORS.get(rid, (128, 128, 128))

            # --- 规划路径（普通线，无 glow）---
            path_list = planned_paths[rid]
            path_idx = path_indices[rid]
            while path_idx + 1 < len(path_list) and path_list[path_idx + 1][0] <= frame_t:
                path_idx += 1
            path_indices[rid] = path_idx
            current_path = path_list[path_idx][1]

            if current_path:
                pts = np.array([world_to_pixel(px, py, bounds, frame.shape) for px, py in current_path])
                valid = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
                pts = pts[valid]
                if len(pts) > 1:
                    for j in range(1, len(pts)):
                        cv2.line(frame, tuple(pts[j-1]), tuple(pts[j]), color, PLANNED_PATH_LINE_WIDTH)

            # --- 真实轨迹（渐变衰减，统一线宽）---
            traj = real_trajs[rid]
            current_traj = []
            current_pos = None
            current_vel = None
            for record in traj:
                t, x, y, vx, vy = record
                if t <= frame_t:
                    current_traj.append((x, y))
                    current_pos = (x, y)
                    current_vel = (vx, vy)
                else:
                    break

            if current_pos and len(current_traj) > 1:
                traj_px = np.array([world_to_pixel(tx, ty, bounds, frame.shape) for tx, ty in current_traj])
                valid = (traj_px[:, 0] >= 0) & (traj_px[:, 0] < w) & (traj_px[:, 1] >= 0) & (traj_px[:, 1] < h)
                traj_px = traj_px[valid]
                n = len(traj_px)
                if n > 1:
                    for j in range(1, n):
                        alpha = j / (n - 1)
                        faded_color = tuple(
                            int(color[c] * alpha + 255 * (1 - alpha)) for c in range(3)
                        )
                        cv2.line(frame, tuple(traj_px[j-1]), tuple(traj_px[j]), faded_color, PAST_TRAJ_LINE_WIDTH)

            # --- 起点 ---
            if start_positions[rid] is not None:
                sx, sy = start_positions[rid]
                su, sv = world_to_pixel(sx, sy, bounds, frame.shape)
                if 0 <= su < w and 0 <= sv < h:
                    triangle_size = 25
                    triangle_points = np.array([
                        (su, sv - triangle_size),
                        (su - triangle_size, sv + triangle_size),
                        (su + triangle_size, sv + triangle_size)
                    ], dtype=np.int32)
                    cv2.fillPoly(frame, [triangle_points], ROBOT_COLORS[rid])

            # --- 四旋翼图标 ---
            if current_pos:
                u, v = world_to_pixel(*current_pos, bounds, frame.shape)
                if 0 <= u < w and 0 <= v < h:
                    yaw = np.pi / 2
                    if current_vel is not None and np.hypot(*current_vel) > 1e-3:
                        yaw = np.arctan2(current_vel[1], current_vel[0])
                    draw_quadrotor(frame, u, v, yaw, color, is_execute=False, scale=1.0)
        frames.append(frame)

    bag.close()

    # === 5. 保存视频 ===
    print("💾 正在写入视频...")
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (frames[0].shape[1], frames[0].shape[0]))
    for f in frames:
        out.write(f)
    out.release()
    print(f"🎉 视频已生成: {os.path.abspath(OUTPUT_VIDEO)}")

    # === 6. 保存 GIF 动图（新增部分）===
    import imageio
    OUTPUT_GIF = "global.gif"
    print("🌀 正在生成 GIF 动图...")
    # OpenCV 是 BGR，imageio 需要 RGB
    gif_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    # 写入 GIF（duration 单位是秒每帧）
    imageio.mimsave(
        OUTPUT_GIF,
        gif_frames,
        fps=FPS,
        loop=0  # 0 表示无限循环
    )
    print(f"🎉 GIF 已生成: {os.path.abspath(OUTPUT_GIF)}")

if __name__ == "__main__":
    main()
