#!/usr/bin/env python3
"""
支持三重动态数据的时间对齐可视化：
  - /map_distribution                → 动态热力图
  - /robot_X/planned_path            → 动态规划路径（每次重规划更新）
  - /robot_X/trajectory/control_sequence → 真实轨迹（逐点发布）
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
PAST_TRAJ_LINE_WIDTH = 6      # 历史真实轨迹：粗
PLANNED_PATH_LINE_WIDTH = 8   # 规划路径（未来）

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
    dx = (xmax - xmin) * 0.01 if (xmax - xmin) > 1e-8 else 1.0
    dy = (ymax - ymin) * 0.01 if (ymax - ymin) > 1e-8 else 1.0
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
        h, w = img_shape[:2]
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

def main():
    EXECUTE_ID = 1
    BAG_PATH = f"../datas/bags/robot_{EXECUTE_ID}.bag"
    OUTPUT_VIDEO = f"robot_{EXECUTE_ID}.mp4"
    ROBOT_IDS = [0, 1, 2, 3]
    FPS = 10

    print(f"📂 正在读取 {BAG_PATH} ...")
    bag = rosbag.Bag(BAG_PATH, 'r')

    # === 1. 加载动态地图 ===
    map_msgs = []  # [(t_sec, img, bounds), ...]
    for _, msg, t in bag.read_messages(topics=["/map_distribution"]):
        map_msgs.append((t.to_sec(), *build_heatmap_image(msg)))
    map_msgs.sort(key=lambda x: x[0])
    if not map_msgs:
        default_img = np.ones((720, 960, 3), dtype=np.uint8) * 255
        map_msgs = [(-float('inf'), default_img, (-10,10,-10,10))]
    print(f"✅ 地图帧数: {len(map_msgs)}")

    # === 2. 加载动态规划路径（每个机器人独立）===
    planned_paths = {rid: [] for rid in ROBOT_IDS}  # rid -> [(t_sec, points), ...]
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
            # 保存: (t, x, y, vx, vy)
            real_trajs[rid].append((
                t_sec,
                msg.position.x,
                msg.position.y,
                msg.velocity.x,
                msg.velocity.y
            ))
            all_times.add(t_sec)
        real_trajs[rid].sort(key=lambda x: x[0])

    # 提取起点（仅用位置）
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

    # 限制帧数
    MAX_FRAMES = 1000
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

                # === 第一轮：绘制所有非 EXECUTE_ID 的机器人 ===
        for rid in ROBOT_IDS:
            if rid == EXECUTE_ID:
                continue  # 跳过目标机器人
            color = ROBOT_COLORS.get(rid, (128, 128, 128))

            # --- 规划路径 ---
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
                    light_color = tuple(int(c) for c in color)
                    for j in range(1, len(pts)):
                        cv2.line(frame, tuple(pts[j-1]), tuple(pts[j]), light_color, PLANNED_PATH_LINE_WIDTH)

            # --- 真实轨迹与当前位置 ---
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

            if not current_pos:
                continue

            # === 普通机器人：渐变轨迹 ===
            if len(current_traj) > 1:
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

            # === 起点（三角形）===
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

            # === 当前位置：四旋翼图标 ===
            u, v = world_to_pixel(*current_pos, bounds, frame.shape)
            if 0 <= u < w and 0 <= v < h:
                arm_len = 40
                rotor_r = 20
                c = color
                cx, cy = u, v

                base_angles = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
                yaw = np.pi / 2  # 默认朝上
                if current_vel is not None and np.hypot(*current_vel) > 1e-3:
                    yaw = np.arctan2(current_vel[1], current_vel[0])

                rotors = []
                for base_ang in base_angles:
                    ang = base_ang + yaw
                    rx = int(cx + arm_len * np.cos(ang))
                    ry = int(cy - arm_len * np.sin(ang))
                    rotors.append((rx, ry))

                for (rx, ry) in rotors:
                    cv2.line(frame, (cx, cy), (rx, ry), c, 2)
                    cv2.circle(frame, (rx, ry), rotor_r, c, -1)
                    cv2.circle(frame, (rx, ry), rotor_r, (0, 0, 0), 1)

                arrow_len = arm_len * 1.3
                dx = int(arrow_len * np.cos(yaw))
                dy = -int(arrow_len * np.sin(yaw))
                cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy), (0, 0, 0), 2, tipLength=0.25)

        # === 第二轮：单独绘制 EXECUTE_ID 机器人（最上层！）===
        rid = EXECUTE_ID
        color = ROBOT_COLORS.get(rid, (128, 128, 128))

        # --- 规划路径（加粗 or 高亮）---
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
                # 可选：让 EXECUTE_ID 的规划路径也更醒目
                highlight_color = tuple(min(255, c + 50) for c in color)  # 稍亮一点
                for j in range(1, len(pts)):
                    cv2.line(frame, tuple(pts[j-1]), tuple(pts[j]), highlight_color, PLANNED_PATH_LINE_WIDTH + 2)

        # --- 真实轨迹（带强发光）---
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

        if current_pos:
            if len(current_traj) > 1:
                traj_px = np.array([world_to_pixel(tx, ty, bounds, frame.shape) for tx, ty in current_traj])
                valid = (traj_px[:, 0] >= 0) & (traj_px[:, 0] < w) & (traj_px[:, 1] >= 0) & (traj_px[:, 1] < h)
                traj_px = traj_px[valid]
                n = len(traj_px)
                if n > 1:
                    glow_color = color
                    main_width = PAST_TRAJ_LINE_WIDTH + 6
                    glow_layer = np.zeros_like(frame)
                    glow_width = 45
                    for j in range(1, n):
                        cv2.line(glow_layer, tuple(traj_px[j-1]), tuple(traj_px[j]), glow_color, glow_width)
                    blurred_glow = cv2.GaussianBlur(glow_layer, (105, 105), 0)
                    frame = np.clip(frame.astype(np.float32) + blurred_glow.astype(np.float32) * 2.0, 0, 255).astype(np.uint8)
                    for j in range(1, n):
                        cv2.line(frame, tuple(traj_px[j-1]), tuple(traj_px[j]), glow_color, main_width)

            # === 起点（三角形，最上层）===
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

            # === 当前位置：四旋翼（最上层）===
            u, v = world_to_pixel(*current_pos, bounds, frame.shape)
            if 0 <= u < w and 0 <= v < h:
                arm_len = 40
                rotor_r = 20
                c = color
                cx, cy = u, v

                base_angles = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
                yaw = np.pi / 2
                if current_vel is not None and np.hypot(*current_vel) > 1e-3:
                    yaw = np.arctan2(current_vel[1], current_vel[0])

                rotors = []
                for base_ang in base_angles:
                    ang = base_ang + yaw
                    rx = int(cx + arm_len * np.cos(ang))
                    ry = int(cy - arm_len * np.sin(ang))
                    rotors.append((rx, ry))

                for (rx, ry) in rotors:
                    cv2.line(frame, (cx, cy), (rx, ry), c, 2)
                    cv2.circle(frame, (rx, ry), rotor_r, c, -1)
                    cv2.circle(frame, (rx, ry), rotor_r, (0, 0, 0), 1)

                arrow_len = arm_len * 1.3
                dx = int(arrow_len * np.cos(yaw))
                dy = -int(arrow_len * np.sin(yaw))
                cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy), (0, 0, 0), 2, tipLength=0.25)
        frames.append(frame)

    bag.close()

    # === 5. 保存视频 ===
    print("💾 正在写入视频...")
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (frames[0].shape[1], frames[0].shape[0]))
    for f in frames:
        out.write(f)
    out.release()

    print(f"🎉 视频已生成: {os.path.abspath(OUTPUT_VIDEO)}")

if __name__ == "__main__":
    main()