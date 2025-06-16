import numpy as np
import random

def generate_binary_matrix_from_animation(resolution=4000, total_frames=360):
    # 配置参数
    total_points = np.random.randint(5, 40)
    num_outliers = np.random.randint(2, total_points // 4 + 1)
    num_inliers_joining = np.random.randint(2, total_points // 4 + 1)
    speed_factor = 4

    angle = np.random.rand() * 2 * np.pi
    initial_direction = np.array([np.cos(angle), np.sin(angle)])

    # 初始化位置和状态
    positions = np.random.rand(total_points, 2) * 600 - 300  # 初始位置在 [-300, 300] 范围内
    velocities = np.zeros_like(positions)

    # 标记哪些是离群点、入群点、被扰动点
    is_outlier = np.array([i < num_outliers for i in range(total_points)])
    is_joiner = np.array([(i >= total_points - num_inliers_joining) for i in range(total_points)])
    is_perturbed = np.zeros(total_points, dtype=bool)  # 新增：记录是否已被扰动

    # 设置初始速度
    for i in range(total_points):
        if is_outlier[i]:
            rand_dir = (np.random.rand(2) - 0.5)
            rand_dir /= np.linalg.norm(rand_dir)  # 归一化
            velocities[i] = rand_dir * speed_factor
        elif is_joiner[i]:
            velocities[i] = (np.random.rand(2) - 0.5) * speed_factor * 2
        else:
            velocities[i] = initial_direction / np.linalg.norm(initial_direction) * speed_factor

    # 设置入群点切换时间（帧数）
    join_frame = 60  # 在第 60 帧时入群点开始跟从主方向

    # 添加扰动时间点（3s=90帧, 6s=180帧, 9s=270帧）
    perturb_frames = [90, 180, 270]

    # 新增：随机生成3个改变群体方向的时间点（避开前几帧）
    group_perturb_frames = sorted(np.random.choice(range(30, total_frames), size=3, replace=False))

    # 当前主方向和速度
    current_direction = initial_direction.copy()
    current_speed_factor = speed_factor

    # 新增：方向和速度的小幅波动参数
    direction_noise_strength = 0.05  # 弧度制，约 ±2.86°
    speed_noise_strength = 0.1       # 速度波动范围：±10%

    # 创建输出矩阵
    output = np.zeros((total_frames, resolution, resolution), dtype=np.uint8)

    # 视图范围（与绘图一致）
    world_size = 2000  # 对应 xlim/ylim [-2000, 2000]
    scale = resolution / (2 * world_size)  # 缩放因子：世界坐标 → 像素坐标

    def world_to_pixel(pos):
        """将世界坐标转换为像素坐标"""
        return ((pos + world_size) * scale).astype(int)

    # 逐帧模拟
    for frame in range(total_frames):

        # 更新入群点的速度
        if frame == join_frame:
            for i in range(total_points):
                if is_joiner[i]:
                    velocities[i] = current_direction / np.linalg.norm(current_direction) * current_speed_factor

        # 添加扰动：在特定帧随机改变一些点的方向和速度
        if frame in perturb_frames:
            num_perturb = np.random.randint(2, 5)  # 随机扰动 2~4 个点
            indices = np.random.choice(np.arange(total_points), size=num_perturb, replace=False)

            for i in indices:
                rand_dir = (np.random.rand(2) - 0.5)
                rand_dir /= np.linalg.norm(rand_dir)
                velocities[i] = rand_dir * current_speed_factor
                is_perturbed[i] = True  # 标记为已扰动

        # 改变群体主方向和速度
        if frame in group_perturb_frames:
            new_dir = (np.random.rand(2) - 0.5)
            new_dir /= np.linalg.norm(new_dir)
            current_direction = new_dir
            current_speed_factor = np.random.uniform(0.8, 1.5)

            for i in range(total_points):
                if not is_outlier[i] and not is_joiner[i] and not is_perturbed[i]:
                    velocities[i] = current_direction * current_speed_factor
                elif is_joiner[i]:
                    velocities[i] = current_direction * current_speed_factor

        # 正常群体点每帧都有小幅波动
        for i in range(total_points):
            if not is_outlier[i] and not is_joiner[i] and not is_perturbed[i]:
                vel = velocities[i]
                speed = np.linalg.norm(vel)
                if speed == 0:
                    continue
                direction = vel / speed

                # 方向扰动
                angle_noise = np.random.uniform(-direction_noise_strength, direction_noise_strength)
                rotation_matrix = np.array([
                    [np.cos(angle_noise), -np.sin(angle_noise)],
                    [np.sin(angle_noise), np.cos(angle_noise)]
                ])
                noisy_direction = rotation_matrix @ direction

                # 速度扰动
                speed_noise = 1.0 + np.random.uniform(-speed_noise_strength, speed_noise_strength)
                velocities[i] = noisy_direction * speed * speed_noise

        # 更新所有点的位置
        positions += velocities

        # 获取当前点的像素坐标
        px_positions = world_to_pixel(positions)
        px_positions = np.clip(px_positions, 0, resolution - 1)

        # 构建当前帧的二值矩阵
        frame_matrix = np.ones((resolution, resolution), dtype=np.uint8)
        for x, y in px_positions:
            frame_matrix[y, x] = 0  # 注意：y 是行号，x 是列号

        output[frame] = frame_matrix

    return output


# 示例调用
if __name__ == "__main__":
    binary_video = generate_binary_matrix_from_animation()
    print("生成完成, shape:", binary_video.shape)

    # 可选：保存为 .npy 文件
    np.save('/Users/lorenzo/Desktop/myStudy/crowdFollow/binary_animation.npy', binary_video)
    print("✅ 二值视频矩阵已保存为 binary_animation.npy")