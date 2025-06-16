

import random
import numpy as np
import matplotlib.pyplot as plt
from cluster import cluster
from fit_2d_gaussian import fit_2d_gaussian
# from draw_mean_and_cov_on_frame import draw_mean_and_cov_on_frame
# from save_video_with_overlay import save_video_with_overlay
import cv2
import tqdm
from generate_binary_matrix_from_animation import generate_binary_matrix_from_animation

def array_set_diff(a1, a2):
    set1 = set(map(tuple, a1))
    set2 = set(map(tuple, a2))
    return np.array(list(set1 - set2))

if __name__ == "__main__":


    for i in range(10):
        i = str(i)
        np_save_path = f'/Users/lorenzo/Desktop/myStudy/crowdFollow/np_arrays/binary_animation_{i}.npy'
        output_video_path = f"/Users/lorenzo/Desktop/myStudy/crowdFollow/results/result_animation_{i}.mp4"

        binary_video = generate_binary_matrix_from_animation()

        np.save(np_save_path, binary_video)
        print("✅ binary nparray saved! shape:", binary_video.shape)

        fps = 30
        # 初始化 VideoWriter
        frame_size = (4000, 4000)  # 匹配你的图像分辨率
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 编码器
        out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

        np_frames = np.load(np_save_path)

        means = []
        covs = []
        for idx, np_frame in tqdm.tqdm(enumerate(np_frames), total=len(np_frames), desc="Processing frames"):
            position = np.argwhere(np_frame == 0)
            labels, cleaned_points = cluster(position, p_flag=False)
            others = array_set_diff(position, cleaned_points)
            mean, cov, fig = fit_2d_gaussian(others, cleaned_points, p_flag=True)
            fig.canvas.draw()

            rgba_buffer = fig.canvas.buffer_rgba()
            image_rgb = np.asarray(rgba_buffer)[:, :, :3]
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            out.write(image_bgr)

            plt.close(fig)

        out.release()
        print(f"video {i} saved!")
