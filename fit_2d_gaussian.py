

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def fit_2d_gaussian(others, cleaned_points, p_flag=False):
    """
    将输入的二维点集拟合为一个二维正态分布。
    
    参数:
        cleaned_points (np.ndarray or list): 形状为 (N, 2) 的二维点集
        
    返回:
        mean (np.ndarray): 均值向量 [mu_x, mu_y]
        cov (np.ndarray): 协方差矩阵 [[var_x, cov_xy],
                                      [cov_xy, var_y]]
    """
    cleaned_points = np.array(cleaned_points)
    
    if len(cleaned_points.shape) != 2 or cleaned_points.shape[1] != 2:
        raise ValueError("输入必须是一个形状为 (N, 2) 的二维数组")

    if len(cleaned_points) < 2:
        raise ValueError("至少需要两个点才能计算协方差矩阵")

    # 计算均值和协方差矩阵
    mean = np.mean(cleaned_points, axis=0)
    cov = np.cov(cleaned_points, rowvar=False)

    if p_flag == True:
        fig = visualize_2d_gaussian(others, cleaned_points, mean, cov, title="Group Tracking")

    return mean, cov, fig


# def visualize_2d_gaussian(others, cleaned_points, mean, cov, title="2D Gaussian Distribution"):
#     """
#     可视化二维正态分布：显示原始点、均值、协方差椭圆
    
#     参数:
#         cleaned_points (np.ndarray or list): 去噪后的二维点集 (N, 2)
#         mean (np.ndarray): 均值 [mu_x, mu_y]
#         cov (np.ndarray): 协方差矩阵 [[var_x, cov_xy], [cov_xy, var_y]]
#         title (str): 图像标题
#     """
#     others = np.array(others)
#     cleaned_points = np.array(cleaned_points)
#     mean = np.array(mean).ravel()
    
#     # 设置图像尺寸为 4000x4000 像素
#     figsize = (8, 8)  # 40 英寸 x 40 英寸
#     dpi = 500           # 每英寸像素数

#     fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

#     # 绘制原始点
#     ax.scatter(cleaned_points[:, 0], cleaned_points[:, 1], c='blue', s=30, label='Cleaned cleaned_points')
#     if len(others) > 0:
#         ax.scatter(others[:, 0], others[:, 1], c='black', s=30, label='others')
    
#     # 绘制均值点
#     # ax.scatter(mean[0], mean[1], c='red', s=100, marker='x', label='Mean (μ)')
#     ax.scatter(mean[0], mean[1], c='red', s=100, marker='x')

    
#     # 计算椭圆参数
#     v, w = np.linalg.eigh(cov)  # 特征值和特征向量
#     v = 2. * np.sqrt(2.) * np.sqrt(v)  # 用 2σ 表示置信区间约 95%
#     u = w[0] / np.linalg.norm(w[0])
    
#     # 如果特征值是负数或奇异矩阵，则跳过绘制椭圆
#     if not np.any(np.isnan(v)) and not np.any(np.isinf(v)):
#         angle = np.arctan2(u[1], u[0])  # 椭圆旋转角度
#         angle = np.degrees(angle)      # 转换为角度
        
#         # 创建椭圆对象
#         # ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=angle,
#         #               edgecolor='green', facecolor='none', lw=2, linestyle='--', label='Covariance Ellipse')
#         ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=angle,
#                       edgecolor='green', facecolor='none', lw=2, linestyle='--')
#         ax.add_patch(ell)
    
#     ax.set_xlim(0, 4000)
#     ax.set_ylim(0, 4000)
#     # 设置图像信息
#     ax.set_title(title)
#     ax.set_xlabel("X Position")
#     ax.set_ylabel("Y Position")
#     ax.legend()
#     ax.grid(True)
#     ax.axis('equal')  # 确保椭圆不被拉伸
    
#     # plt.show()
#     return fig


from matplotlib.patches import Rectangle
def visualize_2d_gaussian(others, cleaned_points, mean, cov, title="Group Tracking"):
    others = np.array(others)
    cleaned_points = np.array(cleaned_points)
    mean = np.array(mean).ravel()

    figsize = (8, 8)
    dpi = 500
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.scatter(cleaned_points[:, 0], cleaned_points[:, 1], c='blue', s=30, label='Cleaned Points')
    if len(others) > 0:
        ax.scatter(others[:, 0], others[:, 1], c='black', s=30, label='Outliers')

    ax.scatter(mean[0], mean[1], c='red', s=100, marker='x', label='Mean')

    min_x, max_x = np.min(cleaned_points[:, 0]), np.max(cleaned_points[:, 0])
    min_y, max_y = np.min(cleaned_points[:, 1]), np.max(cleaned_points[:, 1])

    width = max_x - min_x
    height = max_y - min_y
    square_side = max(width, height)

    center_x, center_y = mean
    half_side = square_side / 2
    square_left = center_x - half_side
    square_bottom = center_y - half_side

    square = Rectangle(
        xy=(square_left, square_bottom),
        width=square_side,
        height=square_side,
        edgecolor='magenta',
        facecolor='none',
        lw=2,
        linestyle='-'
    )
    ax.add_patch(square)

    ax.set_xlim(0, 4000)
    ax.set_ylim(0, 4000)

    ax.set_box_aspect(1)

    ax.set_title(title)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    ax.grid(True)

    return fig