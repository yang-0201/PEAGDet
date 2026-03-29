import os

import matplotlib.pyplot as plt
import numpy as np


def analyze_yolo_labels(label_dir, target_class=0, image_size=(1280, 720)):
    """
    分析YOLO标签中某个类别的目标位置分布
    :param label_dir: YOLO标签文件夹路径
    :param target_class: 要分析的类别ID（默认为0）
    :param image_size: 图像实际宽高（用于反归一化，可选）.
    """
    # 收集所有目标的位置
    centers_x, centers_y = [], []

    # 遍历标签文件
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        with open(os.path.join(label_dir, label_file)) as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # 跳过无效行

            class_id = int(parts[0])
            if class_id != target_class:
                continue  # 跳过非目标类别

            # 解析YOLO格式的bbox（归一化坐标）
            center_x, center_y = float(parts[1]), float(parts[2])

            # 反归一化到实际像素坐标（可选）
            center_x *= image_size[0]
            center_y *= image_size[1]

            centers_x.append(center_x)
            centers_y.append(center_y)

    # 转换为NumPy数组
    centers_x = np.array(centers_x)
    centers_y = np.array(centers_y)

    if len(centers_x) == 0:
        print(f"警告：未找到类别 {target_class} 的目标！")
        return

    # 绘制散点图（分布密度）
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(centers_x, centers_y, alpha=0.5, s=10)
    plt.title(f"Class {target_class} Distribution (N={len(centers_x)})")
    plt.xlabel("X Coordinate (pixels)")
    plt.ylabel("Y Coordinate (pixels)")
    plt.grid(True)

    # 绘制直方图（位置统计）
    plt.subplot(1, 2, 2)
    plt.hist(centers_x, bins=20, alpha=0.7, label="X Axis", color="blue")
    plt.hist(centers_y, bins=20, alpha=0.7, label="Y Axis", color="red")
    plt.title("Position Histogram")
    plt.xlabel("Coordinate Value (pixels)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    label_dir = r"E:\datasets\labels\train-adderson"  # 替换为你的标签文件夹路径
    analyze_yolo_labels(label_dir, target_class=0)  # 分析类别0的目标分布
