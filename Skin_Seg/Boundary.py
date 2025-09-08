import os
import cv2
import numpy as np
from tqdm import tqdm


def extract_aligned_widened_edges(mask, expansion_width=3):
    """
    修正版：确保expansion_width参数有效控制拓宽宽度
    :param mask: 输入二值mask(0或255)
    :param expansion_width: 边缘拓宽的像素宽度（实际宽度≈expansion_width+1）
    :return: 外边缘与原始mask一致的拓宽边缘图
    """
    # 严格二值化
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 步骤1：获取原始mask的精确外边缘
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    original_edge = np.zeros_like(binary)
    cv2.drawContours(original_edge, contours, -1, 255, 1)

    # 步骤2：动态调整结构元素大小（关键修改）
    kernel_size = max(3, 2 * expansion_width - 1)  # 确保核足够大
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 步骤3：精确控制侵蚀程度（关键修改）
    eroded = cv2.erode(binary, kernel, iterations=1)  # 固定1次迭代

    # 步骤4：计算拓宽区域
    widened_area = cv2.subtract(binary, eroded)

    # 步骤5：将原始外边缘合并到拓宽区域
    result = cv2.bitwise_or(widened_area, original_edge)

    return result


# 路径设置
input_path = r'D:\Segmentation\Skin_Seg\data\PH2\val\masks'
output_path = r'D:\Segmentation\Skin_Seg\data\PH2\val\boundary_mask'
os.makedirs(output_path, exist_ok=True)

# 处理流程
for filename in tqdm(os.listdir(input_path)):
    if filename.lower().endswith(('.png', '.jpg', '.tif', '.bmp')):
        try:
            # 读取图像
            mask = cv2.imread(os.path.join(input_path, filename), cv2.IMREAD_GRAYSCALE)

            # 获取对齐拓宽边缘
            edges = extract_aligned_widened_edges(mask, expansion_width=10)

            # 保存结果
            cv2.imwrite(os.path.join(output_path, filename), edges)
        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")

print(f"边缘拓宽完成！结果保存在 {output_path}")