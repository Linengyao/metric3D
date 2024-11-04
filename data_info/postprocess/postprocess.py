import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_images_and_depths(image_dir, depth_dir, camera_suffix):
    """
    加载图像和深度图
    :param image_dir: 图像目录
    :param depth_dir: 深度图目录
    :param camera_suffix: 相机后缀，'1' 表示左相机，'2' 表示右相机
    :return: 图像列表和深度图列表
    """
    images = []
    depths = []
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(f"_{camera_suffix}.jpg"):
            image_path = os.path.join(image_dir, filename)
            depth_filename = filename.replace(".jpg", ".npy")
            depth_path = os.path.join(depth_dir, depth_filename)
            if os.path.exists(depth_path):
                image = cv2.imread(image_path)
                depth = np.load(depth_path)
                images.append(image)
                depths.append(depth)
    return images, depths

def calculate_distance(depth_map, target_bbox):
    """
    计算目标的距离
    :param depth_map: 深度图
    :param target_bbox: 目标边界框 (x, y, w, h)
    :return: 目标的平均距离
    """
    x, y, w, h = target_bbox
    depth_slice = depth_map[0, 0, y:y+h, x:x+w]
    valid_depths = depth_slice[depth_slice > 0]
    if valid_depths.size > 0:
        return np.mean(valid_depths)
    else:
        return None

def plot_distances(distances, target_ids):
    """
    绘制目标距离随时间变化的曲线图
    :param distances: 距离字典，键为目标ID，值为距离列表
    :param target_ids: 目标ID列表
    """
    plt.figure(figsize=(10, 6))
    for target_id in target_ids:
        plt.plot(distances[target_id], label=f'Target {target_id}')
    plt.xlabel('Frame Number')
    plt.ylabel('Distance')
    plt.title('Target Distance Over Time')
    plt.legend()
    plt.show()

def process_video(image_dir, depth_dir, camera_suffix, target_bboxes):
    """
    处理视频，计算目标距离并绘制曲线图
    :param image_dir: 图像目录
    :param depth_dir: 深度图目录
    :param camera_suffix: 相机后缀，'1' 表示左相机，'2' 表示右相机
    :param target_bboxes: 目标边界框列表，每个元素是一个字典，包含 'id' 和 'bbox'
    """
    images, depths = load_images_and_depths(image_dir, depth_dir, camera_suffix)
    distances = {target['id']: [] for target in target_bboxes}
    
    for i, (image, depth) in enumerate(zip(images, depths)):
        for target in target_bboxes:
            distance = calculate_distance(depth, target['bbox'])
            if distance is not None:
                distances[target['id']].append(distance)
            else:
                distances[target['id']].append(None)
    
    plot_distances(distances, [target['id'] for target in target_bboxes])

# 示例调用
image_dir = '/root/autodl-tmp/monodepth2/monodepth2/dataset/mon/mon/images/train/2024-07-29_074234_000_1'
depth_dir = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/depth_npy/train/2024-07-29_074234_000_1'
camera_suffix = '1'  # 选择左相机
target_bboxes = [
    {'id': 1, 'bbox': (100, 100, 50, 50)},
    {'id': 2, 'bbox': (200, 200, 60, 60)}
]

process_video(image_dir, depth_dir, camera_suffix, target_bboxes)


def parse_npy(npy_path):
    data = np.load(npy_path)
    print(data.shape)

if __name__ == '__main__':
    npy_path = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/depth_npy/train/2024-07-29_074234_000_1/000000_4_1.npy'
    parse_npy(npy_path)