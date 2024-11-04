

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from scipy.ndimage import zoom  # 导入缩放库

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
    labels = []
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(f"_{camera_suffix}.jpg"):
            image_path = os.path.join(image_dir, filename)
            depth_filename = filename.replace(".jpg", ".npy")
            depth_path = os.path.join(depth_dir, depth_filename)
            label_filename = filename.replace(".jpg", ".txt")
            label_path = os.path.join(image_dir, label_filename) # 标签文件和image文件在同一路径下

            if os.path.exists(depth_path):
                image = cv2.imread(image_path)
                depth = np.load(depth_path)
                depth = np.squeeze(depth)
                depth = zoom(depth, (720 / 616, 1280 / 1064), order=1)
                images.append(image)
                depths.append(depth)
                labels.append(label_path)
                
    return images, depths, labels

def load_yolo_labels(label_file, image_shape):
    """
    加载YOLOv8格式的标签文件
    :param label_file: 标签文件路径
    :param image_shape: 图像形状 (height, width)
    :return: 目标边界框列表，每个元素是一个字典，包含 'id', 'class_id' 和 'bbox'
    """
    labels = []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            x_center *= image_shape[1]
            y_center *= image_shape[0]
            width *= image_shape[1]
            height *= image_shape[0]
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
            labels.append({'id': None, 'class_id': class_id, 'bbox': (x_min, y_min, x_max - x_min, y_max - y_min)})
    return labels

def calculate_iou(bbox1, bbox2):
    """
    计算两个边界框的IOU
    :param bbox1: 第一个边界框 (x, y, w, h)
    :param bbox2: 第二个边界框 (x, y, w, h)
    :return: IOU值
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x1_min, y1_min = x1, y1
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_min, y2_min = x2, y2
    x2_max, y2_max = x2 + w2, y2 + h2

    intersection_x1 = max(x1_min, x2_min)
    intersection_y1 = max(y1_min, y2_min)
    intersection_x2 = min(x1_max, x2_max)
    intersection_y2 = min(y1_max, y2_max)

    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area

    return intersection_area / union_area

def match_bboxes_hungarian(prev_bboxes, curr_bboxes, iou_threshold=0.1):
    """
    使用匈牙利算法匹配前后两帧的边界框
    :param prev_bboxes: 前一帧的边界框列表
    :param curr_bboxes: 当前帧的边界框列表
    :param iou_threshold: IOU阈值
    :return: 匹配后的边界框列表
    """
    cost_matrix = np.zeros((len(prev_bboxes), len(curr_bboxes)))
    for i, prev_bbox in enumerate(prev_bboxes):
        for j, curr_bbox in enumerate(curr_bboxes):
            iou = calculate_iou(prev_bbox['bbox'], curr_bbox['bbox'])
            cost_matrix[i, j] = 1 - iou  # 将IOU转换为成本

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_bboxes = [None] * len(prev_bboxes)
    unmatched_curr_bboxes = []

    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < 1 - iou_threshold:
            matched_bboxes[i] = curr_bboxes[j]
        else:
            unmatched_curr_bboxes.append(curr_bboxes[j])

    # 找出未被匹配的 curr_bboxes
    matched_indices = set(col_ind)
    for idx, curr_bbox in enumerate(curr_bboxes):
        if idx not in matched_indices:
            unmatched_curr_bboxes.append(curr_bbox)

    return matched_bboxes, unmatched_curr_bboxes

def assign_ids(prev_bboxes, curr_bboxes, next_id):
    """
    动态分配目标ID
    :param prev_bboxes: 前一帧的边界框列表
    :param curr_bboxes: 当前帧的边界框列表
    :param next_id: 下一个可用的目标ID
    :return: 匹配后的边界框列表，包含ID
    """
    matched_bboxes, unmatched_curr_bboxes = match_bboxes_hungarian(prev_bboxes, curr_bboxes)
    new_bboxes = []

    # 为匹配到的边界框分配相同的ID
    for i, prev_bbox in enumerate(prev_bboxes):
        if matched_bboxes[i]:
            matched_bboxes[i]['id'] = prev_bbox['id']
            new_bboxes.append(matched_bboxes[i])

    # 为未匹配的curr_bboxes分配新ID
    for curr_bbox in unmatched_curr_bboxes:
        curr_bbox['id'] = next_id
        new_bboxes.append(curr_bbox)
        next_id += 1

    return new_bboxes, next_id

def calculate_distance(depth_map, target_bbox):
    """
    计算目标中心点的距离
    :param depth_map: 深度图
    :param target_bbox: 目标边界框 (x, y, w, h)
    :return: 目标的中心点距离
    """
    x, y, w, h = target_bbox
    center_x = x + w // 2
    center_y = y + h // 2
    if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1]:
        return depth_map[center_y, center_x]
    else:
        return None

def plot_distances(distances, target_ids, output_dir, min_frames=50):
    """
    绘制目标距离随时间变化的曲线图并保存为图片文件
    :param distances: 距离字典，键为目标ID，值为包含 (帧号, 距离值) 元组的列表
    :param target_ids: 目标ID列表
    :param output_dir: 输出目录
    :param min_frames: 最小连续帧数
    """
    plt.figure(figsize=(10, 6))
    for target_id in target_ids:
        # 提取帧号和距离值，过滤掉None值
        frame_indices = [frame for frame, distance in distances[target_id] if distance is not None]
        target_distances = [distance for _, distance in distances[target_id] if distance is not None]

        # 过滤掉连续帧数小于min_frames的目标
        if len(target_distances) >= min_frames:
            plt.plot(frame_indices, target_distances, label=f'Target {target_id}')
    
    plt.xlabel('Frame Number')
    plt.ylabel('Distance')
    plt.title('Target Distance Over Time')

    # 将图例放置在右侧并设置透明度
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.7)
    
    output_path = os.path.join(output_dir, 'target_distances.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    


def process_video(image_dir, depth_dir, label_dir, camera_suffix, output_dir):
    """
    处理视频，计算目标距离并绘制曲线图
    :param image_dir: 图像目录
    :param depth_dir: 深度图目录
    :param label_dir: 标签目录
    :param camera_suffix: 相机后缀，'1' 表示左相机，'2' 表示右相机
    :param output_dir: 输出目录
    """
    images, depths, labels = load_images_and_depths(image_dir, depth_dir, camera_suffix)
    distances = {}
    prev_bboxes = []
    next_id = 0

    # 在process_video函数内
    for i, (image, depth, label_path) in enumerate(tqdm(zip(images, depths, labels), total=len(images), desc="Processing frames")):
        if os.path.exists(label_path):
            curr_bboxes = load_yolo_labels(label_path, image.shape[:2])
            if i == 0:
                # 第一帧为每个边界框分配初始id
                for bbox in curr_bboxes:
                    bbox['id'] = next_id
                    next_id += 1
            else:
                curr_bboxes, next_id = assign_ids(prev_bboxes, curr_bboxes, next_id)
                
            for bbox in curr_bboxes:
                if bbox['id'] not in distances:
                    distances[bbox['id']] = []
                distance = calculate_distance(depth, bbox['bbox'])
                distances[bbox['id']].append((i, distance))  # 记录帧号和距离值为元组
            prev_bboxes = curr_bboxes
        else:
            for target_id in distances:
                distances[target_id].append((i, None))  # 没有检测到目标时记录None

    
    plot_distances(distances, list(distances.keys()), output_dir, min_frames=5)


def parse_npy(npy_path):
    data = np.load(npy_path)
    print(data.shape)



if __name__ == '__main__':
    # npy_path = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/depth_npy/train/2024-07-29_074234_000_1/000000_4_1.npy'
    # parse_npy(npy_path)


    # 示例调用
    image_dir = '/root/autodl-tmp/monodepth2/monodepth2/dataset/mon/mon/images/train/2024-07-29_074234_000_3'
    depth_dir = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/depth_npy/train/2024-07-29_074234_000_3'
    label_dir = '/root/autodl-tmp/monodepth2/monodepth2/dataset/mon/mon/images/train/2024-07-29_074234_000_3'
    camera_suffix = '2'  # 选择左相机
    output_dir = '/root/autodl-tmp/metric3d/Metric3D/temp'

    process_video(image_dir, depth_dir, label_dir, camera_suffix, output_dir)


