

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from scipy.ndimage import zoom  # 导入缩放库
from kf import *  
def load_images_and_depths(image_dir, depth_dir, camera_suffix, image_extensions=('jpg', 'png')):
    """
    加载图像和深度图
    :param image_dir: 图像目录
    :param depth_dir: 深度图目录
    :param camera_suffix: 相机后缀，'1' 表示左相机，'2' 表示右相机
    :param image_extensions: 支持的图像文件扩展名，默认为 ('jpg', 'png')
    :return: 图像列表和深度图列表
    """
    images = []
    depths = []
    labels = []
    
    for filename in sorted(os.listdir(image_dir)):
        base_name, ext = os.path.splitext(filename)
        if base_name.endswith(f"_{camera_suffix}") and ext[1:] in image_extensions:
            image_path = os.path.join(image_dir, filename)
            depth_filename = f"{base_name}.npy"
            depth_path = os.path.join(depth_dir, depth_filename)
            label_filename = f"{base_name}.txt"
            label_path = os.path.join(image_dir, label_filename)  # 标签文件和image文件在同一路径下

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

def match_bboxes_hungarian(prev_bboxes, curr_bboxes, iou_threshold=0.2):
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

def plot_distances_with_filter(distances, filter_distances, target_ids, output_dir, min_frames=50):
    """
    绘制目标距离随时间变化的曲线图并保存为图片文件
    :param distances: 原始距离字典，键为目标ID，值为包含 (帧号, 距离值) 元组的列表
    :param filter_distances: 滤波后的距离字典，键为目标ID，值为包含 (帧号, 距离值) 元组的列表
    :param target_ids: 目标ID列表
    :param output_dir: 输出目录
    :param min_frames: 最小连续帧数
    """
    plt.figure(figsize=(10, 6))
    for target_id in target_ids:
        # 提取帧号和距离值，过滤掉None值
        frame_indices = [frame for frame, distance in distances[target_id] if distance is not None]
        target_distances = [distance for _, distance in distances[target_id] if distance is not None]
        filter_frame_indices = [frame for frame, distance in filter_distances[target_id] if distance is not None]
        filter_target_distances = [distance for _, distance in filter_distances[target_id] if distance is not None]

        # 过滤掉连续帧数小于min_frames的目标
        if len(target_distances) >= min_frames and len(filter_target_distances) >= min_frames:
            plt.plot(frame_indices, target_distances, label=f'Target {target_id} (Raw)', linestyle='--', color='red')
            plt.plot(filter_frame_indices, filter_target_distances, label=f'Target {target_id} (Filtered)', linestyle='-', color='blue')
    
    plt.xlabel('Frame Number')
    plt.ylabel('Distance')
    plt.title('Target Distance Over Time')

    # 将图例放置在右侧并设置透明度
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.7)
    
    output_path = os.path.join(output_dir, 'target_distances.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    


def plot_distances_with_range(distances, filter_distances, target_ids, output_dir, min_frames=50):
    """
    绘制目标距离随时间变化的曲线图并保存为图片文件，包含原始距离的波动范围
    :param distances: 原始距离字典，键为目标ID，值为包含 (帧号, 距离值) 元组的列表
    :param filter_distances: 滤波后的距离字典，键为目标ID，值为包含 (帧号, 距离值) 元组的列表
    :param target_ids: 目标ID列表
    :param output_dir: 输出目录
    :param min_frames: 最小连续帧数
    """
    plt.figure(figsize=(10, 6))
    for target_id in target_ids:
        # 提取帧号和距离值，过滤掉None值
        frame_indices = [frame for frame, distance in distances[target_id] if distance is not None]
        target_distances = [distance for _, distance in distances[target_id] if distance is not None]
        filter_frame_indices = [frame for frame, distance in filter_distances[target_id] if distance is not None]
        filter_target_distances = [distance for _, distance in filter_distances[target_id] if distance is not None]

        # 过滤掉连续帧数小于min_frames的目标
        if len(target_distances) >= min_frames and len(filter_target_distances) >= min_frames:
            # 计算最大最小值范围，假设偏差为固定值 20（可以根据实际情况调整）
            min_values = [d - 20 for d in target_distances]
            max_values = [d + 20 for d in target_distances]

            # 绘制原始距离的波动范围
            plt.fill_between(frame_indices, min_values, max_values, color='red', alpha=0.3, label=f'Target {target_id} (Raw Range)')

            # 绘制滤波后的距离
            plt.plot(filter_frame_indices, filter_target_distances, label=f'Target {target_id} (Filtered)', linestyle='-', color='blue')

    plt.xlabel('Frame Number')
    plt.ylabel('Distance')
    plt.title('Target Distance with Raw Range and Filtered Line')

    # 将图例放置在右侧并设置透明度
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.7)

    # 保存图片
    output_path = os.path.join(output_dir, 'target_distances_with_range.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
def plot_distances_with_rate(distances, filter_distances, target_ids, output_dir, min_frames=50):
    """
    绘制目标距离随时间变化的曲线图以及变化率，并保存为图片文件
    :param distances: 原始距离字典，键为目标ID，值为包含 (帧号, 距离值) 元组的列表
    :param filter_distances: 滤波后的距离字典，键为目标ID，值为包含 (帧号, 距离值) 元组的列表
    :param target_ids: 目标ID列表
    :param output_dir: 输出目录
    :param min_frames: 最小连续帧数
    """
    plt.figure(figsize=(12, 8))

    for target_id in target_ids:
        # 提取帧号和距离值，过滤掉None值
        frame_indices = [frame for frame, distance in distances[target_id] if distance is not None]
        target_distances = [distance for _, distance in distances[target_id] if distance is not None]
        filter_frame_indices = [frame for frame, distance in filter_distances[target_id] if distance is not None]
        filter_target_distances = [distance for _, distance in filter_distances[target_id] if distance is not None]

        # 过滤掉连续帧数小于min_frames的目标
        if len(target_distances) >= min_frames and len(filter_target_distances) >= min_frames:
            # 计算原始距离和滤波后距离的变化率
            raw_rate = [abs(target_distances[i + 1] - target_distances[i]) for i in range(len(target_distances) - 1)]
            filtered_rate = [abs(filter_target_distances[i + 1] - filter_target_distances[i]) for i in range(len(filter_target_distances) - 1)]
            rate_frame_indices = frame_indices[:-1]  # 因为变化率少一帧

            # 绘制原始距离变化率
            plt.plot(rate_frame_indices, raw_rate, linestyle='--', color='red', alpha=0.7, label=f'Target {target_id} (Raw Rate)')

            # 绘制滤波后距离变化率
            plt.plot(rate_frame_indices, filtered_rate, linestyle='-', color='blue', alpha=0.7, label=f'Target {target_id} (Filtered Rate)')

    plt.xlabel('Frame Number')
    plt.ylabel('Distance Change Rate')
    plt.title('Target Distance Change Rate Over Time')

    # 将图例放置在右侧并设置透明度
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.7)

    # 保存图片
    output_path = os.path.join(output_dir, 'target_distance_change_rate.png')
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
    filter_distances = {}
    prev_bboxes = []
    next_id = 0
    kalman_filters={}
    # 
    for i, (image, depth, label_path) in enumerate(tqdm(zip(images, depths, labels), total=len(images), desc="Processing frames")):
        if os.path.exists(label_path):
            curr_bboxes = load_yolo_labels(label_path, image.shape[:2])
            if i == 0:
                # 第一帧为每个边界框分配初始id
                for bbox in curr_bboxes:
                    bbox['id'] = next_id
                    next_id += 1
                    # 初始化卡尔曼滤波器
                    kalman_filters[bbox['id']] = DepthKalmanFilter()
            else:
                curr_bboxes, next_id = assign_ids(prev_bboxes, curr_bboxes, next_id)
                
            for bbox in curr_bboxes:
                if bbox['id'] not in distances:
                    distances[bbox['id']] = []
                    filter_distances[bbox['id']] = []

                distance = calculate_distance(depth, bbox['bbox'])
                distances[bbox['id']].append((i, distance))  # 记录帧号和距离值为元组

                if distance is not None:
                    if bbox['id'] not in kalman_filters:
                        # 如果还没有初始化卡尔曼滤波器，则初始化
                        kalman_filters[bbox['id']] = DepthKalmanFilter()
                    kf = kalman_filters[bbox['id']]
                    kf.predict()
                    kf.update(distance)
                    filtered_distance = kf.get_current_depth()
                    filter_distances[bbox['id']].append((i, filtered_distance))
                else:
                    filter_distances[bbox['id']].append((i, None))
            prev_bboxes = curr_bboxes
        else:
            for target_id in distances:
                distances[target_id].append((i, None))  # 没有检测到目标时记录None
                filter_distances[target_id].append((i, None))  # 没有检测到目标时记录None
    

    # plot_distances_with_filter(distances, filter_distances, list(distances.keys()), output_dir, min_frames=90)
    # plot_distances_with_range(distances, filter_distances, list(distances.keys()), output_dir, min_frames=90)
    plot_distances_with_rate(distances, filter_distances, list(distances.keys()), output_dir, min_frames=90)


def parse_npy(npy_path):
    data = np.load(npy_path)
    print(data.shape)



if __name__ == '__main__':
    # npy_path = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/depth_npy/train/2024-07-29_074234_000_1/000000_4_1.npy'
    # parse_npy(npy_path)


    # 示例调用
    image_dir = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/rgb_images/images/train/2024-07-29_074234_000_2'
    depth_dir = '/root/autodl-tmp/metric3d/Metric3D/test_output/train/2024-07-29_074234_000_2'
    label_dir = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/rgb_images/images/train/2024-07-29_074234_000_2'
    camera_suffix = '1'  # 选择左相机
    output_dir = '/root/autodl-tmp/metric3d/Metric3D/temp'

    process_video(image_dir, depth_dir, label_dir, camera_suffix, output_dir)


