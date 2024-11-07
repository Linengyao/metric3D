# 这个函数用来处理推理时间的，画成折线图
import json
import os

import cv2
from matplotlib import pyplot as plt
import numpy as np

# 该函数是画出某段视频的推理和后处理时间图的
def handle_process_time(json_file_path, output_path):
    with open(json_file_path, 'r') as f:
        results = json.load(f)

    # 提取特定视频片段的数据
    video_segment = '2024-07-29_074234_000_2'
    filtered_results = [result for result in results if video_segment in result['unique_identifier']]

    # 提取帧序号、推理时间和后处理时间
    frame_numbers = [result['frame_number'] for result in filtered_results]
    inference_times = [result['inference_time'] for result in filtered_results]
    postprocess_times = [result['postprocess_time'] for result in filtered_results]

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = range(len(frame_numbers))

    bars1 = ax.bar(index, inference_times, bar_width, label='Inference Time', alpha=0.7)
    bars2 = ax.bar([i + bar_width for i in index], postprocess_times, bar_width, label='Postprocess Time', alpha=0.7)

    # 计算每个柱状图的中点
    mid_points_inference = [(i + bar_width / 2, inference_times[i]) for i in index]
    mid_points_postprocess = [(i + bar_width * 1.5, postprocess_times[i]) for i in index]

    # 绘制折线
    x_inference, y_inference = zip(*mid_points_inference)
    x_postprocess, y_postprocess = zip(*mid_points_postprocess)
    ax.plot(x_inference, y_inference, '-', label='Inference Time Curve', color='blue', alpha=0.7)
    ax.plot(x_postprocess, y_postprocess, '-', label='Postprocess Time Curve', color='orange', alpha=0.7)

    # 设置图表标题和标签
    ax.set_title(f'Time Analysis for Video Segment {video_segment}')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Time (seconds)')

    # 只显示第一帧和最后一帧的序号
    if frame_numbers:
        ax.set_xticks([0, len(frame_numbers) - 1])
        ax.set_xticklabels([frame_numbers[0], frame_numbers[-1]])

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1.1), prop={'size':5})


    # 调整边框样式
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)



    # 调整箭头位置
    ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)

    # 调整布局
    plt.tight_layout()
  
    
    # 保存为图片
    plt.savefig(output_path, dpi=300)
    # 关闭图表以释放内存
    plt.close(fig)
    print(f"Chart saved to {output_path}")


def handle_pick_depth(image_path, json_file_path, output_image_path):
    """
    读取 JSON 文件并在图像上绘制边界框、轮廓和深度点
    :param image_path: 图像路径
    :param json_file_path: JSON 文件路径
    :param output_image_path: 输出图像路径
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # 读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

     # 获取当前图像的帧数据
    frame_data = results['frame_data']
    image_filename = os.path.basename(image_path)
    frame_info = next((frame for frame in frame_data if os.path.basename(frame['frame']) == image_filename), None)
    if frame_info is None:
        raise ValueError(f"No frame data found for image: {image_path}")

    # 绘制边界框、轮廓和深度点
    for bbox in frame_info['bboxes']:
        bbox_coords = bbox['bbox']
        chose_dict = bbox['chose_dict']
        depth = chose_dict.get('depth', None)
        contour_points = chose_dict.get('contour_points', [])
        centroid = chose_dict.get('centroid', None)

        # 绘制边界框
        if bbox_coords:
            cv2.rectangle(image, (bbox_coords[0], bbox_coords[1]), (bbox_coords[0] + bbox_coords[2], bbox_coords[1] + bbox_coords[3]), (0, 255, 0), 2)

        # 绘制轮廓
        if contour_points:
            contour_points = np.array(contour_points, dtype=np.int32)
            cv2.drawContours(image, [contour_points], -1, (0, 0, 255), 2)

        # 绘制深度点
        if centroid and depth is not None:
            center_x, center_y = centroid
            cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)
            cv2.putText(image, f"Depth: {depth*3:.2f}", (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 保存输出图像
    cv2.imwrite(output_image_path, image)
    print(f"Output image saved to: {output_image_path}")
    

if __name__ == '__main__':
    # # 读取JSON文件
    # json_file_path = '/root/autodl-tmp/metric3d/Metric3D/test_output/my_small_vit_pth_result/results.json'
    # output_path = '/root/autodl-tmp/metric3d/Metric3D/test_output/my_small_vit_pth_result/time_analysis.png'
    # handle_process_time(json_file_path, output_path)
    
    image_path = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/rgb_images/images/train/2024-07-29_074234_000_2/000000_5_1.png'
    json_file_path = '/root/autodl-tmp/metric3d/Metric3D/test_output/my_small_vit_pth_result/2024-07-29_074234_000_2.json' 
    output_image_path = '/root/autodl-tmp/metric3d/Metric3D/test_output/my_small_vit_pth_result/000000_5_1.png'
    handle_pick_depth(image_path, json_file_path, output_image_path)

