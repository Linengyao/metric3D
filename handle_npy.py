import json
import os
import cv2
import numpy as np
from PIL import Image

def parseSingleNpy(npyfile_path):
    data = np.load(npyfile_path)
    # 打印数组的形状和内容
    print("Array shape:", data.shape)
    print("Array content:\n", data)

def npy2pngSingle(npy_path, output_path):
    # 加载 .npy 文件
    data = np.load(npy_path)
    
    # 检查数据的形状
    if data.shape != (1, 1, 616, 1064):
        raise ValueError(f"Unexpected shape: {data.shape}. Expected (1, 1, 616, 1064).")
    
    # 去除多余的维度
    data = np.squeeze(data)
    
    # 将数据从 [0, 500] 映射到 [0, 255]
    data = (data / 500) * 255
    
    # 将数据转换为 uint8 类型
    data = data.astype(np.uint8)
    
    # 将图片的上面10行都变为255
    data[:10, :] = 0
    
    # 将数据转换为 PIL 图像对象
    image = Image.fromarray(data)
    
    # 确保目标目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为 PNG 文件
    image.save(output_path)

def convert_npy_to_png(src_dir, dst_dir):
    # 遍历源目录
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.npy'):
                # 构建源文件路径
                npy_path = os.path.join(root, file)
                
                # 构建目标文件路径
                relative_path = os.path.relpath(root, src_dir)
                png_path = os.path.join(dst_dir, relative_path, file.replace('.npy', '.png'))
                
                # 转换并保存图像
                npy2pngSingle(npy_path, png_path)

def generate_json_file(rgb_dir, depth_dir, output_json_path, cam_in, depth_scale):
    files_list = []

    # 遍历 RGB 目录
    for root, dirs, files in os.walk(rgb_dir):
        for file in files:
            if file.endswith('.png'):
                # 构建 RGB 文件路径
                rgb_path = os.path.join(root, file)
                
                # 构建相对路径
                relative_path = os.path.relpath(root, rgb_dir)
                
                # 构建深度文件路径
                depth_file = file.replace('.png', '.png')  # 保持文件名不变
                depth_path = os.path.join(depth_dir, relative_path, depth_file)
                
                # 检查深度文件是否存在
                if os.path.exists(depth_path):
                    files_list.append({
                        'rgb': os.path.abspath(rgb_path),
                        'depth': os.path.abspath(depth_path),
                        'depth_scale': depth_scale,
                        'cam_in': cam_in
                    })

    # 构建最终的 JSON 数据
    json_data = {
        'files': files_list
    }

    # 写入 JSON 文件
    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
def convert_jpg_to_png_in_place(src_dir):
    # 遍历源目录
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.jpg'):
                # 构建源文件路径
                jpg_path = os.path.join(root, file)
                
                # 构建目标文件路径
                png_path = os.path.join(root, file.replace('.jpg', '.png'))
                
                # 打开图片并保存为 PNG 格式
                with Image.open(jpg_path) as img:
                    img.save(png_path, 'PNG')
                print(f"Converted {jpg_path} to {png_path}")
                
                # 删除原来的 jpg 文件
                os.remove(jpg_path)
                print(f"Deleted {jpg_path}")

# 在单张图片上显示距离
def drawRectOnSinggleImg(img_path, detect_gt_file, depthNpyPath):
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
   
    
    image_depth = np.load(depthNpyPath)
    image_depth = np.squeeze(image_depth)
    # print(f'原图：{image_depth.shape}')
    
    image_depth = cv2.resize(image_depth, (width, height))
    # print(f'resize后：{image_depth.shape}')


    with open(detect_gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            values = line.split(' ')
            class_id = int(values[0])
            x_center, y_center, bbox_width, bbox_height = map(float, values[1:5])

            # 将归一化的坐标转换为图像坐标
            x1 = int((x_center - bbox_width / 2) * width)
            y1 = int((y_center - bbox_height / 2) * height)
            x2 = int((x_center + bbox_width / 2) * width)
            y2 = int((y_center + bbox_height / 2) * height)
            print(f'rect:{x1},{y1},{x2},{y2}')
            # 绘制矩形
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # # 中心点坐标
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            # print(f'point:{center_x}, {center_y}, depth: {image_depth[center_y][center_x]} ')
            cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(img, f'{image_depth[center_y][center_x]}', (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imwrite('/root/autodl-tmp/metric3d/Metric3D/temp/detect_res_img1.jpg', img)


if __name__ == '__main__':
    # npy_path = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/depth_npy/train/2024-07-29_074234_000_1/000000_4_1.npy'
    # parseSingleNpy(npy_path)

    # npy_path = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/depth_npy/train/2024-07-29_074234_000_1/000000_4_1.npy'
    # output_path = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/depth_images/000000_4_1.png'
    # npy2pngSingle(npy_path, output_path)

    # src_dir = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/depth_npy'
    # dst_dir = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/depth_images'
    # convert_npy_to_png(src_dir, dst_dir)

    # rgb_dir = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/rgb_images/images/test'
    # depth_dir = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/depth_images/test'
    # output_json_path = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/test_annotations.json'
    # cam_in = [5333.3333, 5333.3333, 640.0, 360.0]
    # depth_scale = 256.0
    # generate_json_file(rgb_dir, depth_dir, output_json_path, cam_in, depth_scale)


    # src_dir = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/rgb_images/images'
    # convert_jpg_to_png_in_place(src_dir)

    img_path = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/rgb_images/images/train/2024-07-29_074234_000_1/000000_4_1.png'
    detect_gt_file = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/rgb_images/images/train/2024-07-29_074234_000_1/000000_4_1.txt'
    npy_path = '/root/autodl-tmp/metric3d/Metric3D/test_output/train/2024-07-29_074234_000_1/000000_4_1.npy'
    drawRectOnSinggleImg(img_path, detect_gt_file, npy_path)