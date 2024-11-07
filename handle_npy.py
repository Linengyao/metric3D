import json
import os
import cv2
from matplotlib import cm
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
    
    # 调整大小
    image = image.resize((1280, 720), Image.ANTIALIAS)

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

def npy2colored_png(npy_path, output_path, colormap='viridis'):
    # 加载 .npy 文件
    data = np.load(npy_path)
    
    # 检查数据的形状
    if data.shape != (720, 1280):
        raise ValueError(f"Unexpected shape: {data.shape}. Expected (720, 1280).")
    
    # 将数据从 [0, 500] 映射到 [0, 1]
    data = data / 500.0
    
    # 使用 colormap 将数据转换为彩色图像
    cmap = cm.get_cmap(colormap)
    colored_data = cmap(data)
    
    # 将彩色图像转换为 uint8 类型
    colored_data = (colored_data[:, :, :3] * 255).astype(np.uint8)
    
    # 将数据转换为 PIL 图像对象
    image = Image.fromarray(colored_data)
    
    # 确保目标目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为 PNG 文件
    image.save(output_path)

def test():
    x, y, w, h = (597, 400, 147, 141)
    npy_path = '/root/autodl-tmp/metric3d/Metric3D/test_output/my_small_vit_pth_result/train/2024-07-29_074234_000_2/000000_5_1.npy'
    output_path = '/root/autodl-tmp/metric3d/Metric3D/test_output/my_small_vit_pth_result/test.png'
    data = np.load(npy_path)
    
    # 提取检测框区域的深度图
    depth_roi = data[y:y + h, x:x + w]
    
    # 确保深度图的最大值不为零，以免除零错误
    depth_max = np.max(depth_roi)
    if depth_max == 0:
        return {'depth': None, 'contour_points': None, 'centroid': None}  # 深度图无有效值，返回None
    
    # 使用边缘检测算法找到轮廓
    edges = cv2.Canny((depth_roi * 255 / depth_max).astype(np.uint8), 10, 20)
    # 确保轮廓包括四周的边界
    edges[0, :] = 255  # 上边界
    edges[-1, :] = 255  # 下边界
    edges[:, 0] = 255  # 左边界
    edges[:, -1] = 255  # 右边界

    # 进行形态学闭运算，以连接不连续的区域并去除小的孔洞
    kernel = np.ones((5, 5), np.uint8)  # 可以根据需要调整核大小
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)  # 增加迭代次数

    # 找到闭运算后的轮廓
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

   # 计算每个轮廓的面积
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    
    # 排序轮廓面积，找到第二大的轮廓
    if len(contour_areas) < 2:
        return {'depth': None, 'contour_points': None, 'centroid': None}  # 轮廓不足两个，返回None
    
    sorted_contours = sorted(zip(contours, contour_areas), key=lambda x: x[1], reverse=True)
    second_largest_contour = sorted_contours[1][0]
    
    # 创建一个空白图像用于绘制轮廓
    contour_image = np.zeros_like(depth_roi, dtype=np.uint8)
    cv2.drawContours(contour_image, [second_largest_contour], -1, (255, 255, 255), thickness=1)
    
    # 保存处理后的图像
    cv2.imwrite(output_path, contour_image)


if __name__ == '__main__':
    # npy_path = '/root/autodl-tmp/metric3d/Metric3D/temp/000000_4_1.npy'
    # parseSingleNpy(npy_path)

    # npy_path = '/root/autodl-tmp/metric3d/Metric3D/test_output/small_vit_pth_result/train/2024-07-29_074234_000_2/000000_5_1.npy'
    # output_path = '/root/autodl-tmp/metric3d/Metric3D/test_output/small_vit_pth_result/000000_5_1_depth.png'
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

    # img_path = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/rgb_images/images/train/2024-07-29_074234_000_1/000000_4_1.png'
    # detect_gt_file = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/rgb_images/images/train/2024-07-29_074234_000_1/000000_4_1.txt'
    # npy_path = '/root/autodl-tmp/metric3d/Metric3D/test_output/train/2024-07-29_074234_000_1/000000_4_1.npy'
    # drawRectOnSinggleImg(img_path, detect_gt_file, npy_path)

    # npy_path = '/root/autodl-tmp/metric3d/Metric3D/test_output/my_small_vit_pth_result/train/2024-07-29_074234_000_2/000000_5_1.npy'
    # output_path = '/root/autodl-tmp/metric3d/Metric3D/test_output/my_small_vit_pth_result/000000_5_1_depth.png'
    # npy2colored_png(npy_path, output_path)
    
    # 测试中间轮廓提取部分
    # test()

    img_path='/root/autodl-tmp/metric3d/Metric3D/test_output/small_vit_pth_result/000000_5_1_depth.png'
    output_path = '/root/autodl-tmp/metric3d/Metric3D/test_output/small_vit_pth_result/000000_5_1_depth_roi.png'
    x, y, w, h = (597, 400, 147, 141)
    # 读取图像
    img = cv2.imread(img_path)

    # 在指定区域画矩形框
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 保存处理后的图像
    cv2.imwrite(output_path, img)



