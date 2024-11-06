# import time
# import cv2
# import numpy as np
# import torch

# try:
#   from mmcv.utils import Config, DictAction
# except:
#   from mmengine import Config, DictAction

# from mono.model.monodepth_model import get_configured_monodepth_model
# def test_simple_pretrained(rgb_file, depth_file):
#     intrinsic = [5333.3333, 5333.3333, 640.0, 360.0]
#     gt_depth_scale = 256.0
#     rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]

#     #### ajust input size to fit pretrained model
#     # keep ratio resize
#     input_size = (616, 1064) # for vit model
#     # input_size = (544, 1216) # for convnext model
#     h, w = rgb_origin.shape[:2]
#     scale = min(input_size[0] / h, input_size[1] / w)
#     rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
#     # remember to scale intrinsic, hold depth
#     intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
#     # padding to input_size
#     padding = [123.675, 116.28, 103.53]
#     h, w = rgb.shape[:2]
#     pad_h = input_size[0] - h
#     pad_w = input_size[1] - w
#     pad_h_half = pad_h // 2
#     pad_w_half = pad_w // 2
#     rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
#     pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

#     #### normalize
#     mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
#     std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
#     rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
#     rgb = torch.div((rgb - mean), std)
#     rgb = rgb[None, :, :, :].cuda()

#     ###################### canonical camera space ######################
#     # inference
#     model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
#     model.cuda().eval()
#     with torch.no_grad():
#         pred_depth, confidence, output_dict = model.inference({'input': rgb})

#     # un pad
#     pred_depth = pred_depth.squeeze()
#     pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
    
#     # upsample to original size
#     pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()
#     ###################### canonical camera space ######################

#     #### de-canonical transform
#     canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
#     pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
#     pred_depth = torch.clamp(pred_depth, 0, 500)


# def test_simple_trained(rgb_file, model_path, npy_save_path):


#     # 加载本地模型

#     # cfg_file = '/root/autodl-tmp/metric3d/Metric3D/training/mono/configs/RAFTDecoder/vit.raft5.small.kitti.py'
#     # ckpt_file = '/root/autodl-tmp/metric3d/Metric3D/training/work_dirs/vit.raft5.small.kitti/20241031_160529/ckpt/step00020000.pth'
#     # cfg_file = '/root/autodl-tmp/metric3d/Metric3D/training/mono/configs/RAFTDecoder/vit.raft5.small.kitti.py'
#     # ckpt_file = '/root/autodl-tmp/metric3d/Metric3D/metric_depth_vit_small_800k.pth'
#     cfg_file = '/root/autodl-tmp/metric3d/Metric3D/mono/configs/HourglassDecoder/convtiny.0.3_150.py'
#     ckpt_file = '/root/autodl-tmp/metric3d/Metric3D/convtiny_hourglass_v1.pth'
#     cfg = Config.fromfile(cfg_file)
#     model = get_configured_monodepth_model(cfg)
#     model.load_state_dict(
#         torch.load(ckpt_file)['model_state_dict'], 
#         strict=False,
#     )


#     start_time = time.time()
#     intrinsic = [5333.3333, 5333.3333, 640.0, 360.0]
#     gt_depth_scale = 256.0
#     rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]

#     #### ajust input size to fit pretrained model
#     # keep ratio resize
#     # input_size = (616, 1064) # for vit model
#     input_size = (544, 1216) # for convnext model
#     h, w = rgb_origin.shape[:2]
#     scale = min(input_size[0] / h, input_size[1] / w)
#     rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
#     # remember to scale intrinsic, hold depth
#     intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
#     # padding to input_size
#     padding = [123.675, 116.28, 103.53]
#     h, w = rgb.shape[:2]
#     pad_h = input_size[0] - h
#     pad_w = input_size[1] - w
#     pad_h_half = pad_h // 2
#     pad_w_half = pad_w // 2
#     rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
#     pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

#     #### normalize
#     mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
#     std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
#     rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
#     rgb = torch.div((rgb - mean), std)
#     rgb = rgb[None, :, :, :].cuda()

#     ###################### canonical camera space ######################
#     # inference


#     model.cuda().eval()

#     with torch.no_grad():
#         pred_depth, confidence, output_dict = model.inference({'input': rgb})

#     # un pad
#     pred_depth = pred_depth.squeeze()
#     pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
    
#     # upsample to original size
#     pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()
#     ###################### canonical camera space ######################

#     #### de-canonical transform
#     canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
#     pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
#     pred_depth = torch.clamp(pred_depth, 0, 500)

#     end_time = time.time()
#     inference_time = end_time - start_time
#     print(f"Inference time: {inference_time:.4f} seconds")
#     np.save(npy_save_path, pred_depth.cpu().numpy())
   

# if __name__ == '__main__':
#     model_path = '/root/autodl-tmp/metric3d/Metric3D/training/work_dirs/vit.raft5.small.kitti/20241031_160529/ckpt/step00020000.pth'
#     rgb_file = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/rgb_images/images/test/2024-07-29_074234_000_5/000000_8_1.png'
#     npy_save_path= '/root/autodl-tmp/metric3d/Metric3D/test_output/test1/000000_8_1.npy'
#     test_simple_trained(rgb_file, model_path, npy_save_path)

# 2024/11/5
import os
import cv2
import torch
import numpy as np

import time
try:
  from mmcv.utils import Config, DictAction
except:
  from mmengine import Config, DictAction

from mono.model.monodepth_model import get_configured_monodepth_model
from postprocess.kf import DepthKalmanFilter 
from postprocess.postprocess import assign_ids, calculate_distance, load_yolo_labels


distances = {}
filter_distances = {}
prev_bboxes = []
next_id = 0
kalman_filters = {}

def postprocess_param_init():
    global distances, filter_distances, prev_bboxes, next_id, kalman_filters
    distances = {}
    filter_distances = {}
    prev_bboxes = []
    next_id = 0
    kalman_filters = {}

def postprocess(det, depth):
    global distances, filter_distances, prev_bboxes, next_id, kalman_filters
    curr_bboxes = det

    if len(curr_bboxes) == 0:
        for target_id in distances:
            distances[target_id].append((None, None))  # 没有检测到目标时记录None
            filter_distances[target_id].append((None, None))  # 没有检测到目标时记录None
        return

    if len(prev_bboxes) == 0:
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
        distances[bbox['id']].append((None, distance))  # 记录帧号和距离值为元组

        if distance is not None:
            if bbox['id'] not in kalman_filters:
                # 如果还没有初始化卡尔曼滤波器，则初始化
                kalman_filters[bbox['id']] = DepthKalmanFilter()
            kf = kalman_filters[bbox['id']]
            kf.predict()
            kf.update(distance)
            filtered_distance = kf.get_current_depth()
            filter_distances[bbox['id']].append((None, filtered_distance))
        else:
            filter_distances[bbox['id']].append((None, None))

    prev_bboxes = curr_bboxes



def test_simple_trained(rgb_file, model):
    start_time = time.time()
    intrinsic = [5333.3333, 5333.3333, 640.0, 360.0]
    gt_depth_scale = 256.0
    rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]

    #### ajust input size to fit pretrained model
    input_size = (544, 1216) # for convnext model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    #### normalize
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()

    ###################### canonical camera space ######################
    # inference
    model.cuda().eval()
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({'input': rgb})

    # un pad
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
    
    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()
    ###################### canonical camera space ######################

    #### de-canonical transform
    canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
    pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
    pred_depth = torch.clamp(pred_depth, 0, 500)

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.4f} seconds")
    return pred_depth

def load_model(cfg_file, ckpt_file):
    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    model.load_state_dict(torch.load(ckpt_file)['model_state_dict'], strict=False)
    return model

def process_images(input_dir, output_dir, model, camera_suffix):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            video_dir = os.path.join(root, dir_name)
            output_subdir = os.path.join(output_dir, os.path.relpath(video_dir, input_dir))
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            # 重置后处理参数
            postprocess_param_init()

            for file in sorted(os.listdir(video_dir)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) and file.endswith(camera_suffix):
                    input_file_path = os.path.join(video_dir, file)
                    output_file_path = os.path.join(output_subdir, os.path.splitext(file)[0] + '.npy')
                    label_file_path = os.path.join(video_dir, os.path.splitext(file)[0] + '.txt')

                    pred_depth = test_simple_trained(input_file_path, model)
                    # np.save(output_file_path, pred_depth.cpu().numpy())

                    # 进行后处理
                    # 加载YOLO标签文件
                    image_shape = cv2.imread(input_file_path).shape[:2]
                    det = load_yolo_labels(label_file_path, image_shape)
                    
                    start_time = time.time()
                    postprocess(det, pred_depth.cpu().numpy())
                    end_time = time.time()
                    print(f"Postprocessing time: {end_time - start_time:.4f} seconds")

if __name__ == '__main__':
    # cfg_file = '/root/autodl-tmp/metric3d/Metric3D/training/mono/configs/RAFTDecoder/vit.raft5.small.kitti.py'
    # ckpt_file = '/root/autodl-tmp/metric3d/Metric3D/metric_depth_vit_small_800k.pth'
    # input_dir = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/rgb_images/images'
    # output_dir = '/root/autodl-tmp/metric3d/Metric3D/test_output'
    # camera_suffix = '_1.png'  # 选择左相机
    # model = load_model(cfg_file, ckpt_file)
    # process_images(input_dir, output_dir, model, camera_suffix)


    cfg_file = '/root/autodl-tmp/metric3d/Metric3D/training/mono/configs/RAFTDecoder/vit.raft5.large.kitti.py'
    ckpt_file = '/root/autodl-tmp/metric3d/Metric3D/training/work_dirs/vit.raft5.large.kitti/20241105_200723/ckpt/step00020000.pth'
    # ckpt_file = '/root/autodl-tmp/metric3d/Metric3D/metric_depth_vit_small_800k.pth'
    rgb_file = '/root/autodl-tmp/metric3d/Metric3D/gt_depths/rgb_images/images/train/2024-07-29_074234_000_1/000000_4_1.png'
    output_file_path = '/root/autodl-tmp/metric3d/Metric3D/temp/000000_4_1.npy'
    model = load_model(cfg_file, ckpt_file)
    pred_depth = test_simple_trained(rgb_file,model)
    np.save(output_file_path, pred_depth.cpu().numpy())
