cd ../../../

python  mono/tools/test.py \
        mono/configs/RAFTDecoder/vit.raft5.large.kitti.py \
        --load-from /home/share/train_schedule/zhangdaopeng/mmyolo-main/work_dirs/Metric3D-main/training/work_dirs/vit.raft5.large.kitti/20241101_201415/ckpt/step00004000.pth
