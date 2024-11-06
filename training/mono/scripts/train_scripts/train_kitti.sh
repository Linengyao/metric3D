cd ../../../

python  mono/tools/train.py \
        mono/configs/RAFTDecoder/vit.raft5.large.kitti.py \
        --use-tensorboard \
        --launcher slurm \
        # --load-from /home/share/train_schedule/zhangdaopeng/mmyolo-main/work_dirs/Metric3D-main/weight/metric_depth_vit_small_800k.pth \
        --experiment_name set1
