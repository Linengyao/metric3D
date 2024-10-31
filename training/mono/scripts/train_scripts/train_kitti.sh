#!/bin/bash

# 切换到项目根目录
cd ../../../

# 启动训练
python mono/tools/train.py \
  mono/configs/RAFTDecoder/vit.raft5.small.kitti.py \
  --use-tensorboard \
  --experiment_name set1 \
  --seed 42 \
  --launcher None \
#   --load-from None \
#   --resume-from None
  