#!/bin/bash
mkdir ./logs
nohup /home/lizhuang@corp.sse.tongji.edu.cn/.conda/envs/tf_env/bin/python /home/lizhuang@corp.sse.tongji.edu.cn/deep-synergy/model_base.py --epoch 1000 --test_fold 0 --gpu_i 1 --train_mode 2>&1 >>./logs/train-log-0.log &
nohup /home/lizhuang@corp.sse.tongji.edu.cn/.conda/envs/tf_env/bin/python /home/lizhuang@corp.sse.tongji.edu.cn/deep-synergy/model_base.py --epoch 1000 --test_fold 1 --gpu_i 2 --train_mode 2>&1 >>./logs/train-log-1.log &
nohup /home/lizhuang@corp.sse.tongji.edu.cn/.conda/envs/tf_env/bin/python /home/lizhuang@corp.sse.tongji.edu.cn/deep-synergy/model_base.py --epoch 1000 --test_fold 2 --gpu_i 2 --train_mode 2>&1 >>./logs/train-log-2.log &
nohup /home/lizhuang@corp.sse.tongji.edu.cn/.conda/envs/tf_env/bin/python /home/lizhuang@corp.sse.tongji.edu.cn/deep-synergy/model_base.py --epoch 1000 --test_fold 3 --gpu_i 3 --train_mode 2>&1 >>./logs/train-log-3.log &
nohup /home/lizhuang@corp.sse.tongji.edu.cn/.conda/envs/tf_env/bin/python /home/lizhuang@corp.sse.tongji.edu.cn/deep-synergy/model_base.py --epoch 1000 --test_fold 4 --gpu_i 3 --train_mode 2>&1 >>./logs/train-log-4.log &