import os

DATA_ROOT = "/home/lizhuang@corp.sse.tongji.edu.cn/data/deep-synergy"

#
WEIGHT_ROOT = "./weights"
LOG_ROOT = "./logs"

#
PICKLE_DEST_ROOT = "./predict_data"
IMAGE_DEST_ROOT = "./images"

#
layers = [8182, 4096, 1]
act_func = "relu"
dropout = 0.5
input_dropout = 0.2
eta = 0.00001
norm = 'tanh'


#
for _folder in [WEIGHT_ROOT, PICKLE_DEST_ROOT, IMAGE_DEST_ROOT]:
    if not os.path.exists(_folder):
        os.makedirs(_folder)
