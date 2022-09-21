import os

DATA_ROOT = "/home/lizhuang@corp.sse.tongji.edu.cn/data/deep-synergy"

#
WEIGHT_ROOT = "./weights"
LOG_ROOT = "./logs"
HP_ROOT = "./hyper-params"

#
PICKLE_DEST_ROOT = "./predict_data"
IMAGE_DEST_ROOT = "./images"

#
SUM_FOLDS = 5

# HP_SPACE = {
#     "layers": [[8192, 8192, 1], [4096, 4096, 1], [2048, 2048, 1],
#                [8192, 4096, 1], [4096, 2048, 1], [4096, 4096, 4096, 1],
#                [2048, 2048, 2048, 1], [4096, 2048, 1024, 1],
#                [8192, 4096, 2048, 1]],
#     "eta": [0.01, 0.001, 0.0001, 0.00001],
#     "act_func": ["relu"],
#     "dropout": [0.5],
#     "input_dropout": [0.2],
#     "norm": ["norm", "tanh", "tanh_norm"],
# }

HP_SPACE = {
    "layers": [[8192, 4096, 1], [4096, 2048, 1], [2048, 1024, 1]],
    "eta": [0.00001, 0.0001],
    "act_func": ["relu"],
    "dropout": [0.5],
    "input_dropout": [0.2],
    "norm": ["tanh"],
}


#
for _folder in [WEIGHT_ROOT, PICKLE_DEST_ROOT, IMAGE_DEST_ROOT, HP_ROOT]:
    if not os.path.exists(_folder):
        os.makedirs(_folder)
