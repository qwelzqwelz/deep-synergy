import os

DATA_ROOT = "/home/lizhuang@corp.sse.tongji.edu.cn/data/deep-synergy"

#
WEIGHT_ROOT = "./weights"
LOG_ROOT = "./logs"

#
PICKLE_DEST_ROOT = "./predict_data"
IMAGE_DEST_ROOT = "./images"

#
SUM_FOLDS = 5

# SP_SPACE = {
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

SP_SPACE = {
    "layers": [[8192, 8192, 1], [4096, 4096, 1], [2048, 2048, 1]],
    "eta": [0.00001, 0.001, 0.01],
    "act_func": ["relu"],
    "dropout": [0.5],
    "input_dropout": [0.2],
    "norm": ["tanh"],
}

BEST_SP = {
    "layers": [8182, 4096, 1],
    "act_func": "relu",
    "dropout": 0.5,
    "input_dropout": 0.2,
    "eta": 0.00001,
    "norm": "tanh",
}

#
for _folder in [WEIGHT_ROOT, PICKLE_DEST_ROOT, IMAGE_DEST_ROOT]:
    if not os.path.exists(_folder):
        os.makedirs(_folder)
