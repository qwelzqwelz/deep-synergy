from sklearn.metrics import *
import pickle
import numpy as np
import os

from model_test import mse_metrics, classification_perf_metrics, v_mean_and_std_deviation
from constant import *

TARGET_METRIC = [0.92, 0.76, 0.56, 0.57, 0.95, 0.51]

def threshold_select(y, y_predict):
    result = []
    for thres in range(1, 100 + 1):
        _y = np.where(np.array(y) >= thres, 1, 0)
        _y_predict = np.where(np.array(y_predict) >= thres, 1, 0)
        _bc = balanced_accuracy_score(_y, _y_predict)
        if _bc < 0.5 or _bc >= 1.0:
            continue
        result.append((thres, _bc))
        print("\t".join([str(x) for x in result[-1]]))
    result.sort(key=lambda x: x[1], reverse=True)
    print(result[0])
    return result[0][0]




def cv_threshold_select():
    result = []
    y_list = []
    y_predict_list = []
    for i in range(5):
        y_list.append(pickle.load(open(os.path.join(PICKLE_DEST_ROOT, f"./{i}-y_test.p", "rb"))))
        y_predict_list.append(pickle.load(open(os.path.join(PICKLE_DEST_ROOT, f"./{i}-predict.p", "rb"))))
    #
    for thres in range(1, 100 + 1):
        _metric_list = []
        for i in range(5):
            _metric_list.append(
                classification_perf_metrics(
                    y_list[i], y_predict_list[i], thres)
            )
        #
        _metric_list = v_mean_and_std_deviation(_metric_list, output=False)
        mse = mse_metrics([x[0] for x in _metric_list], TARGET_METRIC)
        _metric_list.append(mse)
        result.append((thres, _metric_list))
        print(thres, "\t".join([f"{x[0]:.4f}" for x in _metric_list]))
    result.sort(key=lambda p: p[1][-1][0])
    print(result[0])
    return result[0][0]


    