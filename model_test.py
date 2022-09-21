from keras.models import Sequential
from keras.losses import MeanSquaredError
from sklearn.metrics import *
import numpy as np
import pickle
import os

from model_auc_metrics import classification_pr_auc, regression_pr_auc, regression_roc_auc
from constant import *

''''''


def evaluate_model(uuid: str, model: Sequential, x, y):
    model.load_weights(os.path.join(WEIGHT_ROOT, f"{uuid}.weights"))
    return model.evaluate(x, y, batch_size=1024)


def predict_model(uuid: str, model: Sequential, x, save_pre=True, dump_prefix=None):
    model.load_weights(os.path.join(WEIGHT_ROOT, f"{uuid}.weights"))
    result = model.predict(x, batch_size=1024)
    result = np.array(result).flatten()
    if save_pre:
        pickle.dump(result, open(os.path.join(
            PICKLE_DEST_ROOT, f"{dump_prefix or uuid[0]}-predict.p"), "wb"))
    return result


''''''


def mse_metrics(y, y_predict):
    # return (np.square(np.subtract(y, y_predict))).mean()
    mse = MeanSquaredError()(y, y_predict).numpy()
    rmse = np.sqrt(mse)
    return mse, rmse


def classification_perf_metrics(y, y_predict, threshold=30):
    y = np.where(np.array(y) >= threshold, 1, 0)
    y_predict = np.where(np.array(y_predict) >= threshold, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y, y_predict).ravel()
    #
    return [
        # ACC
        accuracy_score(y, y_predict),
        # BACC
        balanced_accuracy_score(y, y_predict),
        # PREC
        precision_score(y, y_predict),
        # TPR
        tp / (tp + fn),
        # TNR
        tn / (tn + fp),
        # Kappa
        cohen_kappa_score(y, y_predict),
    ]


def v_mean_and_std_deviation(data, output=True):
    data = np.array(data)
    result = []
    if output:
        for _row in data:
            print("\t".join([f"{x:.2f}" for x in _row]))
    for i in range(len(data[0])):
        _col = data[:, i].flatten()
        result.append((np.mean(_col), np.std(_col)))
    if output:
        print("\t".join([f"{x[0]:.2f}±{x[1]:.2f}" for x in result]))
    return result


def calc_cv_average_metrics(threshold):
    print(f"threshold={threshold}")
    mse_metric_mat, perform_metric_mat = [], []
    for i in range(5):
        y = pickle.load(
            open(os.path.join(PICKLE_DEST_ROOT, f"{i}-y_test.p"), "rb"))
        y_predict = pickle.load(
            open(os.path.join(PICKLE_DEST_ROOT, f"{i}-predict.p"), "rb"))
        #
        perf_metrics = [
            regression_roc_auc(y, y_predict),
            regression_pr_auc(y, y_predict),
            classification_pr_auc(y, y_predict, threshold),
        ]
        perf_metrics.extend(
            classification_perf_metrics(y, y_predict, threshold))
        #
        mse_metric_mat.append(mse_metrics(y, y_predict))
        perform_metric_mat.append(perf_metrics)
    #
    v_mean_and_std_deviation(mse_metric_mat)
    v_mean_and_std_deviation(perform_metric_mat)


if __name__ == '__main__':
    calc_cv_average_metrics(threshold=30)
    # cv_threshold_select()
