from keras.models import Sequential
from keras.losses import MeanSquaredError
from sklearn.metrics import *
import numpy as np
import pickle
import os


from constraint import *

''''''


def evaluate_model(uuid: str, model: Sequential, x, y):
    model.load_weights(os.path.join(WEIGHT_ROOT, f"{uuid}.weights"))
    return model.evaluate(x, y, batch_size=1024)


def predict_model(uuid: str, model: Sequential, x, save_pre=True):
    model.load_weights(os.path.join(WEIGHT_ROOT, f"{uuid}.weights"))
    result = model.predict(x, batch_size=1024)
    result = np.array(result).flatten()
    if save_pre:
        pickle.dump(result, open(os.path.join(
            PICKLE_DEST_ROOT, f"{uuid[0]}-predict.p"), "wb"))
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


def print_average_metrics_from_log():
    mse_metrics, perform_metrics = [], []
    for i in range(5):
        _path = os.path.join(LOG_ROOT, f"./test-log-{i}.log")
        with open(_path, "rt", encoding="utf8") as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]
            mse_metrics.append(eval(lines[-2]))
            perform_metrics.append(eval(lines[-1]))
    v_mean_and_std_deviation(mse_metrics)
    v_mean_and_std_deviation(perform_metrics)


if __name__ == '__main__':
    print_average_metrics_from_log()
    # cv_threshold_select()
