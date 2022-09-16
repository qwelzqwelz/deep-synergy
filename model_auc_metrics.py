from sklearn.metrics import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from constraint import *

''''''


def draw_auc_curve(x, y, x_label: str, y_label: str, title: str, save_path=None):
    plt.plot(x, y, 'k--', label='AUC = {0:.2f}'.format(auc(x, y)), lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
    plt.close()


def bin_threshold_map(y, y_predict, threshold):
    return (
        np.where(np.array(y) >= threshold, 1, 0),
        np.where(np.array(y_predict) >= threshold, 1, 0),
    )


def get_list_col(data: list, col: int):
    result = []
    for row in data:
        result.append(row[col])
    return result


''''''


def _regression_curve(y, y_predict, map_func):
    result = []
    for threshold in range(int(min(y_predict)), int(max(y_predict)) + 1):
        _y, _y_predict = bin_threshold_map(y, y_predict, threshold)
        # (tn, fp, fn, tp) -> (x_pos, y_pos)
        x_pos, y_pos = map_func(_y, _y_predict)
        result.append((x_pos, y_pos, threshold))
    result.sort(key=lambda x: x[0])
    return [get_list_col(result, i) for i in range(3)]


def _roc_map_func(_y, _y_predict):
    tn, fp, fn, tp = confusion_matrix(_y, _y_predict).ravel()
    # (fpr, tpr)
    return fp / (fp + tn), tp / (tp + fn)


def _pr_recall_func(_y, _y_predict):
    # (precision, recall)
    return precision_score(_y, _y_predict), recall_score(_y, _y_predict)


''''''


def regression_roc_auc(y, y_predict):
    # roc auc
    return auc(*_regression_curve(y, y_predict, _roc_map_func)[:2])


def regression_pr_auc(y, y_predict):
    # precision-recall auc
    return auc(*_regression_curve(y, y_predict, _pr_recall_func)[:2])


''''''


def plt_roc_curve(index: int, is_std=True):
    y = pickle.load(
        open(os.path.join(PICKLE_DEST_ROOT, f"{index}-y_test.p"), "rb"))
    y_predict = pickle.load(
        open(os.path.join(PICKLE_DEST_ROOT, f"{index}-predict.p"), "rb"))
    #
    if not is_std:
        fpr, tpr, thresholds = _regression_curve(y, y_predict, _roc_map_func)
    else:
        y, y_predict = bin_threshold_map(y, y_predict, 30)
        fpr, tpr, thresholds = roc_curve(y, y_predict)
    #
    title = f"test_fold_{index} ROC Curve" + ("_std" if is_std else "")
    draw_auc_curve(
        fpr, tpr, "FPR", "TPR", title,
        save_path=os.path.join(IMAGE_DEST_ROOT, f"{title}.png"),
    )
    return auc(fpr, tpr)


def plt_pre_recall_curve(index: int, is_std=True):
    y = pickle.load(
        open(os.path.join(PICKLE_DEST_ROOT, f"{index}-y_test.p"), "rb"))
    y_predict = pickle.load(
        open(os.path.join(PICKLE_DEST_ROOT, f"{index}-predict.p"), "rb"))
    #
    if not is_std:
        precision, recall, thresholds = _regression_curve(
            y, y_predict, _pr_recall_func)
    else:
        y, y_predict = bin_threshold_map(y, y_predict, 30)
        precision, recall, thresholds = precision_recall_curve(y, y_predict)
    #
    title = f"test_fold_{index} Precision-Recall Curve" + \
        ("_std" if is_std else "")
    print(thresholds)
    draw_auc_curve(
        precision, recall, "Precision", "Recall", title,
        save_path=os.path.join(IMAGE_DEST_ROOT, f"{title}.png"),
    )
    return auc(precision, recall)


if __name__ == '__main__':
    for i in range(5):
        plt_roc_curve(i, is_std=True)
