import numpy as np
import pandas as pd
import pickle
import os
import random
from constant import *

# contains synergy values and fold split (numbers 0-4)
# (drug_a_name, drug_b_name, cell_line, synergy, fold)
labels = pd.read_csv(os.path.join(DATA_ROOT, 'labels.csv'), index_col=0)
# labels are duplicated for the two different ways of ordering in the data
labels = pd.concat([labels, labels])
label_fold_indexs = labels['fold']

file = open(os.path.join(DATA_ROOT, "X.p"), 'rb')
X = pickle.load(file)
file.close()

# 重新随机划分 fold 
def shuffled_folds():
    random.seed(100)
    RANDOM_FOLD_INDEX_PATH = os.path.join(DATA_ROOT, "random-fold-index.p")
    VALID_FOLDS = [i for i in range(5)]
    _index_list = []
    if not os.path.exists(RANDOM_FOLD_INDEX_PATH):
        _index_list = list(range(len(labels)))
        random.shuffle(_index_list)
        _file = open(RANDOM_FOLD_INDEX_PATH, "wb")
        pickle.dump(_index_list, _file)
        _file.close()
    if not _index_list:
        _file = open(RANDOM_FOLD_INDEX_PATH, "rb")
        _index_list = pickle.load(_file)
        _file.close()
    result = [None for _ in range(len(_index_list))]
    for fold in range(5):
        for i in _index_list[fold::5]:
            result[i] = fold if label_fold_indexs[i] in VALID_FOLDS else "Null"
    return result

def normalize(X, means1=None, std1=None, means2=None, std2=None, feat_filt=None, norm='tanh_norm'):
    if std1 is None:
        std1 = np.nanstd(X, axis=0)
    if feat_filt is None:
        feat_filt = std1 != 0
    X = X[:, feat_filt]
    X = np.ascontiguousarray(X)
    if means1 is None:
        means1 = np.mean(X, axis=0)
    X = (X - means1) / std1[feat_filt]
    if norm == 'norm':
        return (X, means1, std1, feat_filt)
    elif norm == 'tanh':
        return (np.tanh(X), means1, std1, feat_filt)
    elif norm == 'tanh_norm':
        X = np.tanh(X)
        if means2 is None:
            means2 = np.mean(X, axis=0)
        if std2 is None:
            std2 = np.std(X, axis=0)
        X = (X - means2) / std2
        X[:, std2 == 0] = 0
        return (X, means1, std1, means2, std2, feat_filt)


def split_folds(train_folds: list, valid_folds: list, norm: str):
    # indices of training data for model testing
    idx_train = np.isin(label_fold_indexs, train_folds)
    # indices of test data for model testing
    idx_test = np.isin(label_fold_indexs, valid_folds)

    X_train = X[idx_train]
    X_test = X[idx_test]

    y_train = labels.iloc[idx_train]['synergy'].values
    y_test = labels.iloc[idx_test]['synergy'].values

    X_train, mean, std, feat_filt = normalize(X_train, norm=norm)
    X_test, mean, std, feat_filt = normalize(
        X_test, mean, std, feat_filt=feat_filt, norm=norm)

    return X_train, X_test, y_train, y_test


def export_outer_data(test_fold: int, norm='tanh'):
    X_train, X_test, y_train, y_test = split_folds(
        train_folds=[x for x in range(SUM_FOLDS) if x != test_fold],
        valid_folds=[test_fold],
        norm=norm,
    )

    print(f"X_train.shape", X_train.shape)
    print(f"X_test.shape", X_test.shape)
    print(f"y_train.shape", y_train.shape)
    print(f"y_test.shape", y_test.shape)

    dest_path = os.path.join(
        DATA_ROOT, 'data_test_fold%d_%s.p' % (test_fold, norm))

    pickle.dump(
        (X_train, X_test, y_train, y_test),
        open(dest_path, 'wb'),
    )

    return dest_path


if __name__ == "__main__":
    label_fold_indexs = shuffled_folds()
    for i in range(5):
        print(f"export fold={i}, path={export_outer_data(test_fold=i)}")
    # split_folds([0, 1, 2], [3], 'tanh')
