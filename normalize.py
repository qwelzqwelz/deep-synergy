import numpy as np
import pandas as pd
import pickle
import os

from constant import *

file = open(os.path.join(DATA_ROOT, "X.p"), 'rb')
X = pickle.load(file)
file.close()

# contains synergy values and fold split (numbers 0-4)
# (drug_a_name, drug_b_name, cell_line, synergy, fold)
labels = pd.read_csv(os.path.join(DATA_ROOT, 'labels.csv'), index_col=0)
# labels are duplicated for the two different ways of ordering in the data
labels = pd.concat([labels, labels])


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
    idx_train = np.isin(labels['fold'], train_folds)
    # indices of test data for model testing
    idx_test = np.isin(labels['fold'], valid_folds)

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

    dest_path = os.path.join(
        DATA_ROOT, 'data_test_fold%d_%s.p' % (test_fold, norm))

    pickle.dump(
        (X_train, X_test, y_train, y_test),
        open(dest_path, 'wb'),
    )


if __name__ == "__main__":
    # for i in range(5):
    # export_outer_data(test_fold=i)
    split_folds([0, 1, 2], [3], 'tanh')
