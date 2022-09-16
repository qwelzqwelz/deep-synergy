from tensorflow.compat.v1.keras.backend import set_session
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import os
import pickle

from constraint import *
from model_train import train_model
from model_test import predict_model, mse_metrics, classification_perf_metrics
from model_auc_metrics import regression_pr_auc, regression_roc_auc


parser = argparse.ArgumentParser()
parser.add_argument("--test_fold", type=int, default=0)
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--gpu_i", type=str, default="3")
parser.add_argument("--train_mode", action="store_true")
parser.add_argument("--threshold", type=int, default=30)
parse_result = parser.parse_args()
print(parse_result)

os.environ["CUDA_VISIBLE_DEVICES"] = parse_result.gpu_i
TEST_FOLD = parse_result.test_fold
#
file = open(os.path.join(DATA_ROOT, f"data_test_fold{TEST_FOLD}_tanh.p"), 'rb')
X_train, X_test, y_train, y_test = pickle.load(file)
file.close()

config = tf.compat.v1.ConfigProto(
    allow_soft_placement=True,
    gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
set_session(tf.compat.v1.Session(config=config))


def build_model(layers: list, lr: float, act_func, input_dropout, dropout):
    model = Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(Dense(layers[i], input_shape=(X_train.shape[1],), activation=act_func,
                            kernel_initializer='he_normal'))
            model.add(Dropout(float(input_dropout)))
        elif i == len(layers) - 1:
            model.add(Dense(layers[i], activation='linear',
                      kernel_initializer="he_normal"))
        else:
            model.add(Dense(layers[i], activation=act_func,
                      kernel_initializer="he_normal"))
            model.add(Dropout(float(dropout)))
    model.compile(loss='mean_squared_error', optimizer=SGD(
        learning_rate=lr, momentum=0.5))
    return model


def draw_history(history, *keys):
    assert keys
    fig, ax = plt.subplots(figsize=(16, 8))
    for key in keys:
        ax.plot(history.history[key], label=key)
    ax.legend()
    plt.show()


'''
入口函数
'''


def start_training(epochs=100):
    layers = [8182, 4096, 1]
    act_func = "relu"
    dropout = 0.5
    input_dropout = 0.2
    eta = 0.00001
    norm = 'tanh'

    uuid = f"{TEST_FOLD}-{norm}_{','.join([str(x) for x in layers])}_{act_func}_{dropout}_{input_dropout}_{eta}"

    model = build_model(layers, float(eta), getattr(
        tf.nn, act_func), input_dropout, dropout)
    history = train_model(uuid, model, X_train, y_train,
                          X_test, y_test, epochs)
    # draw_history(history, 'val_loss')


def start_test(threshold):
    uuid = f"{TEST_FOLD}-{norm}_{','.join([str(x) for x in layers])}_{act_func}_{dropout}_{input_dropout}_{eta}"
    model = build_model(layers, float(eta), getattr(
        tf.nn, act_func), input_dropout, dropout)

    pickle.dump(y_test, open(os.path.join(
        PICKLE_DEST_ROOT, f"{TEST_FOLD}-y_test.p"), "wb"))
    y_predict = predict_model(uuid, model, X_test)

    print(f"threshold={threshold}")
    print(mse_metrics(y_test, y_predict))

    perf_metrics = classification_perf_metrics(y_test, y_predict, threshold)
    perf_metrics.insert(0, regression_roc_auc(y_test, y_predict))
    perf_metrics.insert(1, regression_pr_auc(y_test, y_predict))
    print(perf_metrics)


if __name__ == '__main__':
    if parse_result.train_mode:
        start_training(parse_result.epoch)
    else:
        start_test(parse_result.threshold)
