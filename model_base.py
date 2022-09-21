from tensorflow.compat.v1.keras.backend import set_session
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import os
import pickle
import json
import numpy as np

from constant import *
from model_train import train_model, train_early_stop
from model_test import predict_model, mse_metrics
from normalize import split_folds


parser = argparse.ArgumentParser()
parser.add_argument("--test_fold", type=int, default=0)
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--patience", type=int, default=50)
parser.add_argument("--gpu_i", type=str, default="3")
parser.add_argument("--train_mode", action="store_true")
parser.add_argument("--test_mode", action="store_true")
parser.add_argument("--cv_mode", action="store_true")
parser.add_argument("--hp_name", type=str, default="official-hp.json")
parse_result = parser.parse_args()
print("[args]", parse_result)

os.environ["CUDA_VISIBLE_DEVICES"] = parse_result.gpu_i
TEST_FOLD = parse_result.test_fold

config = tf.compat.v1.ConfigProto(
    allow_soft_placement=True,
    gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
set_session(tf.compat.v1.Session(config=config))

'''
data
'''


def load_data_from_disk(test_fold: int, norm: str):
    file = open(os.path.join(
        DATA_ROOT, f"data_test_fold{test_fold}_{norm}.p"), 'rb')
    X_train, X_test, y_train, y_test = pickle.load(file)
    file.close()
    return X_train, X_test, y_train, y_test


def load_data_from_mem(avaiable_folds: list, valid_fold: int, norm: str):
    # X_train, X_test, y_train, y_test
    return split_folds(
        train_folds=[x for x in avaiable_folds if x != valid_fold],
        valid_folds=[valid_fold],
        norm=norm,
    )


'''
hyper-parameters
'''


class HyperParameters:
    SP_KEYS = ("layers", "eta", "act_func", "dropout", "input_dropout", "norm")

    def __init__(self, layers: list, act_func: str, dropout: float, input_dropout: float, eta: float, norm: str):
        self.layers = layers
        self.act_func = act_func
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.eta = eta
        self.norm = norm

    def __str__(self) -> str:
        return str(self.p_dict())

    def uuid(self, with_prefix=True) -> str:
        result = [
            self.norm,
            ','.join([str(x) for x in self.layers]),
            self.act_func,
            self.dropout,
            self.input_dropout,
            self.eta,
        ]
        result = '_'.join([str(x) for x in result])
        if with_prefix:
            result = f"{TEST_FOLD}-{result}"
        return result

    def p_dict(self) -> dict:
        result = [getattr(self, key) for key in self.SP_KEYS]
        result = dict(zip(self.SP_KEYS, result))
        return result

    def dump(self, f_name: str):
        path = os.path.join(HP_ROOT, f_name)
        json.dump(self.p_dict(), open(path, "wt"))
        return path

    @staticmethod
    def load_from(f_name: str):
        path = os.path.join(HP_ROOT, f_name)
        assert os.path.exists(path)
        return HyperParameters(**json.load(open(path, "rt")))


'''
model
'''


def build_model(hp: HyperParameters):
    model = Sequential()
    act_func = getattr(tf.nn, hp.act_func)
    # 隐含层
    for i in range(len(hp.layers)):
        if i == 0:
            model.add(Dense(hp.layers[i], input_shape=(8846,), activation=act_func,
                            kernel_initializer='he_normal'))
            model.add(Dropout(float(hp.input_dropout)))
        elif i == len(hp.layers) - 1:
            model.add(Dense(hp.layers[i], activation='linear',
                      kernel_initializer="he_normal"))
        else:
            model.add(Dense(hp.layers[i], activation=act_func,
                      kernel_initializer="he_normal"))
            model.add(Dropout(float(hp.dropout)))
    # 编译
    model.compile(loss='mean_squared_error', optimizer=SGD(
        learning_rate=hp.eta, momentum=0.5))
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

DEFAULT_EPOCHS = 100


def _sp_meshgrid():
    result = [[]]
    for sp_list in HP_SPACE.values():
        new_result = []
        for hp in sp_list:
            for combi in result:
                elem = combi.copy()
                elem.append(hp)
                new_result.append(elem)
        result = new_result
    return result


def start_training(hp: HyperParameters, epochs=None):
    epochs = epochs or DEFAULT_EPOCHS
    model = build_model(hp)
    #
    X_train, X_test, y_train, y_test = load_data_from_disk(TEST_FOLD, hp.norm)
    history = train_model(hp.uuid(), model, X_train,
                          y_train, X_test, y_test, epochs)
    # draw_history(history, 'val_loss')


def start_test(hp: HyperParameters):
    X_train, X_test, y_train, y_test = load_data_from_disk(TEST_FOLD, hp.norm)
    model = build_model(hp)
    pickle.dump(y_test, open(os.path.join(
        PICKLE_DEST_ROOT, f"{TEST_FOLD}-y_test.p"), "wb"))
    predict_model(hp.uuid(), model, X_test)


def start_inner_cv_search(epochs=None, patience=50):
    epochs = epochs or DEFAULT_EPOCHS
    mse_dict = dict()
    for sp_list in _sp_meshgrid():
        hp = HyperParameters(**dict(zip(HP_SPACE.keys(), sp_list)))
        avaiable_folds = [x for x in range(SUM_FOLDS) if x != TEST_FOLD]
        mse_list = []
        for valid_fold in avaiable_folds:
            # 构造数据
            X_train, X_test, y_train, y_test = load_data_from_mem(
                avaiable_folds, valid_fold, hp.norm)
            model = build_model(hp)
            uuid = f"cv_{valid_fold}_{hp.uuid()}"
            # 训练
            print(
                f"[start] TEST_FOLD={TEST_FOLD}, avaible_folds={avaiable_folds}, valid_fold={valid_fold}")
            print("[start] hp:", hp.p_dict())
            train_early_stop(uuid, model, X_train, y_train,
                             X_test, y_test, epochs, patience, cache_weights=True)
            # 测试
            y_predict = np.array(model.predict(
                X_test, batch_size=1024)).flatten()
            mse, rmse = mse_metrics(y_test, y_predict)
            mse_list.append(mse)
            print(
                f"[end] TEST_FOLD={TEST_FOLD}, avaible_folds={avaiable_folds}, valid_fold={valid_fold}")
            print("[end] hp:", hp.p_dict())
            print(f"mse={mse}, rmse={rmse}")
            # 删除变量
            del X_train, X_test, y_train, y_test, model
        mse_dict[hp] = mse_list
    # 排序
    result = list(mse_dict.items())
    result.sort(key=lambda p: np.mean(p[1]))
    # 输出信息
    for _sp, _mse_list in result:
        print(_sp.p_dict(), "=>", _mse_list, "=>", np.mean(_mse_list))
    result = result[0][0]
    print("best-hp:", result)
    result.dump(f"cv-{TEST_FOLD}-best-hp.json")
    return result


if __name__ == '__main__':
    assert parse_result.train_mode or parse_result.test_mode or parse_result.cv_mode

    HP = HyperParameters.load_from(parse_result.hp_name)

    if parse_result.train_mode:
        print(f"test_fold={TEST_FOLD}, hyper-parameters={HP}")
        start_training(HP, epochs=parse_result.epoch)
    elif parse_result.test_mode:
        print(f"test_fold={TEST_FOLD}, hyper-parameters={HP}")
        start_test(HP)
    else:
        start_inner_cv_search(parse_result.epoch, parse_result.patience)
