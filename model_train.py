
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import os.path

from constant import *


def train_early_stop(uuid: str, model: Sequential, x, y, x_val, y_val, epochs, patience=50, cache_weights=True):
    cache_path = os.path.join(WEIGHT_ROOT, f"./{uuid}.weights")
    # callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss', min_delta=0.01, patience=patience)
    save_callback = ModelCheckpoint(
        cache_path, monitor='val_loss', verbose=0,
        save_best_only=True, save_weights_only=True, mode='auto', period=1,
    )
    callbacks = [early_stop_callback]
    if cache_weights:
        callbacks.append(save_callback)
    # load-weights
    if cache_weights and os.path.exists(cache_path) or os.path.exists(cache_path + ".index"):
        model.load_weights(cache_path)
    # train
    hist = model.fit(
        x, y,
        epochs=epochs, shuffle=True, batch_size=64, validation_data=(x_val, y_val),
        callbacks=callbacks,
    )
    return hist


def train_model(uuid: str, model: Sequential, x, y, x_val, y_val, epochs: int):
    cache_path = os.path.join(WEIGHT_ROOT, f"./{uuid}.weights")
    if os.path.exists(cache_path) or os.path.exists(cache_path + ".index"):
        model.load_weights(cache_path)
    hist = model.fit(
        x, y,
        epochs=epochs, shuffle=True, batch_size=64, validation_data=(x_val, y_val),
    )
    model.save_weights(cache_path)
    return hist
