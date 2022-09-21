
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import os.path

from constraint import *


def train_early_stop(uuid: str, model: Sequential, x, y, x_val, y_val, epochs, patience=50):
    early_stop_callback = EarlyStopping(
        monitor='val_loss', min_delta=0.01, patience=patience)
    save_callback = ModelCheckpoint(
        os.path.join(WEIGHT_ROOT, f"./{uuid}.weights"), monitor='val_loss', verbose=0,
        save_best_only=True, save_weights_only=True, mode='auto', period=1,
    )
    model.load_weights(os.path.join(WEIGHT_ROOT, f"./{uuid}.weights"))
    hist = model.fit(
        x, y,
        epochs=epochs, shuffle=True, batch_size=64, validation_data=(x_val, y_val),
        callbacks=[early_stop_callback, save_callback],
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
