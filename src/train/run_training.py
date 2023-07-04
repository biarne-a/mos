import sys

import tensorflow as tf
from tensorflow import keras

from train.config import Config
from train.custom_recall import CustomRecall
from train.datasets import Data
from train.gru4rec_model import Gru4RecModel
from train.save_model_callback import SaveModelCallback


def get_callbacks(config: Config, model: keras.models.Model):
    return [keras.callbacks.TensorBoard(log_dir="logs", update_freq=100),
            SaveModelCallback(config, model)]


def build_model(data: Data, config: Config):
    return Gru4RecModel(data, config)


def _debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, "gettrace") and sys.gettrace() is not None


def run_training(data: Data, config: Config):
    model = build_model(data, config)
    model.compile(
        optimizer=tf.keras.optimizers.Adagrad(learning_rate=2e-1),
        metrics=[CustomRecall(k=100), CustomRecall(k=500), CustomRecall(k=1000)],
        run_eagerly=_debugger_is_active(),
    )
    return model.fit(
        x=data.train_ds,
        epochs=config.nb_epochs,
        steps_per_epoch=data.nb_train // config.batch_size,
        validation_data=data.test_ds,
        validation_steps=data.nb_test // config.batch_size,
        callbacks=get_callbacks(config, model),
        verbose=1,
    )
