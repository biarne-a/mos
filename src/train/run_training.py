from typing import Dict
import tensorflow as tf
from tensorflow import keras

from train.custom_recall import CustomRecall
from train.datasets import Data
from train.gru4rec_model import Gru4RecModel


def get_callbacks():
    return [keras.callbacks.TensorBoard(log_dir="logs", update_freq=100)]


def build_model(unique_train_movie_id_counts: Dict[int, int]):
    return Gru4RecModel(unique_train_movie_id_counts, loss_name="vanilla-sm")


def run_training(data: Data, nb_epochs=3, batch_size=64):
    model = build_model(data.unique_train_movie_id_counts)
    model.compile(
        optimizer=tf.keras.optimizers.Adagrad(learning_rate=5e-2),
        metrics=[CustomRecall(k=100), CustomRecall(k=500), CustomRecall(k=1000)],
        run_eagerly=False,
    )
    return model.fit(
        x=data.train_ds,
        epochs=nb_epochs,
        steps_per_epoch=data.nb_train // batch_size,
        validation_data=data.test_ds,
        validation_steps=data.nb_test // batch_size,
        callbacks=get_callbacks(),
        verbose=1,
    )
