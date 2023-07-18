from typing import List

import tensorflow as tf
from keras import utils
from tensorflow import keras

from mos.train.config import Config
from mos.train.softmaxes import MixtureOfSoftmaxes, SampledMixtureOfSoftmaxes, VanillaSoftmax


class Gru4RecModel(keras.models.Model):
    def __init__(self, movie_id_vocab: List[str], config: Config):
        super().__init__()
        self._movie_id_vocab = movie_id_vocab
        self._inverse_movie_id_lookup = tf.keras.layers.StringLookup(
            vocabulary=movie_id_vocab, invert=True, oov_token="0"
        )
        vocab_length = len(movie_id_vocab)
        self._movie_id_embedding = tf.keras.layers.Embedding(vocab_length + 1, config.embedding_dimension)
        self._gru_layer = tf.keras.layers.GRU(config.embedding_dimension)
        self._softmax = self._get_softmax(config, vocab_length)
        self._config = config

    @classmethod
    def from_config(cls, keras_config, custom_objects=None):
        json_config = keras_config.pop("config")
        config = Config(json_config)
        return Gru4RecModel(keras_config["movie_id_vocab"], config)

    def get_config(self):
        super_config = super().get_config()
        return {"movie_id_vocab": self._movie_id_vocab, "config": self._config.to_json(), **super_config}

    def _get_softmax(self, config: Config, vocab_length: int):
        if config.softmax_type == "vanilla-sm":
            return VanillaSoftmax(self._movie_id_embedding)
        if config.softmax_type == "mos":
            return MixtureOfSoftmaxes(config, self._movie_id_embedding)
        if config.softmax_type == "sampled-mos":
            return SampledMixtureOfSoftmaxes(config, self._movie_id_embedding, vocab_length)
        raise Exception(f"Unknown softmax type: {config.softmax_type}")

    def call(self, inputs, training=False):
        label = inputs["label_movie_id"]
        ctx_movie_emb = self._movie_id_embedding(inputs["context_movie_id"])
        hidden = self._gru_layer(ctx_movie_emb)
        return self._softmax(label, hidden, training)

    def train_step(self, inputs):
        # Forward pass
        with tf.GradientTape() as tape:
            _, loss_val = self(inputs, training=True)

        # Backward pass
        self.optimizer.minimize(loss_val, self.trainable_variables, tape=tape)

        return {"loss": loss_val}

    def test_step(self, inputs):
        # Forward pass
        probs, loss_val = self(inputs, training=False)

        top_indices = self._get_top_indices(probs, at_k=1000)
        # Compute metrics
        metric_results = self.compute_metrics(
            x=None, y=inputs["label_movie_id"], y_pred=top_indices, sample_weight=None
        )

        return {"loss": loss_val, **metric_results}

    def _get_top_indices(self, probs, at_k):
        return tf.math.top_k(probs, k=at_k).indices

    def predict_step(self, data):
        x, _, _ = utils.unpack_x_y_sample_weight(data)
        probs, _ = self(x, training=False)
        top_indices = self._get_top_indices(probs, at_k=100)
        predictions = self._inverse_movie_id_lookup(top_indices)
        label = self._inverse_movie_id_lookup(x["label_movie_id"])
        prev_label = tf.reshape(x["context_movie_id"][:, 0], shape=(-1, 1))
        prev_label = self._inverse_movie_id_lookup(prev_label)
        return tf.concat((prev_label, label, predictions), axis=1)
