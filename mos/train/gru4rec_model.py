import tensorflow as tf
from keras.engine import data_adapter
from tensorflow import keras

from mos.train.config import Config
from mos.train.datasets import Data
from mos.train.softmaxes import MixtureOfSoftmaxes, VanillaSoftmax


class Gru4RecModel(keras.models.Model):
    def __init__(self, data: Data, config: Config):
        super().__init__()
        movie_id_vocab = list(data.movie_id_counts.keys())
        self._movie_id_lookup = tf.keras.layers.StringLookup(vocabulary=movie_id_vocab)
        self._inverse_movie_id_lookup = tf.keras.layers.StringLookup(
            vocabulary=movie_id_vocab, invert=True, oov_token="0"
        )
        self._movie_id_embedding = tf.keras.layers.Embedding(len(data.movie_id_counts) + 1, config.embedding_dimension)
        self._gru_layer = tf.keras.layers.GRU(config.embedding_dimension)
        self._softmax = self._get_softmax(config)
        self._loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self._config = config

    def _get_softmax(self, config):
        if config.softmax_type == "mos":
            return MixtureOfSoftmaxes(config, self._movie_id_embedding)
        if config.softmax_type == "vanilla-sm":
            return VanillaSoftmax(self._movie_id_embedding)
        raise Exception(f"Unknown softmax type: {config.softmax_type}")

    def call(self, inputs, training=False):
        ctx_movie_idx = self._movie_id_lookup(inputs["context_movie_id"])
        ctx_movie_emb = self._movie_id_embedding(ctx_movie_idx)
        hidden = self._gru_layer(ctx_movie_emb)
        return self._softmax(hidden)

    def train_step(self, inputs):
        # Forward pass
        with tf.GradientTape() as tape:
            probs = self(inputs, training=True)
            label = self._movie_id_lookup(inputs["label_movie_id"])
            loss_val = self._loss(label, probs)

        # Backward pass
        self.optimizer.minimize(loss_val, self.trainable_variables, tape=tape)

        return {"loss": loss_val}

    def test_step(self, inputs):
        # Forward pass
        probs = self(inputs, training=False)
        label = self._movie_id_lookup(inputs["label_movie_id"])
        loss_val = self._loss(label, probs)

        top_indices = self._get_top_indices(probs, at_k=1000)
        # Compute metrics
        metric_results = self.compute_metrics(x=None, y=label, y_pred=top_indices, sample_weight=None)

        return {"loss": loss_val, **metric_results}

    def _get_top_indices(self, probs, at_k):
        return tf.math.top_k(probs, k=at_k).indices

    def predict_step(self, data):
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        probs = self(x, training=False)
        top_indices = self._get_top_indices(probs, at_k=100)
        predictions = self._inverse_movie_id_lookup(top_indices)
        prev_label = tf.reshape(x["context_movie_id"][:, 0], shape=(-1, 1))
        return tf.concat((prev_label, x["label_movie_id"], predictions), axis=1)
