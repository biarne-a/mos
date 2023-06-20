import tensorflow as tf
from tensorflow import keras
from keras.engine import data_adapter

from train.config import Config
from train.datasets import Data, get_label_probs_hash_table
from train.losses import VanillaSoftmaxLoss, SampledSoftmaxLoss, InBatchNegSoftmaxLoss


class Gru4RecModel(keras.models.Model):
    def __init__(self, data: Data, config: Config):
        super().__init__()
        movie_id_vocab = list(data.movie_id_counts.keys())
        self._movie_id_lookup = tf.keras.layers.StringLookup(vocabulary=movie_id_vocab)
        self._inverse_movie_id_lookup = tf.keras.layers.StringLookup(vocabulary=movie_id_vocab, invert=True, oov_token='0')
        self._movie_id_embedding = tf.keras.layers.Embedding(len(data.movie_id_counts) + 1, config.embedding_dimension)
        self._gru_layer = tf.keras.layers.GRU(config.embedding_dimension)
        self._loss = self._get_loss(data, config)

    def _get_loss(self, data: Data, config: Config):
        if config.loss == "vanilla-sm":
            return VanillaSoftmaxLoss(self._movie_id_embedding)
        if config.loss == "in-batch-sm":
            label_probs = get_label_probs_hash_table(data, self._movie_id_lookup)
            return InBatchNegSoftmaxLoss(config, self._movie_id_embedding, label_probs)
        if config.loss == "sampled-sm":
            return SampledSoftmaxLoss(data, self._movie_id_embedding)
        raise Exception(f"Unknown loss {config.loss}")

    def call(self, inputs, training=False):
        ctx_movie_idx = self._movie_id_lookup(inputs["context_movie_id"])
        ctx_movie_emb = self._movie_id_embedding(ctx_movie_idx)
        return self._gru_layer(ctx_movie_emb)

    def train_step(self, inputs):
        # Forward pass
        with tf.GradientTape() as tape:
            hidden = self(inputs, training=True)
            label = self._movie_id_lookup(inputs["label_movie_id"])
            loss_val = self._loss(label, hidden)

        # Backward pass
        self.optimizer.minimize(loss_val, self.trainable_variables, tape=tape)

        return {"loss": loss_val}

    def test_step(self, inputs):
        # Forward pass
        hidden = self(inputs, training=False)
        label = self._movie_id_lookup(inputs["label_movie_id"])
        loss_val = self._loss(label, hidden)

        top_indices = self._get_top_indices(hidden, at_k=1000)
        # Compute metrics
        metric_results = self.compute_metrics(x=None, y=label, y_pred=top_indices, sample_weight=None)

        return {"loss": loss_val, **metric_results}

    def _get_top_indices(self, hidden, at_k):
        logits = tf.matmul(hidden, tf.transpose(self._movie_id_embedding.embeddings))
        return tf.math.top_k(logits, k=at_k).indices

    def predict_step(self, data):
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        hidden = self(x, training=False)
        top_indices = self._get_top_indices(hidden, at_k=100)
        predictions = self._inverse_movie_id_lookup(top_indices)
        return tf.concat((x["label_movie_id"], predictions), axis=1)
