from typing import Tuple, List
import tensorflow as tf
from tensorflow import keras

from train.config import Config
from train.datasets import Data, get_label_probs_hash_table
from train.losses import VanillaSoftmaxLoss, SampledSoftmaxLoss, InBatchNegSoftmaxLoss


class Gru4RecModel(keras.models.Model):
    def __init__(self, data: Data, config: Config):
        super().__init__()
        movie_id_vocab = list(data.movie_id_counts.keys())
        self._movie_id_lookup = tf.keras.layers.StringLookup(vocabulary=movie_id_vocab)
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
            logits = self(inputs, training=True)
            label = self._movie_id_lookup(inputs["label_movie_id"])
            loss_val = self._loss(label, logits)

        # Backward pass
        self.optimizer.minimize(loss_val, self.trainable_variables, tape=tape)

        return {"loss": loss_val}

    def test_step(self, inputs):
        # Forward pass
        outputs = self(inputs, training=False)
        logits = tf.matmul(outputs, tf.transpose(self._movie_id_embedding.embeddings))

        label_movie_idx = self._movie_id_lookup(inputs["label_movie_id"])
        loss_val = self._loss(label_movie_idx, logits)

        # Compute metrics
        # We add one to the output indices because everything is shifted because of the OOV token
        top_indices = tf.math.top_k(logits, k=1000).indices + 1
        metric_results = self.compute_metrics(x=None, y=label_movie_idx, y_pred=top_indices, sample_weight=None)

        return {"loss": loss_val, **metric_results}
