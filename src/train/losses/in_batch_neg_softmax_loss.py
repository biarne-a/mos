import tensorflow as tf
from tensorflow import keras

from train.config import Config


class InBatchNegSoftmaxLoss(keras.losses.Loss):
    def __init__(self,
                 config: Config,
                 movie_id_embeddings: tf.keras.layers.Embedding,
                 label_probs: tf.lookup.StaticHashTable):
        super().__init__()
        self._in_batch_fake_labels = tf.range(0, config.batch_size)
        self._movie_id_embeddings = movie_id_embeddings
        self._label_probs = label_probs
        self._loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, label, inputs):
        logits = tf.matmul(inputs, tf.transpose(inputs))
        # Apply log q correction
        label_probs = self._label_probs.lookup(label)
        logits -= tf.math.log(label_probs)
        # Override labels to apply the softmax as if we only had "batch size" classes

        return self._loss(self._in_batch_fake_labels, logits)
