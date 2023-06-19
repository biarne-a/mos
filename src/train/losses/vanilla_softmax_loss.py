import tensorflow as tf


class VanillaSoftmaxLoss(tf.keras.losses.Loss):
    def __init__(self, movie_id_embeddings: tf.keras.layers.Embedding):
        super().__init__()
        self._movie_id_embeddings = movie_id_embeddings
        self._loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, label, inputs):
        logits = tf.matmul(inputs, tf.transpose(self._movie_id_embeddings.embeddings))
        return self._loss(label, logits)
