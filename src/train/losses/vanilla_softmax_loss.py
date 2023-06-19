import tensorflow as tf


class VanillaSoftmaxLoss(tf.keras.losses.Loss):
    def __init__(self, movie_id_embeddings: tf.keras.layers.Embedding):
        super().__init__()
        self._movie_id_embeddings = movie_id_embeddings
        self._loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, y_true, y_pred):
        logits = tf.matmul(y_pred, tf.transpose(self._movie_id_embeddings.embeddings))
        return self._loss(y_true, logits)
