import tensorflow as tf


class VanillaSoftmax(tf.keras.layers.Layer):
    def __init__(self, movie_id_embedding: tf.keras.layers.Embedding):
        super().__init__()
        self._movie_id_embedding = movie_id_embedding

    def call(self, inputs):
        logits = tf.matmul(inputs, tf.transpose(self._movie_id_embedding.embeddings))
        return tf.nn.softmax(logits)
