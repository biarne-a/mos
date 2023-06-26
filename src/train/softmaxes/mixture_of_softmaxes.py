import tensorflow as tf

from train.config import Config


class MixtureOfSoftmaxes(tf.keras.layers.Layer):
    def __init__(self, config: Config, movie_id_embedding: tf.keras.layers.Embedding):
        super().__init__()
        self._movie_id_embedding = movie_id_embedding
        initializer = tf.keras.initializers.GlorotNormal(seed=42)
        self._mos_proj_mat = tf.Variable(
            initial_value=initializer(
                shape=[config.mos_heads * config.embedding_dimension, config.embedding_dimension], dtype=tf.float32
            )
        )
        self._mos_mix_mat = tf.Variable(
            initial_value=initializer(shape=[config.mos_heads, config.embedding_dimension], dtype=tf.float32)
        )
        self._config = config

    def call(self, inputs):
        mos_projections = tf.tanh(tf.matmul(inputs, tf.transpose(self._mos_proj_mat)))
        mos_projections = tf.reshape(
            mos_projections, shape=(self._config.batch_size, self._config.mos_heads, self._config.embedding_dimension)
        )
        pi_values_logits = tf.matmul(inputs, tf.transpose(self._mos_mix_mat))
        pi_values = tf.nn.softmax(pi_values_logits)
        return self._compute_mos_low_mem(mos_projections, pi_values)

    def _compute_mos_high_mem(self, mos_projections, pi_values):
        """The fastest way to compute the mos but requires more memory"""
        head_logits = tf.matmul(mos_projections, tf.transpose(self._movie_id_embedding.embeddings))
        head_sm = tf.nn.softmax(head_logits)
        return tf.reduce_sum(tf.expand_dims(pi_values, axis=-1) * head_sm, axis=1)

    def _compute_mos_low_mem(self, mos_projections, pi_values):
        probs = None
        for i in range(self._config.mos_heads):
            head_logits = tf.matmul(mos_projections[:, i, :], tf.transpose(self._movie_id_embedding.embeddings))
            head_probs = tf.nn.softmax(head_logits)
            if i == 0:
                probs = tf.expand_dims(pi_values[:, i], axis=-1) * head_probs
            else:
                probs += tf.expand_dims(pi_values[:, i], axis=-1) * head_probs
        return probs
