import tensorflow as tf

from mos.train.config import Config


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
        self._loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    def call(self, label, inputs, training):
        mos_projections = tf.tanh(tf.matmul(inputs, tf.transpose(self._mos_proj_mat)))
        mos_projections = tf.reshape(
            mos_projections, shape=(-1, self._config.mos_heads, self._config.embedding_dimension)
        )
        pi_values_logits = tf.matmul(inputs, tf.transpose(self._mos_mix_mat))
        pi_values = tf.nn.softmax(pi_values_logits)
        return self._compute_mos(label, mos_projections, pi_values)

    def _compute_mos(self, label, mos_projections, pi_values):
        probs = None
        for i in range(self._config.mos_heads):
            head_logits = tf.matmul(mos_projections[:, i, :], tf.transpose(self._movie_id_embedding.embeddings))
            head_probs = tf.nn.softmax(head_logits)
            if i == 0:
                probs = tf.expand_dims(pi_values[:, i], axis=-1) * head_probs
            else:
                probs += tf.expand_dims(pi_values[:, i], axis=-1) * head_probs

        loss_value = self._loss(label, probs)
        return probs, loss_value
