import tensorflow as tf
from tensorflow.python.ops.nn_impl import _compute_sampled_logits

from mos.train.config import Config


class SampledMixtureOfSoftmaxes(tf.keras.layers.Layer):
    def __init__(self, config: Config, movie_id_embedding: tf.keras.layers.Embedding, vocab_length: int):
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
        self._movie_id_biases = tf.zeros(shape=[vocab_length + 1], dtype=tf.float32)
        self._config = config
        self._vocab_length = vocab_length
        self._label_modalities_proba = [1 / self._vocab_length] * self._vocab_length
        self._test_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    def call(self, label, inputs, training):
        mos_projections = tf.tanh(tf.matmul(inputs, tf.transpose(self._mos_proj_mat)))
        mos_projections = tf.reshape(
            mos_projections, shape=(-1, self._config.mos_heads, self._config.embedding_dimension)
        )
        pi_values_logits = tf.matmul(inputs, tf.transpose(self._mos_mix_mat))
        pi_values = tf.nn.softmax(pi_values_logits)
        return self._compute_mos(label, mos_projections, pi_values, training)

    def _compute_mos(self, label, mos_projections, pi_values, training):
        sample_range = self._vocab_length + 1  # +1 to account for default embedding mapped at 0
        labels = tf.reshape(tf.cast(label, dtype=tf.int64), [-1, 1])
        num_neg_samples = self._vocab_length // 45
        sampled_values = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,  # list of target-ids (ground-truths) [batch_size x _nb_videos]
            num_true=1,  # ground-truth labels are vectors of len 1 (not a multi-class classification)
            num_sampled=num_neg_samples,  # how many samples to extract for loss computation -> impacts the exec time
            unique=True,  # do not sample the same index/video_id twice for the same batch
            range_max=sample_range,  # number of distinct classes = video vocab
            unigrams=self._label_modalities_proba,
            # list of video occurrences in dataset = sampling probabilites  # noqa:501
            distortion=0.4,  # how much to flatten the unigrams distribution (1.0=unchanged, 0.0=uniform sampling)
            num_reserved_ids=1,  # adds a sampling proba of 0.0 at index 0 to exclude default embedding
            seed=42,
        )

        probs = None
        sampled_labels = None
        for i in range(self._config.mos_heads):
            head_inputs = mos_projections[:, i, :]
            head_probs, sampled_labels = self._compute_head_probs(
                head_inputs, labels, num_neg_samples, sampled_values, training
            )

            if i == 0:
                probs = tf.expand_dims(pi_values[:, i], axis=-1) * head_probs
            else:
                probs += tf.expand_dims(pi_values[:, i], axis=-1) * head_probs

        if training:
            log_probs = tf.math.log(probs)
            loss_values = -tf.math.reduce_sum(sampled_labels * log_probs, axis=1)
            loss_value = tf.reduce_mean(loss_values)
            return probs, loss_value

        # When in test mode, we can directly the categorical cross entropy because we sample the logits
        return probs, self._test_loss(label, probs)

    def _compute_head_probs(self, head_inputs, labels, num_neg_samples, sampled_values, training):
        if training:
            sampled_logits, sampled_labels = _compute_sampled_logits(
                weights=self._movie_id_embedding.embeddings,
                biases=self._movie_id_biases,
                labels=labels,
                inputs=head_inputs,
                num_sampled=num_neg_samples,
                num_classes=self._vocab_length,
                sampled_values=sampled_values,
                subtract_log_q=True,
                seed=42,
            )
            sampled_labels = tf.stop_gradient(sampled_labels)
            head_probs = tf.nn.softmax(sampled_logits)
        else:
            head_logits = tf.matmul(head_inputs, tf.transpose(self._movie_id_embedding.embeddings))
            head_probs = tf.nn.softmax(head_logits)
            sampled_labels = None

        return head_probs, sampled_labels
