import tensorflow as tf
from tensorflow import keras

from datasets import Data


class SampledSoftmaxLoss(keras.losses.Loss):
    """This class defines the Sampled Softmax Loss with fixed unigram sampler.
    It allows to avoid scoring every possible movie from the embedding table.
    Instead, we sample negatives from the embedding table according to popularity.
    Softmax Loss = Softmax layer + Cross Entropy Loss.
    Sampling Strategy:-
        - the fixed_unigram_candidate_sampler chooses negative samples from the corpus of videos of the training set,
        and proportionally to their respective frequencies (unigrams) in the dataset.
        - The selected set of negative samples is the same for every observation of a given batch.
    Sampled Softmax Loss (summary of the computation):
        - From the embedding matrix W (weights [nb_movies, embedding_size]), extract "num_sampled" rows including the
          true class (target video), which yields W' [num_sampled, embedding_size]
        - Apply "inputs" x W'.T, which yields the logits L' [batch_size, num_sampled]
        - Add the sampled biases B', L"=L'+B'
        - Apply softmax to get probabilities from the logits
        - Compute the cross entropy loss of the extract.
    """

    def __init__(self, data: Data, movie_id_embeddings: tf.keras.layers.Embedding):
        super().__init__()
        self._nb_movies = len(data.movie_id_counts)
        self._label_modalities_counts = list(data.movie_id_counts.values())
        self._movie_id_embeddings = movie_id_embeddings
        self._movie_id_biases = tf.zeros(shape=[self._nb_movies + 1], name="proj_b", dtype=tf.float32)

    def call(self, label, inputs):
        """Computes the sampled softmax loss between the model outputs and the targets."""
        sample_range = self._nb_movies + 1  # +1 to account for default embedding mapped at 0
        label = tf.cast(label, dtype=tf.int64)
        num_neg_samples = int(float(self._nb_movies) / 45.0)
        sampled_values = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=tf.reshape(label, [-1, 1]),  # list of target-ids (ground-truths) [batch_size x _nb_videos]
            num_true=1,  # ground-truth labels are vectors of len 1 (not a multi-class classification)
            num_sampled=num_neg_samples,  # how many samples to extract for loss computation -> impacts the exec time
            unique=True,  # do not sample the same index/video_id twice for the same batch
            range_max=sample_range,  # number of distinct classes = video vocab
            unigrams=self._label_modalities_counts,  # list of video occurrences in dataset = sampling probabilites  # noqa:501
            distortion=0.4,  # how much to flatten the unigrams distribution (1.0=unchanged, 0.0=uniform sampling)
            num_reserved_ids=1,  # adds a sampling proba of 0.0 at index 0 to exclude default embedding
            seed=42,
        )

        return tf.nn.sampled_softmax_loss(
            weights=self._movie_id_embeddings.embeddings,  # current embeddings of all video corpus including default embedding (W)
            biases=self._movie_id_biases,
            labels=tf.reshape(label, [-1, 1]),
            inputs=inputs,
            num_sampled=num_neg_samples,
            num_classes=self._nb_movies,
            sampled_values=sampled_values,
            seed=42,
            remove_accidental_hits=False,
        )
