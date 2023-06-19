import tensorflow as tf
from tensorflow import keras


class SampledSoftmaxLoss(keras.losses.Loss):
    """This class defines the Sampled Softmax Loss with fixed unigram sampler.
    The sampled loss allows for shorter computation time during training which is essential for scalability.
    Softmax Loss = Softmax layer (conversion of logits into probabilities) + Cross Entropy Loss.
    Sampling Strategy:
        - nb negative samples << nb of possible classes (i.e. num_neg_samples << _nb_videos)
        - the fixed_unigram_candidate_sampler chooses negative samples from the corpus of videos of the training set,
        and proportionaly to their respective frequencies (unigrams) in the dataset.
        - The selected set of negative samples stays the same for every observation/session of the same batch.
    Sampled Softmax Loss (summary of the computation):
        - From the embedding matrix W (weights [_nb_videos, embedding_size]), extract "num_sampled" rows including the
          true class (target video), which yields W' [num_sampled, embedding_size]
        - Apply "inputs" x W'.T, which yields the logits L' [batch_size, num_sampled]
        - Add the sampled biaises B', L"=L'+B'
        - Apply softmax to get probabilities from the logits
        - Compute the cross entropy loss of the extract.
    """

    def __init__(self, config, nb_videos, additional_data, video_id_weights, video_id_biases):
        super().__init__()
        self._config = config
        self._nb_videos = nb_videos
        self._additional_data = additional_data
        self._video_id_weights = video_id_weights
        self._video_id_biases = video_id_biases

    def call(self, inputs, label):
        """Computes the sampled softmax loss between the model outputs and the targets.
        Args:
            inputs: model outputs of one batch, i.e. the sessions' embeddings [batch_size, embedding_size=256 or 512]
            label: ground-truths of one batch, i.e. the id of the target video [batch_size, num_true(=1)]
                   /!/ labels are video_ids mapped between 1 and _nb_videos.
        Returns:
            tf.Tensor: sampled softmax loss of each batch items [batch_size, 1]
        """
        sample_range = self._nb_videos + 1  # +1 to account for default embedding mapped at 0
        label = tf.cast(label, dtype=tf.int64)
        num_neg_samples = int(float(self._nb_videos) / 45.0)
        sampled_values = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=tf.reshape(label, [-1, 1]),  # list of target-ids (ground-truths) [batch_size x _nb_videos]
            num_true=1,  # ground-truth labels are vectors of len 1 (not a multi-class classification)
            num_sampled=num_neg_samples,  # how many samples to extract for loss computation -> impacts the exec time
            unique=True,  # do not sample the same index/video_id twice for the same batch
            range_max=sample_range,  # number of distinct classes = video vocab
            unigrams=self._additional_data.label_modalities_proba,  # list of video occurrences in dataset = sampling probabilites  # noqa:501
            distortion=0.4,  # how much to flatten the unigrams distribution (1.0=unchanged, 0.0=uniform sampling)
            num_reserved_ids=1,  # adds a sampling proba of 0.0 at index 0 to exclude default embedding
            seed=self._config.seed,
        )

        return tf.nn.sampled_softmax_loss(
            weights=self._video_id_weights,  # current embeddings of all video corpus including default embedding (W)
            biases=self._video_id_biases,
            labels=tf.reshape(label, [-1, 1]),
            inputs=inputs,
            num_sampled=num_neg_samples,
            num_classes=self._nb_videos,
            sampled_values=sampled_values,
            seed=self._config.seed,
            remove_accidental_hits=False,  # TODO: Experiment with True
        )
