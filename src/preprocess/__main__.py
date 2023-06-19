from typing import Tuple, Union, List

import apache_beam as beam
import tensorflow as tf
from apache_beam.pvalue import PCollection


def _sort_views_by_timestamp(group) -> List[int]:
    views = group[0][1]
    views.sort(key=lambda x: x[0])
    return [v[0] for v in views]


def _convert_to_timelines(
    ratings: PCollection, data_desc: str, min_timeline_len: int = 3
) -> Union[PCollection, Tuple[PCollection, PCollection]]:
    """Convert ratings data to user."""
    return (
        ratings
        | f"{data_desc} - Set User Id Key" >> beam.Map(lambda x: (x["userId"], (x["movieId"], x["timestamp"])))
        | f"{data_desc} - Group By User Id" >> beam.GroupByKey()
        | f"{data_desc} - Add Views Counts" >> beam.Map(lambda x: (x, len(x[1])))
        | f"{data_desc} - Filter If Not Enough Views" >> beam.Filter(lambda x: x[1] > min_timeline_len)
        | f"{data_desc} - Sort Views By Timestamp" >> beam.Map(_sort_views_by_timestamp)
    )


def _generate_examples_from_single_timeline(timeline: List[int], max_context_len: int):
    """Generate TF examples from a single user timeline.

    Generate TF examples from a single user timeline. Timeline with length less
    than minimum timeline length will be skipped. And if context user history
    length is shorter than max_context_len, features will be padded with default
    values.

    Args:
      timeline: The timeline to generate TF examples from (A list of movieId).
      max_context_len: The maximum length of the context. If the context history
        length is less than max_context_length, features will be padded with
        default values.

    Returns:
      examples: Generated examples from this single timeline.
    """
    import tensorflow as tf
    examples = []
    for label_idx in range(1, len(timeline)):
        start_idx = max(0, label_idx - max_context_len)
        context_movie_id = timeline[start_idx:label_idx]
        # Pad context with out-of-vocab movie id 0.
        while len(context_movie_id) < max_context_len:
            context_movie_id.append(0)
        label_movie_id = timeline[label_idx]
        feature = {
            "context_movie_id": tf.train.Feature(int64_list=tf.train.Int64List(value=context_movie_id)),
            "label_movie_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_movie_id])),
        }
        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
        examples.append(tf_example.SerializeToString())

    return examples


def _count_movies_in_ratings(ratings: PCollection):
    return (
        ratings
        | "Set Movie Id Key" >> beam.Map(lambda x: (x["movieId"], x["userId"]))
        | "Count By Movie Id" >> beam.combiners.Count.PerKey()
    )


def _generate_examples(
    ratings: PCollection, min_timeline_len: int, max_context_len: int, data_desc: str
) -> PCollection:
    timelines = _convert_to_timelines(ratings, data_desc, min_timeline_len)
    examples_per_user = timelines | f"{data_desc} - Generate examples from timelines" >> beam.Map(_generate_examples_from_single_timeline, max_context_len=max_context_len)
    return examples_per_user | f"{data_desc} - Flatten examples" >> beam.FlatMap(lambda x: x)


def _save_in_tfrecords(data_dir: str, examples: PCollection, data_desc: str):
    import tensorflow as tf
    output_dir = f"{data_dir}/tfrecords/{data_desc}"
    if not tf.io.gfile.exists(output_dir):
        tf.io.gfile.makedirs(output_dir)
    prefix = f"{output_dir}/data"
    examples | f"Write {data_desc} examples" >> beam.io.tfrecordio.WriteToTFRecord(
        prefix,
        file_name_suffix=".tfrecord.gz",
    )


def _save_train_movie_counts(data_dir: str, counts: PCollection):
    counts | "Write train movie counts" >> beam.io.WriteToText(f"{data_dir}/vocab/train_movie_counts.txt", num_shards=1)


def _transform_to_rating(csv_row):
    cells = csv_row.split(",")
    return {
        "userId": int(cells[0]),
        "movieId": int(cells[1]),
        "rating": float(cells[2]),
        "timestamp": int(cells[3])
    }


def preprocess_with_dataflow(data_dir: str,
                             min_rating: float = 2.0,
                             min_timeline_len: int = 3,
                             max_context_len: int = 10):
    opts = beam.pipeline.PipelineOptions(
        experiments=["use_runner_v2"],
        project="concise-haven-277809",
        service_account="biarnes@concise-haven-277809.iam.gserviceaccount.com",
        staging_location="gs://ml-25m/beam/stg",
        temp_location="gs://ml-25m/beam/tmp",
        job_name="ml-25m-preprocess",
        num_workers=4,
        region="europe-west9",
        sdk_container_image="europe-west9-docker.pkg.dev/concise-haven-277809/biarnes/hsm-adasm",
    )
    with beam.Pipeline("DataflowRunner", options=opts) as pipeline:
        ratings = (
            pipeline
            | "Read ratings CSV" >> beam.io.textio.ReadFromText(f"{data_dir}/ratings.csv", skip_header_lines=1)
            | "Transform row to rating dict" >> beam.Map(_transform_to_rating)
            | "Filter low ratings" >> beam.Filter(lambda x: x["rating"] >= min_rating)
        )
        # We explicitly hard code the first test timestamp that corresponds to the last 5% of ordered the dataset
        # This could have been done programmatically but is not convenient with apache beam (so we take a shortcut)
        first_test_timestamp = 1499856234
        train_ratings = ratings | "Filter train" >> beam.Filter(lambda x: x["timestamp"] < first_test_timestamp)
        test_ratings = ratings | "Filter test" >> beam.Filter(lambda x: x["timestamp"] >= first_test_timestamp)

        train_movie_counts = _count_movies_in_ratings(train_ratings)
        train_examples = _generate_examples(train_ratings, min_timeline_len, max_context_len, data_desc="train")
        train_examples = train_examples | beam.Reshuffle()
        test_examples = _generate_examples(test_ratings, min_timeline_len, max_context_len, data_desc="test")

        # Save
        _save_in_tfrecords(data_dir, train_examples, data_desc="train")
        _save_in_tfrecords(data_dir, test_examples, data_desc="test")
        _save_train_movie_counts(data_dir, train_movie_counts)


if __name__ == "__main__":
    preprocess_with_dataflow(data_dir="gs://ml-25m")
