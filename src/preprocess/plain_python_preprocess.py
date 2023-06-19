#   Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Prepare TF.Examples for on-device recommendation model.

Following functions are included: 1) downloading raw data 2) processing to user
activity sequence and splitting to train/test data 3) convert to TF.Examples
and write in output location.

More information about the movielens dataset can be found here:
https://grouplens.org/datasets/movielens/
"""

import collections
import json
import os
import random

import pandas as pd
import tensorflow as tf
from absl import app, flags, logging

FLAGS = flags.FLAGS

# Permalinks to download movielens data.
RATINGS_FILE_NAME = "ratings.csv"
MOVIES_FILE_NAME = "movies.csv"
OUTPUT_TRAINING_DATA_FILENAME = "train_movielens_25m.tfrecord"
OUTPUT_TESTING_DATA_FILENAME = "test_movielens_25m.tfrecord"
OUTPUT_MOVIE_VOCAB_FILENAME = "movie_vocab.json"


def define_flags():
    """Define flags."""
    flags.DEFINE_string("data_dir", "/tmp", "Path to find movielens data.")
    flags.DEFINE_string("output_dir", None, "Path to the directory of output files.")
    flags.DEFINE_integer("min_timeline_length", 3, "The minimum timeline length to construct examples.")
    flags.DEFINE_integer("max_context_length", 10, "The maximum length of user context history.")
    flags.DEFINE_integer("min_rating", None, "Minimum rating of movie that will be used to in " "training data")
    flags.DEFINE_float("train_data_fraction", 0.9, "Fraction of training data.")


class MovieInfo(collections.namedtuple("MovieInfo", ["movie_id", "timestamp", "rating", "title", "genres"])):
    """Data holder of basic information of a movie."""

    __slots__ = ()

    def __new__(cls, movie_id=0, timestamp=0, rating=0.0, title="", genres=""):
        return super(MovieInfo, cls).__new__(cls, movie_id, timestamp, rating, title, genres)


def read_data(data_directory, min_rating=None):
    """Read movielens ratings.dat and movies.dat file into dataframe."""
    ratings_df = pd.read_csv(
        os.path.join(data_directory, RATINGS_FILE_NAME), sep=",", encoding="unicode_escape"
    )  # May contain unicode. Need to escape.
    ratings_df["timestamp"] = ratings_df["timestamp"].apply(int)
    if min_rating is not None:
        ratings_df = ratings_df[ratings_df["rating"] >= min_rating]
    movies_df = pd.read_csv(
        os.path.join(data_directory, MOVIES_FILE_NAME), sep=",", encoding="unicode_escape"
    )  # May contain unicode. Need to escape.
    return ratings_df, movies_df


def convert_to_timelines(ratings_df):
    """Convert ratings data to user."""
    timelines = collections.defaultdict(list)
    movie_counts = collections.Counter()
    for user_id, movie_id, rating, timestamp in ratings_df.values:
        timelines[user_id].append(MovieInfo(movie_id=movie_id, timestamp=int(timestamp), rating=rating))
        movie_counts[movie_id] += 1
    # Sort per-user timeline by timestamp
    for user_id, context in timelines.items():
        context.sort(key=lambda x: x.timestamp)
        timelines[user_id] = context
    return timelines, movie_counts


def _pad_or_truncate_movie_feature(feature, max_len, pad_value):
    feature.extend([pad_value for _ in range(max_len - len(feature))])
    return feature[:max_len]


def generate_examples_from_single_timeline(timeline, max_context_len=100):
    """Generate TF examples from a single user timeline.

    Generate TF examples from a single user timeline. Timeline with length less
    than minimum timeline length will be skipped. And if context user history
    length is shorter than max_context_len, features will be padded with default
    values.

    Args:
      timeline: The timeline to generate TF examples from.
      max_context_len: The maximum length of the context. If the context history
        length is less than max_context_length, features will be padded with
        default values.

    Returns:
      examples: Generated examples from this single timeline.
    """
    examples = []
    for label_idx in range(1, len(timeline)):
        start_idx = max(0, label_idx - max_context_len)
        context = timeline[start_idx:label_idx]
        # Pad context with out-of-vocab movie id 0.
        while len(context) < max_context_len:
            context.append(MovieInfo())
        label_movie_id = int(timeline[label_idx].movie_id)
        context_movie_id = [int(movie.movie_id) for movie in context]
        feature = {
            "context_movie_id": tf.train.Feature(int64_list=tf.train.Int64List(value=context_movie_id)),
            "label_movie_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_movie_id])),
        }
        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
        examples.append(tf_example)

    return examples


def generate_examples(
    ratings_df,
    min_timeline_len=3,
    max_context_len=100,
    random_seed=None,
    shuffle=True,
):
    """Generate tf examples.

    Create user timelines and convert them to tf examples by adding all possible context-label
    pairs in the examples pool.

    Args:
      ratings_df: The dataframe with the ratings.
      min_timeline_len: The minimum length of timeline. If the timeline length is
        less than min_timeline_len, empty examples list will be returned.
      max_context_len: The maximum length of the context. If the context history
        length is less than max_context_length, features will be padded with
        default values.
      random_seed: Seed for randomization.
      shuffle: Whether to shuffle the examples before splitting train and test
        data.

    Returns:
      train_examples: TF example list for training.
      test_examples: TF example list for testing.
    """
    logging.info("Generating movie rating user timelines.")
    timelines, movie_counts = convert_to_timelines(ratings_df)

    examples = []
    progress_bar = tf.keras.utils.Progbar(len(timelines))
    for timeline in timelines.values():
        if len(timeline) < min_timeline_len:
            progress_bar.add(1)
            continue
        single_timeline_examples = generate_examples_from_single_timeline(
            timeline=timeline,
            max_context_len=max_context_len,
        )
        examples.extend(single_timeline_examples)
        progress_bar.add(1)
        if len(examples) == 100_000:
            break

    # Split the examples into train, test sets.
    if shuffle:
        random.seed(random_seed)
        random.shuffle(examples)
    return examples, movie_counts


def write_tfrecords(tf_examples, filename):
    """Writes tf examples to tfrecord file, and returns the count."""
    with tf.io.TFRecordWriter(filename) as file_writer:
        length = len(tf_examples)
        progress_bar = tf.keras.utils.Progbar(length)
        for example in tf_examples:
            file_writer.write(example.SerializeToString())
            progress_bar.add(1)
        return length


def write_vocab_json(vocab, filename):
    """Write generated movie vocabulary to specified file."""
    with open(filename, "w", encoding="utf-8") as jsonfile:
        json.dump(vocab, jsonfile, indent=2)


def generate_datasets(
    extracted_data_dir,
    output_dir,
    min_timeline_length,
    max_context_length,
    min_rating=None,
    train_data_fraction=0.9,
    train_filename=OUTPUT_TRAINING_DATA_FILENAME,
    test_filename=OUTPUT_TESTING_DATA_FILENAME,
    vocab_filename=OUTPUT_MOVIE_VOCAB_FILENAME,
):
    """Generates train and test datasets as TFRecord, and returns stats."""
    logging.info("Reading data to dataframes.")
    ratings_df, movies_df = read_data(extracted_data_dir, min_rating=min_rating)

    logging.info("Split ratings according to timestamp")
    ratings_df = ratings_df.sort_values(by="timestamp")
    last_train_index = round(len(ratings_df) * train_data_fraction)
    train_ratings_df = ratings_df.iloc[:last_train_index]
    test_ratings_df = ratings_df.iloc[last_train_index:]

    logging.info("Generating train examples.")
    train_examples, train_movie_counts = generate_examples(
        ratings_df=train_ratings_df,
        min_timeline_len=min_timeline_length,
        max_context_len=max_context_length,
    )
    logging.info("Generating test examples.")
    test_examples, _ = generate_examples(
        ratings_df=test_ratings_df,
        min_timeline_len=min_timeline_length,
        max_context_len=max_context_length,
        shuffle=False,
    )

    if not tf.io.gfile.exists(output_dir):
        tf.io.gfile.makedirs(output_dir)
    logging.info("Writing generated training examples.")
    train_file = os.path.join(output_dir, train_filename)
    train_size = write_tfrecords(tf_examples=train_examples, filename=train_file)

    logging.info("Writing generated testing examples.")
    test_file = os.path.join(output_dir, test_filename)
    test_size = write_tfrecords(tf_examples=test_examples, filename=test_file)

    stats = {
        "train_size": train_size,
        "test_size": test_size,
        "train_file": train_file,
        "test_file": test_file,
    }

    movie_vocab = list(train_movie_counts.items())
    movie_vocab.sort(key=lambda x: x[1], reverse=True)
    vocab_file = os.path.join(output_dir, vocab_filename)
    write_vocab_json(movie_vocab, filename=vocab_file)
    stats.update(
        {
            "vocab_size": len(movie_vocab),
            "vocab_file": vocab_file,
            "vocab_max_id": max([m[0] for m in movie_vocab]),
        }
    )

    return stats


def main(_):
    stats = generate_datasets(
        extracted_data_dir=FLAGS.data_dir,
        output_dir=FLAGS.output_dir,
        min_timeline_length=FLAGS.min_timeline_length,
        max_context_length=FLAGS.max_context_length,
        min_rating=FLAGS.min_rating,
        train_data_fraction=FLAGS.train_data_fraction,
    )
    logging.info("Generated dataset: %s", stats)


if __name__ == "__main__":
    define_flags()
    app.run(main)
