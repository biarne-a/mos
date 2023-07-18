import pickle

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from mos.train.config import Config


def save_history(history: tf.keras.callbacks.History, config: Config):
    output_file = f"{config.data_dir}/results/history_{config.exp_name}.p"
    pickle.dump(history.history, tf.io.gfile.GFile(output_file, "wb"))


def save_predictions(config, data, model):
    nb_test_batches = data.nb_test // config.batch_size
    local_filename = f"{config.data_dir}/results/predictions_{config.exp_name}.csv"
    with tf.io.gfile.GFile(local_filename, "w") as fileh:
        columns = ["prev_label", "label"] + [f"output_{i}" for i in range(100)]
        header = ",".join(columns)
        fileh.write(f"{header}\n")
        i_batch = 0
        for batch in tqdm(data.test_ds.as_numpy_iterator(), total=nb_test_batches):
            predictions = model.predict_on_batch(batch)
            np.savetxt(fileh, predictions.astype(int), fmt="%i", delimiter=",")
            i_batch += 1
            if i_batch == nb_test_batches:
                break
            fileh.flush()
    return local_filename
