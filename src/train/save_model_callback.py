import glob
import os
import logging

from tensorflow import keras
from google.cloud import storage

from train.config import Config


class SaveModelCallback(keras.callbacks.Callback):
    def __init__(self, config: Config, model: keras.models.Model):
        super().__init__()
        self._config = config
        self._model = model

    def _upload_from_directory(self, local_path: str, dest_bucket_name: str, gcs_path: str):
        logging.info(f"Uploading model to GCS")
        storage_client = storage.Client()
        rel_paths = glob.glob(local_path + '/**', recursive=True)
        bucket = storage_client.bucket(dest_bucket_name)
        for local_file in rel_paths:
            remote_path = f'{gcs_path}/{"/".join(local_file.split(os.sep)[1:])}'
            if os.path.isfile(local_file):
                blob = bucket.blob(remote_path)
                blob.upload_from_filename(local_file)

    def on_epoch_end(self, epoch, logs=None):
        logging.info(f"Saving model at epoch {epoch}")
        save_name = f"{self._config.exp_name}_{epoch}"
        self._model.save(save_name, include_optimizer=False)
        self._upload_from_directory(
            local_path=save_name, dest_bucket_name=self._config.bucket_name, gcs_path=f"models/{save_name}"
        )
