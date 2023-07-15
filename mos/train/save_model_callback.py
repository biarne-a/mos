import logging

from tensorflow import keras
from google.cloud import storage

from mos.train.config import Config


class SaveModelCallback(keras.callbacks.Callback):
    def __init__(self, config: Config, model: keras.models.Model):
        super().__init__()
        self._config = config
        self._model = model

    def _upload_model(self, local_path: str, dest_bucket_name: str, gcs_path: str):
        logging.info(f"Uploading model to GCS")
        storage_client = storage.Client(project="concise-haven-277809")
        bucket = storage_client.bucket(dest_bucket_name)
        remote_path = f'{gcs_path}/{local_path}'
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_path)

    def on_epoch_end(self, epoch, logs=None):
        logging.info(f"Saving model at epoch {epoch}")
        print(f"Saving model at epoch {epoch}")
        save_name = f"{self._config.exp_name}_{epoch}.keras"
        self._model.save(save_name, include_optimizer=False)
        self._upload_model(local_path=save_name,
                           dest_bucket_name=self._config.bucket_name,
                           gcs_path=f"models/{self._config.exp_name}")
