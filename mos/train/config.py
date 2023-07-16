import re
from typing import Dict


class Config:
    def __init__(self, args: Dict):
        self._args = args
        self.data_dir = args.get("data_dir")
        self.gcs_dir = args.get("gcs_dir", "")
        self.softmax_type = args.get("softmax_type")
        self.nb_epochs = args.get("nb_epochs")
        self.batch_size = args.get("batch_size")
        self.embedding_dimension = args.get("embedding_dimension")
        self.mos_heads = args.get("mos_heads")
        self.bucket_name = self._extract_bucket_name()

    def _extract_bucket_name(self):
        match = re.match("^gs://(.+)$", self.gcs_dir)
        if not match:
            raise Exception("gcs_dir must start with gs://")
        return match.groups()[0]

    @property
    def exp_name(self):
        if self.softmax_type == "mos":
            return f"{self.softmax_type}_{self.mos_heads}_{self.embedding_dimension}"
        return f"{self.softmax_type}_{self.embedding_dimension}"

    def to_json(self):
        return self._args
