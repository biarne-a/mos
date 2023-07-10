import re
from typing import Dict


class Config:
    def __init__(self, args: Dict):
        self.data_dir = args.get("data_dir")
        self.softmax_type = args.get("softmax_type")
        self.nb_epochs = args.get("nb_epochs")
        self.batch_size = args.get("batch_size")
        self.embedding_dimension = args.get("embedding_dimension")
        self.mos_heads = args.get("mos_heads")
        self.bucket_name = self._extract_bucket_name(self.data_dir)

    @staticmethod
    def _extract_bucket_name(data_dir):
        match = re.match("^gs://(.+)$", data_dir)
        return match.groups()[0]

    @property
    def exp_name(self):
        return f"{self.softmax_type}{self.mos_heads}_{self.embedding_dimension}"
