import json

import easyfile

from lineflow.download import cached_download
from lineflow.core import MapDataset


TRAIN_V1_URL = 'https://github.com/yasufumy/SQuAD_JSONL/blob/master/dataset/train-v1.1.jsonl?raw=true'
DEV_V1_URL = 'https://raw.githubusercontent.com/yasufumy/SQuAD_JSONL/master/dataset/dev-v1.1.jsonl'

TRAIN_V2_URL = 'https://github.com/yasufumy/SQuAD_JSONL/blob/master/dataset/train-v2.0.jsonl?raw=true'
DEV_V2_URL = 'https://raw.githubusercontent.com/yasufumy/SQuAD_JSONL/master/dataset/dev-v2.0.jsonl'


class Squad(MapDataset):
    def __init__(self,
                 split: str = 'train',
                 version: int = 1) -> None:
        if version == 1:
            train_url = TRAIN_V1_URL
            dev_url = DEV_V1_URL
        elif version == 2:
            train_url = TRAIN_V2_URL
            dev_url = DEV_V2_URL
        else:
            raise ValueError(f"only 1 and 2 are valid for 'version', but {version} is given.")

        if split == 'train':
            path = cached_download(train_url)
        elif split == 'dev':
            path = cached_download(dev_url)
        else:
            raise ValueError(f"only 'train' and 'dev' are valid for 'split', but '{split}' is given.")

        dataset = easyfile.TextFile(path)

        super().__init__(dataset, json.loads)
