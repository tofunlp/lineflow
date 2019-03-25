import json

from ..download import cached_download
from ..core import TextDataset, MapDataset


TRAIN_URL = 'https://github.com/yasufumy/SQuAD_JSONL/blob/master/dataset/train-v1.1.jsonl?raw=true'
DEV_URL = 'https://raw.githubusercontent.com/yasufumy/SQuAD_JSONL/master/dataset/dev-v1.1.jsonl'


class Squad(MapDataset):
    def __init__(self, split: str = 'train') -> None:
        if split == 'train':
            path = cached_download(TRAIN_URL)
        elif split == 'dev':
            path = cached_download(DEV_URL)
        else:
            raise ValueError(f"only 'train' and 'dev' are valid for 'split', but '{split}' is given.")

        dataset = TextDataset(path)

        super().__init__(dataset, json.loads)
