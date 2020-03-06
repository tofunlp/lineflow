import io
import os
import pickle
from functools import lru_cache
from typing import Dict, List

import gdown

from lineflow import Dataset, download


def get_penn_treebank() -> Dict[str, List[str]]:

    url = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.{}.txt'
    root = download.get_cache_directory(os.path.join('datasets', 'ptb'))

    def creator(path):
        dataset = {}
        for split in ('train', 'dev', 'test'):
            data_path = gdown.cached_download(url.format(split if split != 'dev' else 'valid'))
            with io.open(data_path, 'rt') as f:
                dataset[split] = [line.rstrip(os.linesep) for line in f]

        with io.open(path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset

    def loader(path):
        with io.open(path, 'rb') as f:
            return pickle.load(f)

    pkl_path = os.path.join(root, 'ptb.pkl')
    return download.cache_or_load_file(pkl_path, creator, loader)


cached_get_penn_treebank = lru_cache()(get_penn_treebank)


class PennTreebank(Dataset):
    def __init__(self, split: str = 'train') -> None:
        if split not in {'train', 'dev', 'test'}:
            raise ValueError(f"only 'train', 'dev' and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_penn_treebank()
        super(PennTreebank, self).__init__(raw[split])
