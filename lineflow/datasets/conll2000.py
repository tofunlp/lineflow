import gzip
import io
import os
import pickle
from functools import lru_cache
from typing import Dict, List

import gdown

from lineflow import download
from lineflow.core import Dataset


def get_conll2000() -> Dict[str, List[str]]:

    url = 'https://www.clips.uantwerpen.be/conll2000/chunking/{}.txt.gz'
    root = download.get_cache_directory(os.path.join('datasets', 'conll2000'))

    def creator(path):
        dataset = {}
        for split in ('train', 'test'):
            data_path = gdown.cached_download(url.format(split))
            with gzip.open(data_path) as f:
                data = f.read().decode('utf-8').split('\n\n')

            dataset[split] = data

        with io.open(path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset

    def loader(path):
        with io.open(path, 'rb') as f:
            return pickle.load(f)

    pkl_path = os.path.join(root, 'conll2000.pkl')
    return download.cache_or_load_file(pkl_path, creator, loader)


cached_get_conll2000 = lru_cache()(get_conll2000)


class Conll2000(Dataset):
    def __init__(self, split: str = 'train') -> None:
        if split not in {'train', 'test'}:
            raise ValueError(f"only 'train' and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_conll2000()
        super(Conll2000, self).__init__(raw[split])
