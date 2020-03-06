import io
import os
import pickle
from functools import lru_cache
from typing import Dict, List, Tuple

import gdown

from lineflow import Dataset, download


def get_small_parallel_enja() -> Dict[str, Tuple[List[str]]]:

    en_url = 'https://raw.githubusercontent.com/odashi/small_parallel_enja/master/{}.en'
    ja_url = 'https://raw.githubusercontent.com/odashi/small_parallel_enja/master/{}.ja'
    root = download.get_cache_directory(os.path.join('datasets', 'small_parallel_enja'))

    def creator(path):
        dataset = {}
        for split in ('train', 'dev', 'test'):
            en_path = gdown.cached_download(en_url.format(split))
            ja_path = gdown.cached_download(ja_url.format(split))
            with io.open(en_path, 'rt') as en, io.open(ja_path, 'rt') as ja:
                dataset[split] = [(x.rstrip(os.linesep), y.rstrip(os.linesep))
                                  for x, y in zip(en, ja)]

        with io.open(path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset

    def loader(path):
        with io.open(path, 'rb') as f:
            return pickle.load(f)

    pkl_path = os.path.join(root, 'enja.pkl')
    return download.cache_or_load_file(pkl_path, creator, loader)


cached_get_small_parallel_enja = lru_cache()(get_small_parallel_enja)


class SmallParallelEnJa(Dataset):
    def __init__(self, split: str = 'train') -> None:
        if split not in {'train', 'dev', 'test'}:
            raise ValueError(f"only 'train', 'dev' and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_small_parallel_enja()
        super().__init__(raw[split])
