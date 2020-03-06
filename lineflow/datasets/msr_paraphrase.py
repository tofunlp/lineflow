import csv
import io
import os
import pickle
from functools import lru_cache
from typing import Dict, List

import gdown

from lineflow import Dataset, download


def get_msr_paraphrase() -> Dict[str, List[Dict[str, str]]]:

    url = 'https://raw.githubusercontent.com/wasiahmad/paraphrase_identification/master/dataset/msr-paraphrase-corpus/msr_paraphrase_{}.txt'  # NOQA
    root = download.get_cache_directory(os.path.join('datasets', 'msr_paraphrase'))

    def creator(path):
        dataset = {}
        fieldnames = ('quality', 'id1', 'id2', 'string1', 'string2')
        for split in ('train', 'test'):
            data_path = gdown.cached_download(url.format(split))
            with io.open(data_path, 'r', encoding='utf-8') as f:
                f.readline()  # skip header
                reader = csv.DictReader(f, delimiter='\t', fieldnames=fieldnames)
                dataset[split] = [dict(row) for row in reader]

        with io.open(path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset

    def loader(path):
        with io.open(path, 'rb') as f:
            return pickle.load(f)

    pkl_path = os.path.join(root, 'msr_paraphrase.pkl')
    return download.cache_or_load_file(pkl_path, creator, loader)


cached_get_msr_paraphrase = lru_cache()(get_msr_paraphrase)


class MsrParaphrase(Dataset):
    def __init__(self,
                 split: str = 'train') -> None:
        if split not in {'train', 'test'}:
            raise ValueError(f"only 'train', 'dev' and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_msr_paraphrase()
        super().__init__(raw[split])
