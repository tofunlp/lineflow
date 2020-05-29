import io
import json
import os
import pickle
import zipfile
from functools import lru_cache
from typing import Dict, List

import gdown

from lineflow import Dataset, download


def get_snli() -> Dict[str, List[str]]:

    url = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
    root = download.get_cache_directory(os.path.join('datasets', 'snli'))

    def creator(path):
        archive_path = gdown.cached_download(url)
        with zipfile.ZipFile(archive_path, 'r') as archive:
            dataset = {}
            path2key = {
                'snli_1.0/snli_1.0_train.jsonl': 'train',
                'snli_1.0/snli_1.0_dev.jsonl': 'dev',
                'snli_1.0/snli_1.0_test.jsonl': 'test',
            }
            for p, key in path2key.items():
                print(f'Extracting {p}...')
                with archive.open(p) as f:
                    lines = [json.loads(line.decode('utf-8')) for line in f]
                dataset[key] = lines

        with io.open(path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset

    def loader(path):
        with io.open(path, 'rb') as f:
            return pickle.load(f)

    pkl_path = os.path.join(root, 'snil.pkl')
    return download.cache_or_load_file(pkl_path, creator, loader)


cached_get_snli = lru_cache()(get_snli)


class Snli(Dataset):

    def __init__(self,
                 split: str = 'train') -> None:
        if split not in {'train', 'dev', 'test'}:
            raise ValueError(f"only 'train', 'dev', and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_snli()

        super().__init__(raw[split])
