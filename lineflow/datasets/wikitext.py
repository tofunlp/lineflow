from typing import Dict
import os
import io
import zipfile
from functools import lru_cache
import pickle

import easyfile

from lineflow.core import Dataset
from lineflow import download


def get_wikitext(name: str) -> Dict[str, easyfile.TextFile]:

    url = f'https://s3.amazonaws.com/research.metamind.io/wikitext/{name}-v1.zip'
    root = download.get_cache_directory(os.path.join('datasets', 'wikitext'))

    def creator(path):
        archive_path = download.cached_download(url)
        with zipfile.ZipFile(archive_path, 'r') as archive:
            dataset = {}
            path2key = {f'{name}/wiki.train.tokens': 'train',
                        f'{name}/wiki.valid.tokens': 'dev',
                        f'{name}/wiki.test.tokens': 'test'}
            for p, key in path2key.items():
                print(f'Extracting {p}...')
                with archive.open(p) as f:
                    lines = [line.decode('utf-8').rstrip(os.linesep) for line in f]
                dataset[key] = lines

        with io.open(path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset

    def loader(path):
        with io.open(path, 'rb') as f:
            return pickle.load(f)

    pkl_path = os.path.join(root, f'{name.replace("-", "")}.pkl')
    return download.cache_or_load_file(pkl_path, creator, loader)


cached_get_wikitext = lru_cache()(get_wikitext)


class WikiText2(Dataset):
    def __init__(self, split: str = 'train') -> None:
        if split not in ('train', 'dev', 'test'):
            raise ValueError(f"only 'train', 'dev' and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_wikitext('wikitext-2')
        return super(WikiText2, self).__init__(raw[split])


class WikiText103(Dataset):
    def __init__(self, split: str = 'train') -> None:
        if split not in ('train', 'dev', 'test'):
            raise ValueError(f"only 'train', 'dev' and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_wikitext('wikitext-103')
        return super(WikiText103, self).__init__(raw[split])
