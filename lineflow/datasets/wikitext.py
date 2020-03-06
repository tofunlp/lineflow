import io
import os
import pickle
import zipfile
from functools import lru_cache
from typing import Dict, List, Union

import arrayfiles
import gdown

from lineflow import download
from lineflow.core import Dataset


def get_wikitext(name: str) -> Dict[str, Union[arrayfiles.TextFile, List]]:

    url = f'https://s3.amazonaws.com/research.metamind.io/wikitext/{name}-v1.zip'
    root = download.get_cache_directory(os.path.join('datasets', 'wikitext'))

    def list_creator(path):
        archive_path = gdown.cached_download(url)
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

    def easyfile_creator(path):
        archive_path = gdown.cached_download(url)
        with zipfile.ZipFile(archive_path, 'r') as archive:
            print(f'Extracting to {root}...')
            archive.extractall(root)

        dataset = {}
        for split in ('train', 'dev', 'test'):
            filename = 'wiki.{}.tokens'.format(split if split != 'dev' else 'valid')
            dataset[split] = arrayfiles.TextFile(os.path.join(root, name, filename))

        with io.open(path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset

    def loader(path):
        with io.open(path, 'rb') as f:
            return pickle.load(f)

    assert name == 'wikitext-2' or name == 'wikitext-103'

    if name == 'wikitext-2':
        creator = list_creator
    elif name == 'wikitext-103':
        creator = easyfile_creator

    pkl_path = os.path.join(root, f'{name.replace("-", "")}.pkl')
    return download.cache_or_load_file(pkl_path, creator, loader)


cached_get_wikitext = lru_cache()(get_wikitext)


class WikiText2(Dataset):
    def __init__(self, split: str = 'train') -> None:
        if split not in {'train', 'dev', 'test'}:
            raise ValueError(f"only 'train', 'dev' and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_wikitext('wikitext-2')
        return super(WikiText2, self).__init__(raw[split])


class WikiText103(Dataset):
    def __init__(self, split: str = 'train') -> None:
        if split not in {'train', 'dev', 'test'}:
            raise ValueError(f"only 'train', 'dev' and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_wikitext('wikitext-103')
        return super(WikiText103, self).__init__(raw[split])
