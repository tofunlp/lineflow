import io
import json
import os
import pickle
from functools import lru_cache
from typing import Dict, List

import gdown

from lineflow import Dataset, download


def get_squad(version: int) -> Dict[str, List]:
    version_str = 'v1.1' if version == 1 else 'v2.0'

    train_url = f'https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-{version_str}.json'
    dev_url = f'https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-{version_str}.json'
    root = download.get_cache_directory(os.path.join('datasets', 'squad'))

    def creator(path):
        train_path = gdown.cached_download(train_url)
        dev_path = gdown.cached_download(dev_url)

        dataset = {}
        for split in ('train', 'dev'):
            data_path = train_path if split == 'train' else dev_path
            with io.open(data_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)['data']
            temp = []
            for x in data:
                title = x['title']
                for paragraph in x['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        qa['title'] = title
                        qa['context'] = context
                        temp.append(qa)
            dataset[split] = temp

        with io.open(path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset

    def loader(path):
        with io.open(path, 'rb') as f:
            return pickle.load(f)

    pkl_path = os.path.join(root, f'squad.{version_str}.pkl')
    return download.cache_or_load_file(pkl_path, creator, loader)


cached_get_squad = lru_cache()(get_squad)


class Squad(Dataset):

    def __init__(self,
                 split: str = 'train',
                 version: int = 1) -> None:
        if version != 1 and version != 2:
            raise ValueError(f"only 1 and 2 are valid for 'version', but {version} is given.")

        if split not in {'train', 'dev'}:
            raise ValueError(f"only 'train' and 'dev' are valid for 'split', but '{split}' is given.")

        raw = cached_get_squad(version)

        super().__init__(raw[split])
