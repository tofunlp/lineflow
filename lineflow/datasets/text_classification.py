from typing import Dict, List
import sys
import os
import io
import tarfile
import csv
from functools import lru_cache
import pickle

import gdown

from lineflow import Dataset
from lineflow import download


urls = {
    'ag_news': 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms',
    'sogou_news': 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE',
    'dbpedia': 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k',
    'yelp_review_polarity': 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg',
    'yelp_review_full': 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0',
    'yahoo_answers': 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU',
    'amazon_review_polarity': 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM',
    'amazon_review_full': 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA'
}


def get_text_classification_dataset(key) -> Dict[str, List[List[str]]]:

    url = urls[key]
    root = download.get_cache_directory(os.path.join('datasets', key))

    def creator(path):
        dataset = {}
        archive_path = gdown.cached_download(url)

        maxsize = sys.maxsize
        while True:
            try:
                csv.field_size_limit(maxsize)
                break
            except OverflowError:
                maxsize = int(maxsize / 10)
        csv.field_size_limit(maxsize)

        with tarfile.open(archive_path, 'r') as archive:
            for split in ('train', 'test'):
                filename = f'{key}_csv/{split}.csv'
                print(f'Processing {filename}...')
                reader = csv.reader(
                    io.TextIOWrapper(archive.extractfile(filename), encoding='utf-8'))
                dataset[split] = list(reader)

        with io.open(path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset

    def loader(path):
        with io.open(path, 'rb') as f:
            return pickle.load(f)

    pkl_path = os.path.join(root, f'{key}.pkl')
    return download.cache_or_load_file(pkl_path, creator, loader)


cached_get_text_classification_dataset = lru_cache()(get_text_classification_dataset)


class TextClassification(Dataset):
    def __init__(self, name: str, split: str = 'train') -> None:
        if name not in urls:
            raise ValueError()
        if split != 'train' and split != 'test':
            raise ValueError()

        raw = cached_get_text_classification_dataset(name)
        super(TextClassification, self).__init__(raw[split])
