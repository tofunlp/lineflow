import csv
import io
import os
import pickle
import sys
import tarfile
from functools import lru_cache
from typing import Dict, List, Union

import arrayfiles
import gdown

from lineflow import Dataset, download

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


def get_text_classification_dataset(key) -> Dict[str, Union[List, arrayfiles.CsvFile]]:

    url = urls[key]
    root = download.get_cache_directory(os.path.join('datasets', 'text_classification', key))

    def list_creator(path):
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

    def easyfile_creator(path):
        dataset = {}
        archive_path = gdown.cached_download(url)

        with tarfile.open(archive_path, 'r') as archive:
            print(f'Extracting to {root}...')
            archive.extractall(root)

        dataset = {}
        for split in ('train', 'test'):
            filename = f'{key}_csv/{split}.csv'
            dataset[split] = arrayfiles.CsvFile(os.path.join(root, filename))

        with io.open(path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset

    def loader(path):
        with io.open(path, 'rb') as f:
            return pickle.load(f)

    assert key in urls

    if key in ('ag_news', 'dpbedia'):
        creator = list_creator
    else:
        creator = easyfile_creator

    pkl_path = os.path.join(root, f'{key}.pkl')
    return download.cache_or_load_file(pkl_path, creator, loader)


cached_get_text_classification_dataset = lru_cache()(get_text_classification_dataset)


class AgNews(Dataset):
    def __init__(self, split: str = 'train') -> None:
        if split not in {'train', 'test'}:
            raise ValueError(f"only 'train' and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_text_classification_dataset('ag_news')
        super(AgNews, self).__init__(raw[split])


class SogouNews(Dataset):
    def __init__(self, split: str = 'train') -> None:
        if split not in {'train', 'test'}:
            raise ValueError(f"only 'train' and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_text_classification_dataset('sogou_news')
        super(SogouNews, self).__init__(raw[split])


class Dbpedia(Dataset):
    def __init__(self, split: str = 'train') -> None:
        if split not in {'train', 'test'}:
            raise ValueError(f"only 'train' and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_text_classification_dataset('dbpedia')
        super(Dbpedia, self).__init__(raw[split])


class YelpReviewPolarity(Dataset):
    def __init__(self, split: str = 'train') -> None:
        if split not in {'train', 'test'}:
            raise ValueError(f"only 'train' and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_text_classification_dataset('yelp_review_polarity')
        super(YelpReviewPolarity, self).__init__(raw[split])


class YelpReviewFull(Dataset):
    def __init__(self, split: str = 'train') -> None:
        if split not in {'train', 'test'}:
            raise ValueError(f"only 'train' and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_text_classification_dataset('yelp_review_full')
        super(YelpReviewFull, self).__init__(raw[split])


class YahooAnswers(Dataset):
    def __init__(self, split: str = 'train') -> None:
        if split not in {'train', 'test'}:
            raise ValueError(f"only 'train' and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_text_classification_dataset('yahoo_answers')
        super(YahooAnswers, self).__init__(raw[split])


class AmazonReviewPolarity(Dataset):
    def __init__(self, split: str = 'train') -> None:
        if split not in {'train', 'test'}:
            raise ValueError(f"only 'train' and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_text_classification_dataset('amazon_review_polarity')
        super(AmazonReviewPolarity, self).__init__(raw[split])


class AmazonReviewFull(Dataset):
    def __init__(self, split: str = 'train') -> None:
        if split not in {'train', 'test'}:
            raise ValueError(f"only 'train' and 'test' are valid for 'split', but '{split}' is given.")

        raw = cached_get_text_classification_dataset('amazon_review_full')
        super(AmazonReviewFull, self).__init__(raw[split])
