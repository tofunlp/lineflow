from typing import Dict, List, Tuple
import io
import os
import pickle
import tarfile

from lineflow.core import MapDataset
from lineflow import download


def get_imdb() -> Dict[str, List[str]]:

    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    root = download.get_cache_directory(os.path.join('datasets'))

    def creator(path):
        archive_path = download.cached_download(url)
        with tarfile.open(archive_path, 'r') as archive:
            print(f'Extracting to {root}...')
            archive.extractall(root)

        extracted_path = os.path.join(root, 'aclImdb')

        dataset = {}
        for split in ('train', 'test'):
            pos_path = os.path.join(extracted_path, split, 'pos')
            neg_path = os.path.join(extracted_path, split, 'neg')
            dataset[split] = [x.path for x in os.scandir(pos_path)
                              if x.is_file() and x.name.endswith('.txt')] + \
                             [x.path for x in os.scandir(neg_path)
                              if x.is_file() and x.name.endswith('.txt')]

        with io.open(path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset

    def loader(path):
        with io.open(path, 'rb') as f:
            return pickle.load(f)

    pkl_path = os.path.join(root, 'aclImdb', 'imdb.pkl')
    return download.cache_or_load_file(pkl_path, creator, loader)


class Imdb(MapDataset):
    def __init__(self, split: str = 'train') -> None:
        if split not in ('train', 'test'):
            raise ValueError(f"only 'train' and 'test' are valid for 'split', but '{split}' is given.")

        raw = get_imdb()

        def map_func(x: str) -> Tuple[str, int]:
            with io.open(x, 'rt', encoding='utf-8') as f:
                string = f.read()
            label = 0 if 'pos' in x else 1
            return (string, label)

        super().__init__(raw[split], map_func)
