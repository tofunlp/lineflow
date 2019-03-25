import tarfile
from pathlib import Path
from itertools import chain

from ..download import cached_download
from ..download import get_cache_root
from ..core import MapDataset


IMDB_URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

TRAIN_DIR = 'aclImdb/train'
TEST_DIR = 'aclImdb/test'

ALL = (TRAIN_DIR, TEST_DIR)


class Imdb(MapDataset):
    def __init__(self, split: str = 'train') -> None:
        path = cached_download(IMDB_URL)
        tf = tarfile.open(path, 'r')
        cache_dir = Path(get_cache_root())
        if not all((cache_dir / p).exists() for p in ALL):
            print(f'Extracting from {path}...')
            tf.extractall(cache_dir)

        if split == 'train':
            pos_dir = f'{cache_dir / TRAIN_DIR}/pos'
            neg_dir = f'{cache_dir / TRAIN_DIR}/neg'
        elif split == 'test':
            pos_dir = f'{cache_dir / TEST_DIR}/pos'
            neg_dir = f'{cache_dir / TEST_DIR}/neg'
        else:
            raise ValueError(f"only 'train' and 'test' are valid for 'split', but '{split}' is given.")

        path = list(chain(Path(pos_dir).glob('*.txt'),
                          Path(neg_dir).glob('*.txt')))

        def map_func(x):
            string = x.read_text()
            label = 0 if 'pos' in str(x) else 1
            return (string, label)

        super().__init__(path, map_func)
