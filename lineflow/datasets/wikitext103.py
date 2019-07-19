import zipfile
import os.path

from lineflow.download import cached_download
from lineflow.download import get_cache_root
from lineflow import TextDataset


WIKITEXT103_URL = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'

TRAIN_PATH = os.path.join(get_cache_root(), 'wikitext-103/wiki.train.tokens')
DEV_PATH = os.path.join(get_cache_root(), 'wikitext-103/wiki.valid.tokens')
TEST_PATH = os.path.join(get_cache_root(), 'wikitext-103/wiki.test.tokens')

ALL = (TRAIN_PATH, DEV_PATH, TEST_PATH)


class WikiText103(TextDataset):
    def __init__(self, split: str = 'train') -> None:
        zpath = cached_download(WIKITEXT103_URL)
        zf = zipfile.ZipFile(zpath, 'r')
        if not all(os.path.exists(p) for p in ALL):
            print(f'Extracting from {zpath}...')
            cache_dir = get_cache_root()
            zf.extractall(cache_dir)

        if split == 'train':
            path = TRAIN_PATH
        elif split == 'dev':
            path = DEV_PATH
        elif split == 'test':
            path = TEST_PATH
        else:
            raise ValueError(f"only 'train', 'dev' and 'test' are valid for 'split', but '{split}' is given.")

        super(WikiText103, self).__init__(path)
