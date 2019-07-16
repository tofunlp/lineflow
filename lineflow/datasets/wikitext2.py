import easyfile

from lineflow.download import cached_download
from lineflow import Dataset


TRAIN_URL = 'https://raw.githubusercontent.com/sobamchan/wikitext-2/master/train.txt'
DEV_URL = 'https://raw.githubusercontent.com/sobamchan/wikitext-2/master/valid.txt'
TEST_URL = 'https://raw.githubusercontent.com/sobamchan/wikitext-2/master/test.txt'


class WikiText2(Dataset):
    def __init__(self, split: str = 'train') -> None:
        if split == 'train':
            path = cached_download(TRAIN_URL)
        elif split == 'dev':
            path = cached_download(DEV_URL)
        elif split == 'test':
            path = cached_download(TEST_URL)
        else:
            raise ValueError(f"only 'train', 'dev', and 'test' are valid for 'split', but '{split}' is given.")

        dataset = easyfile.TextFile(path)
        super().__init__(dataset)
