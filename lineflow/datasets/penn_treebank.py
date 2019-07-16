import easyfile

from lineflow import Dataset
from lineflow.download import cached_download


TRAIN_URL = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt'
DEV_URL = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt'
TEST_URL = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt'


class PennTreebank(Dataset):
    def __init__(self, split: str = 'train') -> None:
        if split == 'train':
            path = cached_download(TRAIN_URL)
        elif split == 'dev':
            path = cached_download(DEV_URL)
        elif split == 'test':
            path = cached_download(TEST_URL)
        else:
            raise ValueError(f"only 'train', 'dev', and 'test' are valid for 'split', but '{split} is given.'")

        dataset = easyfile.TextFile(path)
        super(PennTreebank, self).__init__(dataset)
