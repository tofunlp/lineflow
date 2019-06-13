from lineflow.download import cached_download
from lineflow.text import TextDataset


TRAIN_URL = 'https://raw.githubusercontent.com/sobamchan/wikitext-2/master/train.txt'
VALID_URL = 'https://raw.githubusercontent.com/sobamchan/wikitext-2/master/valid.txt'
TEST_URL = 'https://raw.githubusercontent.com/sobamchan/wikitext-2/master/test.txt'


class WikiText(TextDataset):
    def __init__(self, split: str = 'train') -> None:
        if split == 'train':
            path = cached_download(TRAIN_URL)
        elif split == 'valid':
            path = cached_download(VALID_URL)
        elif split == 'test':
            path = cached_download(TEST_URL)
        else:
            raise ValueError(f"only 'train', 'valid', and 'test' are valid for 'split', but '{split}' is give.")
        super().__init__(path)
