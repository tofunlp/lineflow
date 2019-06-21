import os.path as osp
import pickle
from collections import Counter
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Sampler, BatchSampler

from tqdm import tqdm

import lineflow as lf
import lineflow.datasets as lfds


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
START_TOKEN = '<s>'
END_TOKEN = '</s>'

IGNORE_INDEX = -100


class SortedSampler(Sampler):
    def __init__(self, dataset, sort_key, sorting_size=None):
        self._num_samples = len(dataset)
        self._dataset = dataset
        self._sort_key = sort_key
        self._sorting_size = sorting_size or self._num_samples

    def __iter__(self):
        chunk = []
        for i, x in enumerate(self._dataset):
            chunk.append((i, self._sort_key(x)))
            if len(chunk) == self._sorting_size:
                chunk.sort(key=lambda x: x[1])
                yield from (i for i, _ in chunk)
                chunk = []
        if chunk:
            chunk.sort(key=lambda x: x[1])
            yield from (i for i, _ in chunk)

    def __len__(self):
        return self._num_samples


class RandomBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, pool_size=100):
        super().__init__(sampler, batch_size, drop_last)

        self.pool_size = pool_size

    def __iter__(self):
        bucket = []
        batch = []
        for index in self.sampler:
            batch.append(index)
            if len(batch) == self.batch_size:
                bucket.append(batch)
                batch = []
            if len(bucket) == self.pool_size:
                random.shuffle(bucket)
                yield from bucket
                bucket = []
        if len(bucket) > 0:
            yield from bucket
        if len(batch) > 0 and not self.drop_last:
            yield batch


def to_dict(x):
    return {'en': x[0], 'ja': x[1]}


@lf.apply('en')
@lf.apply('ja')
def preprocess(x):
    return [START_TOKEN] + x.split() + [END_TOKEN]


def build_vocab(tokens, cache='vocab.pkl', max_size=50000):
    if not osp.isfile(cache):
        counter = Counter(tokens)
        words, _ = zip(*counter.most_common(max_size))
        words = [PAD_TOKEN, UNK_TOKEN] + list(words)
        token_to_index = dict(zip(words, range(len(words))))
        if START_TOKEN not in token_to_index:
            token_to_index[START_TOKEN] = len(token_to_index)
            words += [START_TOKEN]
        if END_TOKEN not in token_to_index:
            token_to_index[END_TOKEN] = len(token_to_index)
            words += [END_TOKEN]
        with open(cache, 'wb') as f:
            pickle.dump((token_to_index, words), f)
    else:
        with open(cache, 'rb') as f:
            token_to_index, words = pickle.load(f)

    return token_to_index, words


def get_indexer(key, token_to_index, unk_index):
    @lf.apply(key)
    def indexing(x):
        return [token_to_index.get(token, unk_index) for token in x]
    return indexing


def get_collate_fn(pad_index):
    def f(batch):
        src, tgt = zip(*((x['en'], x['ja']) for x in batch))
        src_max_length = max(len(x) for x in src)
        tgt_max_length = max(len(y) for y in tgt)
        padded_src = [x + [pad_index] * (src_max_length - len(x)) for x in src]
        padded_tgt = [y + [IGNORE_INDEX] * (tgt_max_length - len(y)) for y in tgt]
        return torch.LongTensor(padded_src), torch.LongTensor(padded_tgt)
    return f


if __name__ == '__main__':
    print('Reading...')
    train = lfds.SmallParallelEnJa('train').map(to_dict)
    validation = lfds.SmallParallelEnJa('dev').map(to_dict)

    train = train.map(preprocess)
    validation = validation.map(preprocess)

    en_tokens = lf.flat_map(lambda x: x['en'],
                            train + validation,
                            lazy=True)
    ja_tokens = lf.flat_map(lambda x: x['ja'],
                            train + validation,
                            lazy=True)
    print('Building vocabulary...')
    en_token_to_index, _ = build_vocab(en_tokens, 'en.vocab')
    ja_token_to_index, _ = build_vocab(ja_tokens, 'ja.vocab')
    print(f'Vocab Size: {len(en_token_to_index)}')
    print(f'Vocab Size: {len(ja_token_to_index)}')

    en_unk_index = en_token_to_index[UNK_TOKEN]
    ja_unk_index = ja_token_to_index[UNK_TOKEN]

    train = train \
        .map(get_indexer('en', en_token_to_index, en_token_to_index)) \
        .map(get_indexer('ja', ja_token_to_index, ja_token_to_index))

    pad_index = en_token_to_index[PAD_TOKEN]

    batch_size = 64
    pool_size = 100

    loader = DataLoader(
        train,
        batch_sampler=RandomBatchSampler(
            SortedSampler(train, lambda x: - len(x['en']), batch_size * pool_size),
            batch_size, False, pool_size),
        num_workers=4,
        collate_fn=get_collate_fn(pad_index))

    for batch in tqdm(loader):
        ...
    del loader
