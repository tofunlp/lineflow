from collections import Counter
from functools import partial

import torch
from torch.utils.data import DataLoader

import lineflow.datasets as lfds


PAD_TOKEN = '<pad>'


def build_vocab(tokens):
    counter = Counter(tokens)
    words, _ = zip(*counter.most_common())
    words = (PAD_TOKEN,) + words
    return dict(zip(words, range(len(words))))


def indexing(token_to_index, x):
    return [token_to_index[token] for token in x]


def collate(pad_index, batch):
    max_length = max(len(x) for x in batch)
    batch = [x + [pad_index] * (max_length - len(x)) for x in batch]
    base = torch.LongTensor(batch)
    return base[:, :-1], base[:, 1:]


if __name__ == '__main__':
    print('Reading...')
    train = lfds.PennTreebank('train').flat_map(lambda x: x.split() + ['</s>'])

    print('Building vocabulary...')
    token_to_index = build_vocab(train)

    bptt_len = 35
    train = train.window(bptt_len + 1).map(partial(indexing, token_to_index))

    loader = DataLoader(train,
                        batch_size=64,
                        shuffle=True,
                        num_workers=4,
                        collate_fn=partial(collate, token_to_index[PAD_TOKEN]))

    for src, tgt in loader:
        ...
