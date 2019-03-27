import os.path as osp
import pickle
from collections import Counter

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

import lineflow as lf
import lineflow.datasets as lfds


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
START_TOKEN = '<s>'
END_TOKEN = '</s>'

IGNORE_INDEX = -100


def preprocess(x):
    source_string = x[0]
    target_string = x[1]
    return ([START_TOKEN] + source_string.split() + [END_TOKEN],
            [START_TOKEN] + target_string.split() + [END_TOKEN])


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


def postprocess(en_token_to_index,
                en_unk_index,
                ja_token_to_index,
                ja_unk_index):
    def f(x):
        source_ids = [en_token_to_index.get(token, en_unk_index) for token in x[0]]
        target_ids = [ja_token_to_index.get(token, ja_unk_index) for token in x[1]]
        return source_ids, target_ids
    return f


def get_collate_fn(pad_index):
    def f(batch):
        src, tgt = zip(*batch)
        src_max_length = max(len(x) for x in src)
        tgt_max_length = max(len(y) for y in tgt)
        padded_src = [x + [pad_index] * (src_max_length - len(x)) for x in src]
        padded_tgt = [y + [IGNORE_INDEX] * (tgt_max_length - len(y)) for y in tgt]
        return torch.LongTensor(padded_src), torch.LongTensor(padded_tgt)
    return f


if __name__ == '__main__':
    print('Reading...')
    train = lfds.SmallParallelEnJa('train')
    validation = lfds.SmallParallelEnJa('dev')

    train = train.map(preprocess)
    validation = validation.map(preprocess)

    en_tokens = lf.flat_map(lambda x: x[0],
                            train + validation,
                            lazy=True)
    ja_tokens = lf.flat_map(lambda x: x[1],
                            train + validation,
                            lazy=True)
    print('Building vocabulary...')
    en_token_to_index, _ = build_vocab(en_tokens, 'en.vocab')
    ja_token_to_index, _ = build_vocab(ja_tokens, 'ja.vocab')
    print(f'Vocab Size: {len(en_token_to_index)}')
    print(f'Vocab Size: {len(ja_token_to_index)}')

    pad_index = en_token_to_index[PAD_TOKEN]
    en_unk_index = en_token_to_index[UNK_TOKEN]
    ja_unk_index = ja_token_to_index[UNK_TOKEN]

    loader = DataLoader(
        train
        .map(postprocess(en_token_to_index, en_unk_index, ja_token_to_index, ja_unk_index))
        .save('enja.cache'),
        batch_size=32,
        num_workers=4,
        collate_fn=get_collate_fn(pad_index))

    for batch in tqdm(loader):
        ...
    del loader
