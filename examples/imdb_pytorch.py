import os.path as osp
import pickle
from collections import Counter
from functools import partial

import torch
from torch.utils.data import DataLoader

import spacy

from tqdm import tqdm

import lineflow.datasets as lfds


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
START_TOKEN = '<s>'
END_TOKEN = '</s>'

IGNORE_INDEX = -100

NLP = spacy.load('en_core_web_sm',
                 disable=['vectors', 'textcat', 'tagger', 'ner', 'parser'])


def preprocess(x):
    tokens = [token.text.lower() for token in NLP(x[0]) if not token.is_space]
    return ([START_TOKEN] + tokens + [END_TOKEN], x[1])


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


def postprocess(token_to_index,
                unk_index, x):
    token_index = [token_to_index.get(token, unk_index) for token in x[0]]
    return token_index, x[1]


def collate_fn(pad_index, batch):
    indices, labels = zip(*batch)
    max_length = max(len(x) for x in indices)
    padded = [x + [pad_index] * (max_length - len(x)) for x in indices]
    return torch.LongTensor(padded), torch.LongTensor(labels)


if __name__ == '__main__':
    print('Reading...')
    train = lfds.Imdb('train').map(preprocess)

    tokens = train.flat_map(lambda x: x[0])
    print('Building vocabulary...')
    token_to_index, _ = build_vocab(tokens, 'vocab.pkl')
    print(f'Vocab Size: {len(token_to_index)}')

    pad_index = token_to_index[PAD_TOKEN]
    unk_index = token_to_index[UNK_TOKEN]

    loader = DataLoader(
        train
        .map(partial(postprocess, token_to_index, unk_index))
        .save('imdb.train.cache'),
        batch_size=32,
        num_workers=4,
        collate_fn=partial(collate_fn, pad_index))

    for batch in tqdm(loader):
        ...
    del loader
