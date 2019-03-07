import os
import os.path as osp
import pickle
from collections import Counter

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

import lineflow as lf
from lineflow.datasets import Seq2SeqDataset


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
START_TOKEN = '<s>'
END_TOKEN = '</s>'

IGNORE_INDEX = -100

SOURCE_LENGTH_LIMIT = 400
TARGET_LENGTH_LIMIT = 100


def preprocess(x):
    source_string = x['source_string']
    target_string = x['target_string']
    return {'source_tokens': [START_TOKEN] + source_string.split()[:SOURCE_LENGTH_LIMIT] + [END_TOKEN],
            'target_tokens': [START_TOKEN] + target_string.split()[:TARGET_LENGTH_LIMIT] + [END_TOKEN]}


def build_vocab(tokens, cache='vocab.pkl', max_size=50000):
    if not osp.isfile(cache):
        counter = Counter(tokens)
        words, _ = zip(*counter.most_common(max_size))
        words = [PAD_TOKEN, UNK_TOKEN] + list(words)
        token_to_index = dict(zip(words, range(len(words))))
        with open(cache, 'wb') as f:
            pickle.dump((token_to_index, words), f)
    else:
        with open(cache, 'rb') as f:
            token_to_index, words = pickle.load(f)

    return token_to_index, words


def postprocess(token_to_index):
    def f(x):
        # 1 is unk_token index
        source_ids = [token_to_index.get(token, 1) for token in x['source_tokens']]
        target_ids = [token_to_index.get(token, 1) for token in x['target_tokens']]
        return source_ids, target_ids
    return f


def collate(batch):
    src, tgt = zip(*batch)
    src_max_length = max(len(x) for x in src)
    tgt_max_length = max(len(y) for y in tgt)
    # 0 is pad_token index
    padded_src = [x + [0] * (src_max_length - len(x)) for x in src]
    padded_tgt = [y + [IGNORE_INDEX] * (tgt_max_length - len(y)) for y in tgt]
    return torch.LongTensor(padded_src), torch.LongTensor(padded_tgt)


if __name__ == '__main__':

    if not osp.exists('./cnndm'):
        print('downloading...')
        os.system('curl -sOL https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz')
        os.system('mkdir cnndm')
        os.system('tar xf cnndm.tar.gz -C cnndm')

    print('reading...')
    train = Seq2SeqDataset(
        source_file_path='./cnndm/train.txt.src',
        target_file_path='./cnndm/train.txt.tgt.tagged').to_dict()
    validation = Seq2SeqDataset(
        source_file_path='./cnndm/val.txt.src',
        target_file_path='./cnndm/val.txt.tgt.tagged').to_dict()

    train = train.map(preprocess)
    validation = validation.map(preprocess)

    tokens = lf.flat_map(lambda x: x,
                         train.map(lambda x: x['source_tokens'] + x['target_tokens'])
                         + validation.map(lambda x: x['source_tokens'] + x['target_tokens']),
                         lazy=True)
    print('building vocabulary...')
    token_to_index, words = build_vocab(tokens)
    print(f'vocab size: {len(token_to_index)}')

    loader = DataLoader(
        train.map(postprocess(token_to_index)).save('cnndm.preprossed'),
        batch_size=32,
        num_workers=4,
        collate_fn=collate)

    for batch in tqdm(loader):
        ...
    del loader
