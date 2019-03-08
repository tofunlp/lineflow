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

SOURCE_FIELD = 'src'
TARGET_FIELD = 'tgt'


def preprocess(x):
    source_string = x[SOURCE_FIELD]
    target_string = x[TARGET_FIELD]
    return {SOURCE_FIELD: [START_TOKEN] + source_string.split()[:SOURCE_LENGTH_LIMIT] + [END_TOKEN],
            TARGET_FIELD: [START_TOKEN] + target_string.split()[:TARGET_LENGTH_LIMIT] + [END_TOKEN]}


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


def postprocess(token_to_index, unk_index):
    def f(x):
        source_ids = [token_to_index.get(token, unk_index) for token in x[SOURCE_FIELD]]
        target_ids = [token_to_index.get(token, unk_index) for token in x[TARGET_FIELD]]
        return source_ids, target_ids
    return f


def collate(pad_index):
    def fn(batch):
        src, tgt = zip(*batch)
        src_max_length = max(len(x) for x in src)
        tgt_max_length = max(len(y) for y in tgt)
        # 0 is pad_token index
        padded_src = [x + [pad_index] * (src_max_length - len(x)) for x in src]
        padded_tgt = [y + [IGNORE_INDEX] * (tgt_max_length - len(y)) for y in tgt]
        return torch.LongTensor(padded_src), torch.LongTensor(padded_tgt)
    return fn


if __name__ == '__main__':

    if not osp.exists('./cnndm'):
        print('Downloading...')
        os.system('curl -sOL https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz')
        os.system('mkdir cnndm')
        os.system('tar xf cnndm.tar.gz -C cnndm')

    print('Reading...')
    train = Seq2SeqDataset(
        source_file_path='./cnndm/train.txt.src',
        target_file_path='./cnndm/train.txt.tgt.tagged') \
        .to_dict(source_field_name=SOURCE_FIELD, target_field_name=TARGET_FIELD)
    validation = Seq2SeqDataset(
        source_file_path='./cnndm/val.txt.src',
        target_file_path='./cnndm/val.txt.tgt.tagged') \
        .to_dict(source_field_name=SOURCE_FIELD, target_field_name=TARGET_FIELD)

    train = train.map(preprocess)
    validation = validation.map(preprocess)

    tokens = lf.flat_map(lambda x: x[SOURCE_FIELD] + x[TARGET_FIELD],
                         train + validation,
                         lazy=True)
    print('Building vocabulary...')
    token_to_index, words = build_vocab(tokens)
    print(f'Vocab Size: {len(token_to_index)}')

    pad_index = token_to_index[PAD_TOKEN]
    unk_index = token_to_index[UNK_TOKEN]

    loader = DataLoader(
        train.map(postprocess(token_to_index, unk_index)).save('cnndm.preprossed'),
        batch_size=32,
        num_workers=4,
        collate_fn=collate(pad_index))

    for batch in tqdm(loader):
        ...
    del loader
