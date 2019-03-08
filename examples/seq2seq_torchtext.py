import os
import os.path as osp

from torchtext import data
from tqdm import tqdm

from lineflow.datasets import Seq2SeqDataset


if __name__ == '__main__':

    if not osp.exists('./cnndm'):
        print('Downloading...')
        os.system('curl -sOL https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz')
        os.system('mkdir cnndm')
        os.system('tar xf cnndm.tar.gz -C cnndm')

    print('Reading...')
    src = data.Field(tokenize=str.split, fix_length=400)
    tgt = data.Field(tokenize=str.split, fix_length=100)
    fields = [('src',  src), ('tgt', tgt)]

    train = Seq2SeqDataset(
        source_file_path='./cnndm/train.txt.src',
        target_file_path='./cnndm/train.txt.tgt.tagged') \
        .to_torchtext(fields)
    validation = Seq2SeqDataset(
        source_file_path='./cnndm/val.txt.src',
        target_file_path='./cnndm/val.txt.tgt.tagged') \
        .to_torchtext(fields)

    print('Building vocabulary...')
    src.build_vocab(train, validation, max_size=50000)
    tgt.vocab = src.vocab
    print(f'Vocab Size: {len(src.vocab)}')

    iterator = data.BucketIterator(
        dataset=train, batch_size=32, sort_key=lambda x: len(x.src))

    for batch in tqdm(iterator):
        ...
