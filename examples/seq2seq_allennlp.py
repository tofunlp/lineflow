import os
import os.path as osp
import math

from allennlp.common.tqdm import Tqdm
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator


from lineflow.datasets import Seq2SeqDataset


if __name__ == '__main__':

    if not osp.exists('./cnndm'):
        print('Downloading...')
        os.system('curl -sOL https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz')
        os.system('mkdir cnndm')
        os.system('tar xf cnndm.tar.gz -C cnndm')

    print('Reading...')
    source_field_name = 'source_field_name'
    target_field_name = 'target_field_name'

    train = Seq2SeqDataset(
        source_file_path='./cnndm/train.txt.src',
        target_file_path='./cnndm/train.txt.tgt.tagged') \
        .to_allennlp(source_field_name=source_field_name,
                     target_field_name=target_field_name,
                     source_length_limit=400,
                     target_length_limit=100)
    validation = Seq2SeqDataset(
        source_file_path='./cnndm/val.txt.src',
        target_file_path='./cnndm/val.txt.tgt.tagged') \
        .to_allennlp(source_field_name=source_field_name,
                     target_field_name=target_field_name,
                     source_length_limit=400,
                     target_length_limit=100)

    if not osp.exists('./vocabulary'):
        print('Building vocabulary...')
        vocab = Vocabulary.from_instances(train + validation, max_vocab_size=50000)
        print(f'Vocab Size: {vocab.get_vocab_size()}')

        print('Saving...')
        vocab.save_to_files('./vocabulary')
    else:
        print('Loading vocabulary...')
        vocab = Vocabulary.from_files('./vocabulary')

    iterator = BucketIterator(sorting_keys=[(source_field_name, 'num_tokens')], batch_size=32)
    iterator.index_with(vocab)

    num_batches = math.ceil(len(train) / iterator._batch_size)

    for batch in Tqdm.tqdm(iterator(train, num_epochs=1), total=num_batches):
        ...
