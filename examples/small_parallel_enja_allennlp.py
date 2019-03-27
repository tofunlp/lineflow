import os.path as osp

from allennlp.common.tqdm import Tqdm
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator

import lineflow.datasets as lfds


SOURCE_FIELD_NAME = 'source_tokens'
TARGET_FIELD_NAME = 'target_tokens'


if __name__ == '__main__':
    print('Reading...')
    train = lfds.SmallParallelEnJa('train') \
        .to_allennlp(source_field_name=SOURCE_FIELD_NAME, target_field_name=TARGET_FIELD_NAME).all()
    validation = lfds.SmallParallelEnJa('dev') \
        .to_allennlp(source_field_name=SOURCE_FIELD_NAME, target_field_name=TARGET_FIELD_NAME).all()

    if not osp.exists('./enja_vocab'):
        print('Building vocabulary...')
        vocab = Vocabulary.from_instances(train + validation, max_vocab_size=50000)
        print(f'Vocab Size: {vocab.get_vocab_size()}')

        print('Saving...')
        vocab.save_to_files('./enja_vocab')
    else:
        print('Loading vocabulary...')
        vocab = Vocabulary.from_files('./enja_vocab')

    iterator = BucketIterator(sorting_keys=[(SOURCE_FIELD_NAME, 'num_tokens')], batch_size=32)
    iterator.index_with(vocab)

    num_batches = iterator.get_num_batches(train)

    for batch in Tqdm.tqdm(iterator(train, num_epochs=1), total=num_batches):
        ...
