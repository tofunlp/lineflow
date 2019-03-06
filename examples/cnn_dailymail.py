import os
import os.path as osp
import math

from allennlp.common.tqdm import Tqdm
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import WordTokenizer, Token
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator

import lineflow as lf


class CnnDailymailReader(DatasetReader):
    def __init__(self,
                 source_tokenizer=None,
                 target_tokenizer=None,
                 source_token_indexers=None,
                 source_length_limit=400,
                 target_length_limit=100,
                 source_field_name='source_tokens',
                 target_field_name='target_tokens'):
        super().__init__(lazy=False)

        self._source_tokenizer = source_tokenizer or WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._target_token_indexers = self._source_token_indexers
        self._source_length_limit = source_length_limit
        self._target_length_limit = target_length_limit
        self._source_field_name = source_field_name
        self._target_field_name = target_field_name

    def text_to_instance(self, x):
        source_string, target_string = x

        tokenized_source = self._source_tokenizer.tokenize(source_string)[:self._source_length_limit - 2]
        tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)

        tokenized_target = self._target_tokenizer.tokenize(target_string)[:self._target_length_limit - 2]
        tokenized_target.insert(0, Token(START_SYMBOL))
        tokenized_target.append(Token(END_SYMBOL))
        target_field = TextField(tokenized_target, self._target_token_indexers)

        return Instance({self._source_field_name: source_field,
                         self._target_field_name: target_field})

    def read(self, file_path):
        source_file_path = f'{file_path}.src'
        target_file_path = f'{file_path}.tgt.tagged'
        return lf.TextDataset([source_file_path, target_file_path]) \
            .map(self.text_to_instance)


if __name__ == '__main__':

    if not osp.exists('./cnndm'):
        print('downloading...')
        os.system('curl -sOL https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz')
        os.system('mkdir cnndm')
        os.system('tar xf cnndm.tar.gz -C cnndm')

    print('reading...')
    source_field_name = 'source_field_name'
    target_field_name = 'target_field_name'
    reader = CnnDailymailReader(
        source_length_limit=400,
        target_length_limit=100,
        source_field_name=source_field_name,
        target_field_name=target_field_name)
    train = reader.read('./cnndm/train.txt')
    validation = reader.read('./cnndm/val.txt')

    if not osp.exists('./vocabulary'):
        print('building vocabulary...')
        vocab = Vocabulary.from_instances(train + validation, max_vocab_size=50000)
        print(f'vocab size: {vocab.get_vocab_size()}')

        print('saving...')
        vocab.save_to_files('./vocabulary')
    else:
        print('loading vocabulary...')
        vocab = Vocabulary.from_files('./vocabulary')

    iterator = BucketIterator(sorting_keys=[(source_field_name, 'num_tokens')], batch_size=32)
    iterator.index_with(vocab)

    num_batches = math.ceil(len(train) / iterator._batch_size)

    for batch in Tqdm.tqdm(iterator(train, num_epochs=1), total=num_batches):
        ...
