import json
import math

import spacy
from lineflow import TextDataset
from torch.utils.data import DataLoader
from chainer.iterators import MultiprocessIterator
from keras.utils import OrderedEnqueuer, Sequence


class TextSequence(Sequence):
    def __init__(self, dataset, batch_size):
        self._dataset = dataset
        self._batch_size = batch_size

    def __len__(self):
        return int(math.ceil(len(self._dataset)) / float(self._batch_size))

    def __getitem__(self, index):
        return [self._dataset[i]
                for i in range(index * self._batch_size, (index + 1) * self._batch_size)]


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm',
                     disable=['vectors', 'textcat', 'tagger', 'ner'])
    ds = TextDataset('dev-v1.1.jsonl').map(json.loads) \
        .map(lambda x: [token.text for token in nlp(x['question'])
                        if not token.is_space])

    # PyTorch
    print('PyTorch')
    loader = DataLoader(ds, batch_size=3, num_workers=4, shuffle=True)
    it = iter(loader)
    print(next(it))
    del it

    # Chainer
    print('Chainer')
    it = MultiprocessIterator(ds, batch_size=3, n_processes=4, shuffle=True)
    print(next(it))
    it.finalize()

    # Keras
    print('Keras')
    sequence = TextSequence(ds, batch_size=3)
    enqueuer = OrderedEnqueuer(sequence, use_multiprocessing=True, shuffle=True)
    enqueuer.start()
    it = enqueuer.get()
    print(next(it))
    enqueuer.stop()
