# lineflow: Framework-Agnostic NLP Data Pipeline in Python

## Installation

To install lineflow, simply:

```sh
$ pip install lineflow
```

## Usage

Load a text dataset and peek items:

```py
import lineflow as lf


ds = lf.TextDataset('/path/to/dataset')

print(ds.first())  # peek a first item
print(ds.take(5))  # peek a first 5 items
print(ds[100])  # random access

ds.map(tokenize)  # apply your own processing line by line (lazy evaluation)
```

Use lineflow with PyTorch:

```py
import lineflow as lf
from pytorch.utils.data import DataLoader


ds = lf.TextDataset('/path/to/dataset').map(tokenize)

loader = DataLoader(ds, batch_size=3, shuffle=True, num_workers=4)
it = iter(loader)
print(next(it))
del it
```

Use lineflow with Keras:

```py
import math

import lineflow as lf
from keras.utils import OrderedEnqueuer, Sequence


class TextSequence(Sequence):
    def __init__(self, dataset, batch_size):
        self._dataset = dataset
        self._batch_size = batch_size

    def __len__(self):
        return int(math.ceil(len(self._dataset)) / float(self._batch_size))

    def __getitem__(self, index):
        return [self._dataset[i]]
                for i in range(index * self._batch_size, (index + 1 ) * self._batch_size)


ds = lf.TextDataset('/path/to/dataset').map(tokenize)
sequence = TextSequence(ds, batch_size=3)
enqueuer = OrderedEnqueuer(sequence, shuffle=True, use_multiprocessing=True)
enqueuer.start()
it = enqueuer.get()
print(next(it))
enqueuer.stop()
```

Use lineflow with Chainer:

```py
import lineflow as lf
from chainer.iterators import MultiprocessIterator


ds = lf.TextSequence('/path/to/dataset').map(tokenize)
it = MultiprocessIterator(ds, batch_size=3, shuffle=True, n_processes=4)
print(next(it))
it.finalize()
```
