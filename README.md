# lineflow: Framework-Agnostic NLP Data Loader in Python
[![Build Status](https://travis-ci.org/yasufumy/lineflow.svg?branch=master)](https://travis-ci.org/yasufumy/lineflow)
[![codecov](https://codecov.io/gh/yasufumy/lineflow/branch/master/graph/badge.svg)](https://codecov.io/gh/yasufumy/lineflow)

lineflow is a simple text dataset loader for NLP deep learning tasks.

- lineflow was designed to use in all deep learning frameworks.
- lineflow enables you to build pipelines.
- lineflow supports functional API and lazy evaluation.

## Installation

To install lineflow, simply:

```sh
$ pip install lineflow
```

If you'd like to use lineflow with [AllenNLP](https://allennlp.org/):

```sh
$ pip install "lineflow[allennlp]"
```

Also, if you'd like to use lineflow with [torchtext](https://torchtext.readthedocs.io/en/latest/):

```sh
$ pip install "lineflow[torchtext]"
```

## Usage

lineflow.TextDataset expects line-oriented text files:

```py
import lineflow as lf


def preprocess(x):
    return x.split()

'''/path/to/text will look like below:
i 'm a line 1 .
i 'm a line 2 .
i 'm a line 3 .
'''
ds = lf.TextDataset('/path/to/text')
ds.first()  # "i 'm a line 1 ."
ds[1]  # "i 'm a line 2 ."

ds = ds.map(preprocess)
ds.first()  # ["i", "'m", "a", "line", "1", "."]

ds = lf.TextDataset(['/path/to/text', '/path/to/text'])
ds.first()  # ("i 'm a line 1 .", "i 'm a line 1 .")

ds = ds.map(lambda x: (x[0].split(), x[1].split()))
ds.first()  # (["i", "'m", "a", "line", "1", "."], ["i", "'m", "a", "line", "1", "."])
```

## lineflow with Deep Learning Frameworks

Use lineflow with AllenNLP:

```py
import math

from allennlp.common.tqdm import Tqdm
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator

from lineflow.datasets import Seq2SeqDataset


ds = Seq2SeqDataset(
    source_file_path='/path/to/source',
    target_file_path='/path/to/target'
).to_allennlp()

vocab = Vocabulary.from_instances(ds)

iterator = BucketIterator(sorting_keys=[('source_tokens', 'num_tokens')])
iterator.index_with(vocab)

num_batches = math.ceil(len(ds) / iterator._batch_size)

for batch in Tqdm.tqdm(iterator(train, num_epochs=1), total=num_batches):
    ...  # Your training code here
```

You can find other examples [here](https://github.com/yasufumy/lineflow/tree/master/examples).
