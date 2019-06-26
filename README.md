# Lineflow: Framework-Agnostic NLP Data Loader in Python
[![Build Status](https://travis-ci.org/yasufumy/lineflow.svg?branch=master)](https://travis-ci.org/yasufumy/lineflow)
[![codecov](https://codecov.io/gh/yasufumy/lineflow/branch/master/graph/badge.svg)](https://codecov.io/gh/yasufumy/lineflow)

Lineflow is a simple text dataset loader for NLP deep learning tasks.

- Lineflow was designed to use in all deep learning frameworks.
- Lineflow enables you to build pipelines.
- Lineflow supports functional API and lazy evaluation.

Lineflow is heavily inspired by [tensorflow.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) and [chainer.dataset](https://docs.chainer.org/en/stable/reference/datasets.html).

## Installation

To install Lineflow:

```sh
pip install lineflow
```

## Basic Usage

lineflow.TextDataset expects line-oriented text files:

```py
import lineflow as lf


'''/path/to/text will be expected as follows:
i 'm a line 1 .
i 'm a line 2 .
i 'm a line 3 .
'''
ds = lf.TextDataset('/path/to/text')

ds.first()  # "i 'm a line 1 ."
ds.all() # ["i 'm a line 1 .", "i 'm a line 2 .", "i 'm a line 3 ."]
len(ds)  # 3
ds.map(lambda x: x.split()).first()  # ["i", "'m", "a", "line", "1", "."]
```

## Example

- Please check out the [examples/small\_parallel\_enja\_pytorch.py](https://github.com/yasufumy/lineflow/blob/master/examples/small_parallel_enja_pytorch.py) to see how to tokenize a sentence, build vocabulary, and do indexing.
- Also check out the other [examples](https://github.com/yasufumy/lineflow/tree/master/examples) to see how to use Lineflow.

Load the predefined dataset:

```py
>>> import lineflow.datasets as lfds
>>> train = lfds.SmallParallelEnJa('train')
>>> train.first()
("i can 't tell who will arrive first .", '誰 が 一番 に 着 く か 私 に は 分か り ま せ ん 。')
```

Split the sentence to the words:

```py
>>> # continuing from above
>>> train = train.map(lambda x: (x[0].split(), x[1].split()))
>>> train.first()
(['i', 'can', "'t", 'tell', 'who', 'will', 'arrive', 'first', '.'],
 ['誰', 'が', '一番', 'に', '着', 'く', 'か', '私', 'に', 'は', '分か', 'り', 'ま', 'せ', 'ん', '。'])
```

Obtain words in dataset:

```py
>>> # continuing from above
>>> import lineflow as lf
>>> en_tokens = lf.flat_map(lambda x: x[0], train)
>>> en_tokens[:5] # This is useful to build vocabulary.
['i', 'can', "'t", 'tell', 'who']
```

## Datasets

[CNN / Daily Mail](https://github.com/harvardnlp/sent-summary):

```py
import lineflow.datasets as lfds

train = lfds.CnnDailymail('train')
dev = lfds.CnnDailymail('dev')
test = lfds.CnnDailymail('test')
```

[IMDB](http://ai.stanford.edu/~amaas/data/sentiment/):

```py
import lineflow.datasets as lfds

train = lfds.Imdb('train')
test = lfds.Imdb('test')
```

[Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398):

```py
import lineflow.datasets as lfds

train = lfds.MsrParaphrase('train')
test = lfds.MsrParaphrase('test')
```

[small_parallel_enja](https://github.com/odashi/small_parallel_enja):

```py
import lineflow.datasets as lfds

train = lfds.SmallParallelEnJa('train')
dev = lfds.SmallParallelEnJa('dev')
test = lfd.SmallParallelEnJa('test')
```

[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/):

```py
import lineflow.datasets as lfds

train = lfds.Squad('train')
dev = lfds.Squad('dev')
```

[WikiText-2](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) (Added by [@sobamchan](https://github.com/sobamchan), thanks.)

```py
import lineflow.datasets as lfds

train = lfds.WikiText2('train')
dev = lfds.WikiText2('dev')
test = lfds.WikiText2('test')
```
