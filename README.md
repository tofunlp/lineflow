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

- Please check out the [examples](https://github.com/yasufumy/lineflow/tree/master/examples) to see how to use Lineflow, especially for tokenization, building vocabulary, and indexing.

Load Penn Treebank:

```py
>>> import lineflow.datasets as lfds
>>> train = lfds.PennTreebank('train')
>>> train.first()
' aer banknote berlitz calloway centrust cluett fromstein gitano guterman hydro-quebec ipo kia memotec mlx nahb punts rake regatta rubens sim snack-food ssangyong swapo wachter '
```

Split the sentence to the words:

```py
>>> # continuing from above
>>> train = train.map(str.split)
>>> train.first()
['aer', 'banknote', 'berlitz', 'calloway', 'centrust', 'cluett', 'fromstein', 'gitano', 'guterman', 'hydro-quebec', 'ipo', 'kia', 'memotec', 'mlx', 'nahb', 'punts', 'rake', 'regatta', 'rubens', 'sim', 'snack-food', 'ssangyong', 'swapo', 'wachter']
```

Obtain words in dataset:

```py
>>> # continuing from above
>>> words = train.flat_map(lambda x: x)
>>> words.take(5) # This is useful to build vocabulary.
['aer', 'banknote', 'berlitz', 'calloway', 'centrust']
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

[Penn Treebank](https://catalog.ldc.upenn.edu/docs/LDC95T7/cl93.html)

```py
import lineflow.datasets as lfds

train = lfds.PennTreebank('train')
dev = lfds.PennTreebank('dev')
test = lfds.PennTreebank('test')
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
