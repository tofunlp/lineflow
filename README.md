# lineflow: Framework-Agnostic NLP Data Loader in Python
[![Build Status](https://travis-ci.org/yasufumy/lineflow.svg?branch=master)](https://travis-ci.org/yasufumy/lineflow)
[![codecov](https://codecov.io/gh/yasufumy/lineflow/branch/master/graph/badge.svg)](https://codecov.io/gh/yasufumy/lineflow)

lineflow is a simple text dataset loader for NLP deep learning tasks.

- lineflow was designed to use in all deep learning frameworks.
- lineflow enables you to build pipelines.
- lineflow supports functional API and lazy evaluation.

lineflow is heavily inspired by [tensorflow.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) and [chainer.dataset](https://docs.chainer.org/en/stable/reference/datasets.html).

## Installation

To install lineflow, simply:

```sh
pip install lineflow
```

If you'd like to use lineflow with [AllenNLP](https://allennlp.org/):

```sh
pip install "lineflow[allennlp]"
```

Also, if you'd like to use lineflow with [torchtext](https://torchtext.readthedocs.io/en/latest/):

```sh
pip install "lineflow[torchtext]"
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
```

## lineflow with PyTorch, torchtext, AllenNLP

- [PyTorch](#pytorch)
- [torchtext](#torchtext)
- [AllenNLP](#allennlp)

You can find more examples [here](https://github.com/yasufumy/lineflow/tree/master/examples).


### PyTorch

You can check full code [here](https://github.com/yasufumy/lineflow/blob/master/examples/small_parallel_enja_pytorch.py).

```py
...
import lineflow as lf
import lineflow.datasets as lfds

...


if __name__ == '__main__':
    train = lfds.SmallParallelEnJa('train')
    validation = lfds.SmallParallelEnJa('dev')

    train = train.map(preprocess)
    validation = validation.map(preprocess)

    en_tokens = lf.flat_map(lambda x: x[0],
                            train + validation,
                            lazy=True)
    ja_tokens = lf.flat_map(lambda x: x[1],
                            train + validation,
                            lazy=True)

    en_token_to_index, _ = build_vocab(en_tokens, 'en.vocab')
    ja_token_to_index, _ = build_vocab(ja_tokens, 'ja.vocab')

    ...

    loader = DataLoader(
        train
        .map(postprocess(en_token_to_index, en_unk_index, ja_token_to_index, ja_unk_index))
        .save('enja.cache'),
        batch_size=32,
        num_workers=4,
        collate_fn=get_collate_fn(pad_index))
```

### torchtext

You can check full code [here](https://github.com/yasufumy/lineflow/blob/master/examples/small_parallel_enja_torchtext.py).

```py
...
import lineflow.datasets as lfds


if __name__ == '__main__':
    src = data.Field(tokenize=str.split, init_token='<s>', eos_token='</s>')
    tgt = data.Field(tokenize=str.split, init_token='<s>', eos_token='</s>')
    fields = [('src', src), ('tgt', tgt)]
    train = lfds.SmallParallelEnJa('train').to_torchtext(fields)
    validation = lfds.SmallParallelEnJa('dev').to_torchtext(fields)

    src.build_vocab(train, validation)
    tgt.build_vocab(train, validation)

    iterator = data.BucketIterator(
        dataset=train, batch_size=32, sort_key=lambda x: len(x.src))
```

### AllenNLP

You can check full code [here](https://github.com/yasufumy/lineflow/blob/master/examples/small_parallel_enja_allennlp.py).

```py
...
import lineflow.datasets as lfds


if __name__ == '__main__':
    train = lfds.SmallParallelEnJa('train') \
        .to_allennlp(source_field_name=SOURCE_FIELD_NAME, target_field_name=TARGET_FIELD_NAME).all()
    validation = lfds.SmallParallelEnJa('dev') \
        .to_allennlp(source_field_name=SOURCE_FIELD_NAME, target_field_name=TARGET_FIELD_NAME).all()

    if not osp.exists('./enja_vocab'):
        vocab = Vocabulary.from_instances(train + validation, max_vocab_size=50000)
        vocab.save_to_files('./enja_vocab')
    else:
        vocab = Vocabulary.from_files('./enja_vocab')

    iterator = BucketIterator(sorting_keys=[(SOURCE_FIELD_NAME, 'num_tokens')], batch_size=32)
    iterator.index_with(vocab)
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
