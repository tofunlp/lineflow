# LineFlow: Framework-Agnostic NLP Data Loader in Python
[![Build Status](https://img.shields.io/travis/tofunlp/lineflow/master.svg?logo=travis)](https://travis-ci.com/tofunlp/lineflow)
[![codecov](https://codecov.io/gh/tofunlp/lineflow/branch/master/graph/badge.svg)](https://codecov.io/gh/tofunlp/lineflow)

LineFlow is a simple text dataset loader for NLP deep learning tasks.

- LineFlow was designed to use in all deep learning frameworks.
- LineFlow enables you to build pipelines via functional APIs (`.map`, `.filter`, `.flat_map`).
- LineFlow provides common NLP datasets.

LineFlow is heavily inspired by [tensorflow.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) and [chainer.dataset](https://docs.chainer.org/en/stable/reference/datasets.html).

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

- Please check out the [examples](https://github.com/yasufumy/lineflow/tree/master/examples) to see how to use LineFlow, especially for tokenization, building vocabulary, and indexing.

Loads Penn Treebank:

```py
>>> import lineflow.datasets as lfds
>>> train = lfds.PennTreebank('train')
>>> train.first()
' aer banknote berlitz calloway centrust cluett fromstein gitano guterman hydro-quebec ipo kia memotec mlx nahb punts rake regatta rubens sim snack-food ssangyong swapo wachter '
```

Splits the sentence to the words:

```py
>>> # continuing from above
>>> train = train.map(str.split)
>>> train.first()
['aer', 'banknote', 'berlitz', 'calloway', 'centrust', 'cluett', 'fromstein', 'gitano', 'guterman', 'hydro-quebec', 'ipo', 'kia', 'memotec', 'mlx', 'nahb', 'punts', 'rake', 'regatta', 'rubens', 'sim', 'snack-food', 'ssangyong', 'swapo', 'wachter']
```

Obtains words in dataset:

```py
>>> # continuing from above
>>> words = train.flat_map(lambda x: x)
>>> words.take(5) # This is useful to build vocabulary.
['aer', 'banknote', 'berlitz', 'calloway', 'centrust']
```

Further more:

- [How to fine-tune BERT with pytorch-lightning](https://towardsdatascience.com/how-to-fine-tune-bert-with-pytorch-lightning-ba3ad2f928d2) by [@sobamchan](https://towardsdatascience.com/@sobamchan)

## Requirements

- Python3.6+

## Installation

To install LineFlow:

```sh
pip install lineflow
```

## Datasets

Is the dataset you want to use not supported? [Suggest a new dataset](https://github.com/tofunlp/lineflow/issues/new?template=dataset.md&title=Add+support+for+<dataset>) :tada:

- [Commonsense Reasoning](#commonsense-reasoning)
- [Language Modeling](#language-modeling)
- [Machine Translation](#machine-translation)
- [Paraphrase](#paraphrase)
- [Question Answering](#question-answering)
- [Sentiment Analysis](#sentiment-analysis)
- [Sequence Tagging](#sequence-tagging)
- [Text Summarization](#text-summarization)


### Commonsense Reasoning

#### [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa)

Loads the CommonsenseQA dataset:

```py
>>> import lineflow.datasets as lfds

>>> train = lfds.CommonsenseQA("train")
>>> dev = lfds.CommonsenseQA("dev")
>>> test = lfds.CommonsenseQA("test")
```

The items in this datset as follows:

```py
>>> import lineflow.datasets as lfds

>>> train = lfds.CommonsenseQA("train")
>>> train.first()
{"id": "075e483d21c29a511267ef62bedc0461",
 "answer_key": "A",
 "options": {"A": "ignore",
 "B": "enforce",
 "C": "authoritarian",
 "D": "yell at",
 "E": "avoid"},
 "stem": "The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?"}
}
```

### Language Modeling


#### [Penn Treebank](https://catalog.ldc.upenn.edu/docs/LDC95T7/cl93.html)

Loads the Penn Treebank dataset:

```py
import lineflow.datasets as lfds

train = lfds.PennTreebank('train')
dev = lfds.PennTreebank('dev')
test = lfds.PennTreebank('test')
```
#### [WikiText-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)

Loads the WikiText-103 dataset:

```py
import lineflow.datasets as lfds

train = lfds.WikiText103('train')
dev = lfds.WikiText103('dev')
test = lfds.WikiText103('test')
```

This dataset is preprossed, so you can tokenize each line with `str.split`:

```py
>>> import lineflow.datasets as lfds
>>> train = lfds.WikiText103('train').flat_map(lambda x: x.split() + ['<eos>'])
>>> train.take(5)
['<eos>', '=', 'Valkyria', 'Chronicles', 'III']
```

#### [WikiText-2](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) (Added by [@sobamchan](https://github.com/sobamchan), thanks.)

Loads the WikiText-2 dataset:

```py
import lineflow.datasets as lfds

train = lfds.WikiText2('train')
dev = lfds.WikiText2('dev')
test = lfds.WikiText2('test')
```

This dataset is preprossed, so you can tokenize each line with `str.split`:

```py
>>> import lineflow.datasets as lfds
>>> train = lfds.WikiText2('train').flat_map(lambda x: x.split() + ['<eos>'])
>>> train.take(5)
['<eos>', '=', 'Valkyria', 'Chronicles', 'III']
```

### Machine Translation

#### [small_parallel_enja](https://github.com/odashi/small_parallel_enja):

Loads the small_parallel_enja dataset which is small English-Japanese parallel corpus:

```py
import lineflow.datasets as lfds

train = lfds.SmallParallelEnJa('train')
dev = lfds.SmallParallelEnJa('dev')
test = lfd.SmallParallelEnJa('test')
```

This dataset is preprossed, so you can tokenize each line with `str.split`:

```py
>>> import lineflow.datasets as lfds
>>> train = lfds.SmallParallelEnJa('train').map(lambda x: (x[0].split(), x[1].split()))
>>> train.first()
(['i', 'can', "'t", 'tell', 'who', 'will', 'arrive', 'first', '.'], ['誰', 'が', '一番', 'に', '着', 'く', 'か', '私', 'に', 'は', '分か', 'り', 'ま', 'せ', 'ん', '。']
```

#### [WMT 14](http://www.statmt.org/wmt14/)

Loads the WMT14 dataset:

```py
import lineflow.datasets as lfds

train = lfds.Wmt14('train')
dev = lfds.Wmt14('dev')
test = lfd.Wmt14('test')
```

### Paraphrase

#### [Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398):

Loads the Miscrosoft Research Paraphrase Corpus:

```py
import lineflow.datasets as lfds

train = lfds.MsrParaphrase('train')
test = lfds.MsrParaphrase('test')
```

The item in this dataset as follows:

```py
>>> import lineflow.datasets as lfds
>>> train = lfds.MsrParaphrase('train')
>>> train.first()
{'quality': '1',
 'id1': '702876',
 'id2': '702977',
 'string1': 'Amrozi accused his brother, whom he called "the witness", of deliberately distorting his evidence.',
 'string2': 'Referring to him as only "the witness", Amrozi accused his brother of deliberately distorting his evidence.'
}
```

### Question Answering

[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/):

Loads the SQuAD dataset:

```py
import lineflow.datasets as lfds

train = lfds.Squad('train')
dev = lfds.Squad('dev')
```

The item in this dataset as follows:

```py
>>> import lineflow.datasets as lfds
>>> train = lfds.Squad('train')
>>> train.first()
{'answers': [{'answer_start': 515, 'text': 'Saint Bernadette Soubirous'}],
 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
 'id': '5733be284776f41900661182',
 'title': 'University_of_Notre_Dame',
 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'}
```

### Sentiment Analysis

#### [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/):

Loads the IMDB dataset:

```py
import lineflow.datasets as lfds

train = lfds.Imdb('train')
test = lfds.Imdb('test')
```

The item in this dataset as follows:

```py
>>> import lineflow.datasets as lfds
>>> train = lfds.Imdb('train')
>>> train.first()
('For a movie that gets no respect there sure are a lot of memorable quotes listed for this gem. Imagine a movie where Joe Piscopo is actually funny! Maureen Stapleton is a scene stealer. The Moroni character is an absolute scream. Watch for Alan "The Skipper" Hale jr. as a police Sgt.', 0)
```

### Sequence Tagging

#### [CoNLL2000](https://www.clips.uantwerpen.be/conll2000/chunking/)

Loads the CoNLL2000 dataset:

```py
import lineflow.datasets as lfds

train = lfds.Conll2000('train')
test = lfds.Conll2000('test')
```

### Text Summarization

#### [CNN / Daily Mail](https://github.com/harvardnlp/sent-summary):

Loads the CNN / Daily Mail dataset:

```py
import lineflow.datasets as lfds

train = lfds.CnnDailymail('train')
dev = lfds.CnnDailymail('dev')
test = lfds.CnnDailymail('test')
```

This dataset is preprossed, so you can tokenize each line with `str.split`:

```py
>>> import lineflow.datasets as lfds
>>> train = lfds.CnnDailymail('train').map(lambda x: (x[0].split(), x[1].split()))
>>> train.first()
... # the output is omitted because it's too long to display here.
```

#### [SciTLDR](https://github.com/allenai/scitldr)

Loads the TLDR dataset:

```py
import lineflow.datasets as lfds

train = lfds.SciTLDR('train')
dev = lfds.SciTLDR('dev')
test = lfds.SciTLDR('test')
```
