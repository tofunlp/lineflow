import json
import pickle
import os.path as osp
from collections import Counter

import spacy
from lineflow import TextDataset
from lineflow.reducers import FlatMap, Concat


PREPROCESSED = ('train-v1.1.preprocessed', 'dev-v1.1.preprocessed')
POSTPROCESSED = ('train-v1.1.postprocessed', 'dev-v1.1.postprocessed')


def preprocess(nlp):

    def char_span_to_token_span(token_offsets, char_start, char_end):
        start_index = 0
        while start_index < len(token_offsets) and token_offsets[start_index][0] < char_start:
            start_index += 1
        if token_offsets[start_index][0] > char_start:
            start_index -= 1

        end_index = start_index
        while end_index < len(token_offsets) and token_offsets[end_index][1] < char_end:
            end_index += 1

        return (start_index, end_index)

    def f(x):
        x['question'] = [token.text for token in nlp(x['question']) if not token.is_space]
        x['context'], offsets = zip(*((token.text, (token.idx, token.idx + len(token.text)))
                                      for token in nlp(x['context']) if not token.is_space))

        answers, starts = zip(
            *((a['text'], a['answer_start']) for a in x['answers']))
        spans = [(start, start + len(text))
                 for text, start in zip(answers, starts)]
        most_frequent_span = Counter(spans).most_common(1)[0][0]
        target_index = max(range(len(spans)),
                           key=lambda x: spans[x] == most_frequent_span)
        x['answer'] = answers[target_index]

        start, end = char_span_to_token_span(offsets, *most_frequent_span)
        x['start'] = start
        x['end'] = end
        return x

    return f


def build_vocab(dataset, cache='vocab.pkl'):
    if not osp.isfile(cache):
        seen = {}

        def extract_tokens(x):
            tokens = x['question']
            context = x['context']
            if context not in seen:
                seen[context] = True
                tokens += context
            return tokens

        counter = Counter(FlatMap(extract_tokens)(dataset))
        words, _ = zip(*counter.most_common())
        token_to_index = dict(zip(words, range(len(words))))
        with open(cache, 'wb') as f:
            pickle.dump((token_to_index, words), f)
    else:
        with open(cache, 'rb') as f:
            token_to_index, words = pickle.load(f)

    return token_to_index, words


def postprocess_train(token_to_index):
    def f(x):
        question = [token_to_index[token] for token in x['question']]
        context = [token_to_index[token] for token in x['context']]
        return question, context, x['start'], x['end']
    return f


def postprocess_dev(token_to_index):
    def f(x):
        question = [token_to_index[token] for token in x['question']]
        context = [token_to_index[token] for token in x['context']]
        return question, context, x['context'], x['id']
    return f


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm',
                     disable=['vectors', 'textcat', 'tagger', 'ner'])
    # training data
    if not osp.exists(PREPROCESSED[0]):
        ds_train = TextDataset('./train-v1.1.jsonl').map(json.loads)
        ds_train = ds_train.map(preprocess(nlp)).save(PREPROCESSED[0])
    else:
        ds_train = TextDataset.load(PREPROCESSED[0])
    # dev data
    if not osp.exists(PREPROCESSED[1]):
        ds_dev = TextDataset('./dev-v1.1.jsonl').map(json.loads)
        ds_dev = ds_dev.map(preprocess(nlp)).save(PREPROCESSED[1])
    else:
        ds_dev = TextDataset.load(PREPROCESSED[1])

    # peek a first item
    print(ds_train.first())
    print(ds_dev.first())
    # support random access
    print(ds_train[100])
    print(ds_dev[100])

    token_to_index, words = build_vocab(Concat()(ds_train, ds_dev))

    # training data
    if not osp.exists(POSTPROCESSED[0]):
        ds_train = ds_train \
            .map(postprocess_train(token_to_index)) \
            .save(POSTPROCESSED[0])
    else:
        ds_train = ds_train.load(POSTPROCESSED[0])
    # dev data
    if not osp.exists(POSTPROCESSED[1]):
        ds_dev = ds_dev \
            .map(postprocess_dev(token_to_index)) \
            .save(POSTPROCESSED[1])
    else:
        ds_dev = ds_dev.load(POSTPROCESSED[1])

    print(ds_train.first())
    print(ds_dev.first())
    print(ds_train[100])
    print(ds_dev[100])
