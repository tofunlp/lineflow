import spacy

from torch.utils.data import DataLoader

from lineflow.core import CsvDataset
import lineflow as lf


NLP = spacy.load('en_core_web_sm')


def tokenize(string):
    return [token.text.lower() for token in NLP(string) if not token.is_space]


def preprocess(x):
    lines = ' '.join([x['InputSentence1'], x['InputSentence2'],
                      x['InputSentence3'], x['InputSentence4']])
    story = tokenize(lines)
    option1 = tokenize(x['RandomFifthSentenceQuiz1'])
    option2 = tokenize(x['RandomFifthSentenceQuiz2'])
    answer = int(x['AnswerRightEnding']) - 1
    return story, option1, option2, answer


def build_vocab(tokens):
    ...


def postprocess(x):
    ...


def collate(batch):
    ...


train = CsvDataset('./rocstories/cloze_test_val__spring2016 - cloze_test_ALL_val.csv',
                   header=True).map(preprocess)
dev = CsvDataset('./rocstories/cloze_test_test__spring2016 - cloze_test_ALL_test.csv',
                 header=True).map(preprocess)

vocab = build_vocab(lf.flat_map(lambda x: x[0] + x[1] + x[2],
                                train + dev, lazy=True))

processed = train.map(postprocess).save('train')

loader = DataLoader(processed, batch_size=32, collate_fn=collate)

for batch in loader:
    ...
