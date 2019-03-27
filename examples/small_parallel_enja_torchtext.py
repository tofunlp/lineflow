from torchtext import data
from tqdm import tqdm

import lineflow.datasets as lfds


if __name__ == '__main__':
    print('Reading...')
    src = data.Field(tokenize=str.split, init_token='<s>', eos_token='</s>')
    tgt = data.Field(tokenize=str.split, init_token='<s>', eos_token='</s>')
    fields = [('src', src), ('tgt', tgt)]
    train = lfds.SmallParallelEnJa('train').to_torchtext(fields)
    validation = lfds.SmallParallelEnJa('dev').to_torchtext(fields)

    print('Building vocabulary...')
    src.build_vocab(train, validation)
    tgt.build_vocab(train, validation)
    print(f'En Vocab Size: {len(src.vocab)}')
    print(f'Ja Vocab Size: {len(tgt.vocab)}')

    iterator = data.BucketIterator(
        dataset=train, batch_size=32, sort_key=lambda x: len(x.src))

    for batch in tqdm(iterator):
        ...
