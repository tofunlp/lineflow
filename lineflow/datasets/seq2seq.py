from typing import List, Tuple, Optional, Callable, TYPE_CHECKING

from ..core import MapDataset, TextDataset

if TYPE_CHECKING:
    from allennlp.data import Tokenizer, TokenIndexer  # NOQA
    from torchtext import data  # NOQA


class Seq2SeqDataset(TextDataset):
    def __init__(self,
                 source_file_path: str,
                 target_file_path: str) -> None:
        super().__init__([source_file_path, target_file_path])

    def to_dict(self,
                source_field_name: str = 'source_string',
                target_field_name: str = 'target_string') -> MapDataset:
        return MapDataset(
            self, lambda x: {source_field_name: x[0], target_field_name: x[1]})

    def to_allennlp(self,
                    source_tokenizer: 'Tokenizer' = None,
                    target_tokenizer: 'Tokenizer' = None,
                    source_token_indexers: 'TokenIndexer' = None,
                    target_token_indexers: 'TokenIndexer' = None,
                    source_field_name: str = 'source_tokens',
                    target_field_name: str = 'target_tokens',
                    source_length_limit: Optional[int] = None,
                    target_length_limit: Optional[int] = None) -> MapDataset:
        try:
            from allennlp.common.util import START_SYMBOL, END_SYMBOL
            from allennlp.data import Instance
            from allennlp.data.fields import TextField
            from allennlp.data.tokenizers import WordTokenizer, Token
            from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
            from allennlp.data.token_indexers import SingleIdTokenIndexer
        except ImportError as e:
            message = 'lineflow AllenNLP extension is not enabled.\n\n' \
                'Please install requirements as follows:\n\n' \
                '   pip install -U "lineflow[allennlp]"'
            raise ImportError(f'{e}\n\n{message}')

        source_tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
        target_tokenizer = target_tokenizer or source_tokenizer
        source_token_indexers = source_token_indexers or {'tokens': SingleIdTokenIndexer()}
        target_token_indexers = target_token_indexers or source_token_indexers

        def text_to_instance(x: Tuple[str]) -> Instance:
            source_string = x[0]
            target_string = x[1]

            tokenized_source = source_tokenizer.tokenize(source_string)[:source_length_limit]
            tokenized_source.insert(0, Token(START_SYMBOL))
            tokenized_source.append(Token(END_SYMBOL))
            source_field = TextField(tokenized_source, source_token_indexers)

            tokenized_target = target_tokenizer.tokenize(target_string)[:target_length_limit]
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, target_token_indexers)

            return Instance({source_field_name: source_field,
                             target_field_name: target_field})

        return MapDataset(self, text_to_instance)

    def to_torchtext(self,
                     fields: List[Tuple[str, 'data.Field']],
                     filter_pred: Optional[Callable[['data.Example'], bool]] = None) -> 'data.Dataset':
        try:
            from torchtext.data import Example, Dataset
        except ImportError as e:
            message = 'lineflow torchtext extension is not enabled.\n\n' \
                'Please install requirements as follows:\n\n' \
                '   pip install -U "lineflow[torchtext]"'
            raise ImportError(f'{e}\n\n{message}')

        def text_to_example(x: Tuple[str]) -> Example:
            return Example.fromlist(data=x, fields=fields)

        return Dataset(examples=MapDataset(self, text_to_example),
                       fields=fields,
                       filter_pred=filter_pred)
