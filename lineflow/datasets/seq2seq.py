from lineflow.core import MapDataset
from lineflow.text import TextDataset


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
