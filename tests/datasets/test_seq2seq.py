from unittest import TestCase
import tempfile

from lineflow.datasets import Seq2SeqDataset


class Seq2SeqDatasetTestCase(TestCase):

    def setUp(self):
        # Dummy text file
        source_lines = ['source string 1', 'source string 2']
        target_lines = ['target string 1', 'target string 2']
        source_fp = tempfile.NamedTemporaryFile()
        for x in source_lines:
            source_fp.write(f'{x}\n'.encode('utf-8'))
        source_fp.seek(0)
        target_fp = tempfile.NamedTemporaryFile()
        for x in target_lines:
            target_fp.write(f'{x}\n'.encode('utf-8'))
        target_fp.seek(0)

        self.source_fp = source_fp
        self.source_lines = source_lines
        self.target_fp = target_fp
        self.target_lines = target_lines

    def tearDown(self):
        self.source_fp.close()
        self.target_fp.close()

    def test_dunder_init(self):
        ds = Seq2SeqDataset(self.source_fp.name, self.target_fp.name)
        for i, x in enumerate(ds):
            self.assertTupleEqual(x, (self.source_lines[i], self.target_lines[i]))

    def test_to_dict(self):
        ds = Seq2SeqDataset(self.source_fp.name, self.target_fp.name)
        source_field_name = 'source'
        target_field_name = 'target'
        ds_dict = ds.to_dict(source_field_name,
                             target_field_name)
        for i, x in enumerate(ds_dict):
            self.assertDictEqual(x, {source_field_name: self.source_lines[i],
                                     target_field_name: self.target_lines[i]})
