from unittest import TestCase
from unittest.mock import patch
import tempfile

from allennlp.data.instance import Instance

from lineflow.datasets import CnnDailymailDataset


class CnnDailymailDatasetTestCase(TestCase):

    def setUp(self):
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

        lines_dict = {source_fp.name: source_lines, target_fp.name: target_lines}
        linecache_getline_patcher = patch('lineflow.core.linecache.getline')
        linecache_getline_mock = linecache_getline_patcher.start()
        linecache_getline_mock.side_effect = lambda filename, i: lines_dict[filename][i - 1]

        self.linecache_getline_patcher = linecache_getline_patcher
        self.linecache_getline_mock = linecache_getline_mock

    def tearDown(self):
        self.source_fp.close()
        self.target_fp.close()
        self.linecache_getline_patcher.stop()

    def test_init(self):
        ds = CnnDailymailDataset(self.source_fp.name, self.target_fp.name)
        for i, x in enumerate(ds):
            self.assertTupleEqual(x, (self.source_lines[i], self.target_lines[i]))

    def test_to_dict(self):
        ds = CnnDailymailDataset(self.source_fp.name, self.target_fp.name)
        source_field_name = 'source'
        target_field_name = 'target'
        ds_dict = ds.to_dict(source_field_name,
                             target_field_name)
        for i, x in enumerate(ds_dict):
            self.assertDictEqual(x, {source_field_name: self.source_lines[i],
                                     target_field_name: self.target_lines[i]})

    def test_to_allennlp(self):
        ds = CnnDailymailDataset(self.source_fp.name, self.target_fp.name)
        ds_allennlp = ds.to_allennlp()
        for i, x in enumerate(ds_allennlp):
            self.assertIsInstance(x, Instance)
            source_tokens = [token.text for token in x.fields['source_tokens'].tokens[1:-1]]
            self.assertListEqual(source_tokens, self.source_lines[i].split())
            target_tokens = [token.text for token in x.fields['target_tokens'].tokens[1:-1]]
            self.assertListEqual(target_tokens, self.target_lines[i].split())
