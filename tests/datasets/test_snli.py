import shutil
import tempfile
from unittest import TestCase, mock

from lineflow import download
from lineflow.datasets.snli import Snli, get_snli


class SnliTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.default_cache_root = download.get_cache_root()
        cls.temp_dir = tempfile.mkdtemp()
        download.set_cache_root(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        download.set_cache_root(cls.default_cache_root)
        shutil.rmtree(cls.temp_dir)

    def test_get_snil(self):
        raw = get_snli()
        self.assertIn('train', raw)
        self.assertEqual(len(raw['train']), 550_152)
        self.assertIn('dev', raw)
        self.assertEqual(len(raw['dev']), 10_000)
        self.assertIn('test', raw)
        self.assertEqual(len(raw['test']), 10_000)

    def test_get_snli_twice(self):
        get_snli()
        with mock.patch('lineflow.datasets.snli.pickle', autospec=True) as mock_pickle:
            get_snli()
        mock_pickle.dump.assert_not_called()
        self.assertEqual(mock_pickle.load.call_count, 1)

    def test_loads_each_split(self):
        train = Snli(split='train')
        self.assertEqual(len(train), 550_152)
        dev = Snli(split='dev')
        self.assertEqual(len(dev), 10_000)
        test = Snli(split='test')
        self.assertEqual(len(test), 10_000)

    def test_raises_value_error_with_invalid_split(self):
        with self.assertRaises(ValueError):
            Snli(split='invalid_split')
