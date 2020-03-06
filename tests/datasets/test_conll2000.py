import shutil
import tempfile
from unittest import TestCase, mock

from lineflow import download
from lineflow.datasets.conll2000 import Conll2000, get_conll2000


class Conll2000TestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.default_cache_root = download.get_cache_root()
        cls.temp_dir = tempfile.mkdtemp()
        download.set_cache_root(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        download.set_cache_root(cls.default_cache_root)
        shutil.rmtree(cls.temp_dir)

    def test_get_conll2000(self):
        raw = get_conll2000()
        self.assertIn('train', raw)
        self.assertEqual(len(raw['train']), 8_937)
        self.assertIn('test', raw)
        self.assertEqual(len(raw['test']), 2_013)

    def test_get_conll2000_twice(self):
        get_conll2000()
        with mock.patch('lineflow.datasets.conll2000.pickle', autospec=True) as mock_pickle:
            get_conll2000()
        mock_pickle.dump.assert_not_called()
        self.assertEqual(mock_pickle.load.call_count, 1)

    def test_loads_each_split(self):
        train = Conll2000(split='train')
        self.assertEqual(len(train), 8_937)
        test = Conll2000(split='test')
        self.assertEqual(len(test), 2_013)

    def test_raises_value_error_with_invalid_split(self):
        with self.assertRaises(ValueError):
            Conll2000(split='invalid_split')
