import shutil
import tempfile
from unittest import TestCase, mock

from lineflow import download
from lineflow.datasets.wmt14 import Wmt14, get_wmt14


class Wmt14TestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.default_cache_root = download.get_cache_root()
        cls.temp_dir = tempfile.mkdtemp()
        download.set_cache_root(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        download.set_cache_root(cls.default_cache_root)
        shutil.rmtree(cls.temp_dir)

    def test_get_wmt14(self):
        raw = get_wmt14()
        # train
        self.assertIn('train', raw)
        self.assertEqual(len(raw['train']), 2)
        for x in raw['train']:
            self.assertEqual(len(x), 4_500_966)
        # dev
        self.assertIn('dev', raw)
        self.assertEqual(len(raw['dev']), 2)
        for x in raw['dev']:
            self.assertEqual(len(x), 3_000)
        # test
        self.assertIn('test', raw)
        self.assertEqual(len(raw['test']), 2)
        for x in raw['test']:
            self.assertEqual(len(x), 3_003)

    def test_get_wmt14_twice(self):
        get_wmt14()
        with mock.patch('lineflow.datasets.wmt14.pickle', autospec=True) as \
                mock_pickle:
            get_wmt14()
        mock_pickle.dump.assert_not_called()
        self.assertEqual(mock_pickle.load.call_count, 1)

    def test_loads_each_split(self):
        train = Wmt14(split='train')
        self.assertEqual(len(train), 4_500_966)
        dev = Wmt14(split='dev')
        self.assertEqual(len(dev), 3_000)
        test = Wmt14(split='test')
        self.assertEqual(len(test), 3_003)

    def test_raises_value_error_with_invalid_split(self):
        with self.assertRaises(ValueError):
            Wmt14(split='invalid_split')
