import tempfile
import shutil
from unittest import TestCase
from unittest import mock

from lineflow import download
from lineflow.datasets.imdb import Imdb, get_imdb


class ImdbTestCase(TestCase):

    def setUp(self):
        self.default_cache_root = download.get_cache_root()
        self.temp_dir = tempfile.mkdtemp()
        download.set_cache_root(self.temp_dir)

    def tearDown(self):
        download.set_cache_root(self.default_cache_root)
        shutil.rmtree(self.temp_dir)

    def test_get_imdb(self):
        raw = get_imdb()
        # train
        self.assertIn('train', raw)
        self.assertEqual(len(raw['train']), 25_000)
        # test
        self.assertIn('test', raw)
        self.assertEqual(len(raw['test']), 25_000)

    def test_get_imdb_twice(self):
        get_imdb()
        with mock.patch('lineflow.datasets.imdb.pickle', autospec=True) as mock_pickle:
            get_imdb()
        mock_pickle.dump.assert_not_called()
        self.assertEqual(mock_pickle.load.call_count, 1)

    def test_loads_each_split(self):
        train = Imdb(split='train')
        self.assertEqual(len(train), 25_000)
        self.assertEqual(train[0][1], 0)
        test = Imdb(split='test')
        self.assertEqual(len(test), 25_000)
        self.assertEqual(test[0][1], 0)

    def test_raises_value_error_with_invalid_split(self):
        with self.assertRaises(ValueError):
            Imdb(split='invalid_split')
