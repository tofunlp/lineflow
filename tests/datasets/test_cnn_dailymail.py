import shutil
import tempfile
from unittest import TestCase, mock

from lineflow import download
from lineflow.datasets.cnn_dailymail import CnnDailymail, get_cnn_dailymail


class CnnDailymailTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.default_cache_root = download.get_cache_root()
        cls.temp_dir = tempfile.mkdtemp()
        download.set_cache_root(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        download.set_cache_root(cls.default_cache_root)
        shutil.rmtree(cls.temp_dir)

    def test_get_cnn_dailymail(self):
        raw = get_cnn_dailymail()
        # train
        self.assertIn('train', raw)
        self.assertEqual(len(raw['train']), 2)
        for x in raw['train']:
            self.assertEqual(len(x), 287_227)
        # dev
        self.assertIn('dev', raw)
        self.assertEqual(len(raw['dev']), 2)
        for x in raw['dev']:
            self.assertEqual(len(x), 13_368)
        # test
        self.assertIn('test', raw)
        self.assertEqual(len(raw['test']), 2)
        for x in raw['test']:
            self.assertEqual(len(x), 11_490)

    def test_get_cnn_dailymail_twice(self):
        get_cnn_dailymail()
        with mock.patch('lineflow.datasets.cnn_dailymail.pickle', autospec=True) as \
                mock_pickle:
            get_cnn_dailymail()
        mock_pickle.dump.assert_not_called()
        self.assertEqual(mock_pickle.load.call_count, 1)

    def test_loads_each_split(self):
        train = CnnDailymail(split='train')
        self.assertEqual(len(train), 287_227)
        dev = CnnDailymail(split='dev')
        self.assertEqual(len(dev), 13_368)
        test = CnnDailymail(split='test')
        self.assertEqual(len(test), 11_490)

    def test_raises_value_error_with_invalid_split(self):
        with self.assertRaises(ValueError):
            CnnDailymail(split='invalid_split')
