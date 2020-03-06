import shutil
import tempfile
from unittest import TestCase, mock

from lineflow import download
from lineflow.datasets.wikitext import WikiText2, WikiText103, get_wikitext


class WikiTextTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.default_cache_root = download.get_cache_root()
        cls.temp_dir = tempfile.mkdtemp()
        download.set_cache_root(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        download.set_cache_root(cls.default_cache_root)
        shutil.rmtree(cls.temp_dir)

    def test_get_wikitext(self):
        params = [('wikitext-2', 36_718, 3_760, 4_358),
                  ('wikitext-103', 1_801_350, 3_760, 4_358)]
        for name, train_size, dev_size, test_size in params:
            with self.subTest(name=name, train_size=train_size,
                              dev_size=dev_size, test_size=test_size):
                raw = get_wikitext(name)
                # train
                self.assertIn('train', raw)
                self.assertEqual(len(raw['train']), train_size)
                # dev
                self.assertIn('dev', raw)
                self.assertEqual(len(raw['dev']), dev_size)
                # test
                self.assertIn('test', raw)
                self.assertEqual(len(raw['test']), test_size)

    def test_get_wikitext_twice(self):
        for name in ('wikitext-2', 'wikitext-103'):
            with self.subTest(name=name):
                get_wikitext(name)
                with mock.patch('lineflow.datasets.wikitext.pickle', autospec=True) as mock_pickle:
                    get_wikitext(name)
                mock_pickle.dump.assert_not_called()
                self.assertEqual(mock_pickle.load.call_count, 1)

    def test_loads_each_split(self):
        params = [(WikiText2, 36_718, 3_760, 4_358),
                  (WikiText103, 1_801_350, 3_760, 4_358)]
        for dataset, train_size, dev_size, test_size in params:
            with self.subTest(dataset=dataset, train_size=train_size,
                              dev_size=dev_size, test_size=test_size):
                train = dataset(split='train')
                self.assertEqual(len(train), train_size)
                dev = dataset(split='dev')
                self.assertEqual(len(dev), dev_size)
                test = dataset(split='test')
                self.assertEqual(len(test), test_size)

    def test_raises_value_error_with_invalid_split(self):
        for dataset in (WikiText2, WikiText103):
            with self.subTest(dataset=dataset):
                with self.assertRaises(ValueError):
                    dataset(split='invalid_split')
