import tempfile
import sys
import shutil
from unittest import TestCase
from unittest import mock

from lineflow import download
from lineflow.datasets.text_classification import TextClassification, \
    get_text_classification_dataset, urls \



class TextClassificationTestCaseBase(TestCase):

    names = list(urls.keys())
    sizes = [(120_000, 7_600), (450_000, 60_000), (560_000, 70_000), (560_000, 38_000),
             (650_000, 50_000), (1_400_000, 60_000), (3_600_000, 400_000), (3_000_000, 650_000)]

    def setUp(self):
        self.default_cache_root = download.get_cache_root()
        self.temp_dir = tempfile.mkdtemp()
        download.set_cache_root(self.temp_dir)
        self.patcher = mock.patch('lineflow.datasets.text_classification.sys.maxsize',
                                  int(sys.float_info.max))
        self.patcher.start()

    def tearDown(self):
        download.set_cache_root(self.default_cache_root)
        shutil.rmtree(self.temp_dir)
        self.patcher.stop()

    def get_text_classification_dataset(self, name, train_size, test_size):
        raw = get_text_classification_dataset(name)
        # train
        self.assertIn('train', raw)
        self.assertEqual(len(raw['train']), train_size)
        # test
        self.assertIn('test', raw)
        self.assertEqual(len(raw['test']), test_size)

    def get_text_classification_dataset_twice(self, name):
        get_text_classification_dataset(name)
        with mock.patch('lineflow.datasets.text_classification.pickle', autospec=True) as \
                mock_pickle:
            get_text_classification_dataset(name)
        mock_pickle.dump.assert_not_called()
        self.assertEqual(mock_pickle.load.call_count, 1)

    def loads_each_split(self, name, train_size, test_size):
        train = TextClassification(name, split='train')
        self.assertEqual(len(train), train_size)
        test = TextClassification(name, split='test')
        self.assertEqual(len(test), test_size)

    def test_raises_key_error_with_invalid_name(self):
        with self.assertRaises(KeyError):
            TextClassification('invalid_name')

    def raises_value_error_with_invalid_split(self, name):
        with self.assertRaises(ValueError):
            TextClassification(name, split='invalid_split')


class AgNewsTestCase(TextClassificationTestCaseBase):

    def setUp(self):
        super(AgNewsTestCase, self).setUp()
        self.name = self.names[0]
        self.size = self.sizes[0]

    def test_get_text_classification_dataset(self):
        self.get_text_classification_dataset(self.name, *self.size)

    def test_get_text_classification_dataset_twice(self):
        self.get_text_classification_dataset_twice(self.name)

    def test_loads_each_split(self):
        self.loads_each_split(self.name, *self.size)

    def test_raises_value_error_with_invalid_split(self):
        self.raises_value_error_with_invalid_split(self.name)
