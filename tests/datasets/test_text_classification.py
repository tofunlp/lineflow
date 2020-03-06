import shutil
import string
import sys
import tempfile
from unittest import TestCase, mock

from lineflow import download
from lineflow.datasets import text_classification
from lineflow.datasets.text_classification import (
    get_text_classification_dataset, urls)


class TextClassificationTestCaseBase(TestCase):

    names = list(urls.keys())
    sizes = [(120_000, 7_600), (450_000, 60_000), (560_000, 70_000), (560_000, 38_000),
             (650_000, 50_000), (1_400_000, 60_000), (3_600_000, 400_000), (3_000_000, 650_000)]

    @classmethod
    def setUpClass(cls):
        cls.default_cache_root = download.get_cache_root()
        cls.temp_dir = tempfile.mkdtemp()
        download.set_cache_root(cls.temp_dir)
        cls.patcher = mock.patch('lineflow.datasets.text_classification.sys.maxsize',
                                 int(sys.float_info.max))
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        download.set_cache_root(cls.default_cache_root)
        shutil.rmtree(cls.temp_dir)
        cls.patcher.stop()

    def name2class(self, name):
        return getattr(text_classification, string.capwords(name, '_').replace('_', ''))

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
        train = self.name2class(name)(split='train')
        self.assertEqual(len(train), train_size)
        test = self.name2class(name)(split='test')
        self.assertEqual(len(test), test_size)

    def test_raises_key_error_with_invalid_name(self):
        with self.assertRaises(KeyError):
            get_text_classification_dataset('invalid_name')

    def raises_value_error_with_invalid_split(self, name):
        with self.assertRaises(ValueError):
            self.name2class(name)(split='invalid_split')


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


class SogouNewsTestCase(AgNewsTestCase):

    def setUp(self):
        super(SogouNewsTestCase, self).setUp()
        self.name = self.names[1]
        self.size = self.sizes[1]


class DbpediaTestCase(AgNewsTestCase):

    def setUp(self):
        super(DbpediaTestCase, self).setUp()
        self.name = self.names[2]
        self.size = self.sizes[2]


class YelpReviewPolarityTestCase(AgNewsTestCase):

    def setUp(self):
        super(YelpReviewPolarityTestCase, self).setUp()
        self.name = self.names[3]
        self.size = self.sizes[3]


class YelpReviewFullTestCase(AgNewsTestCase):

    def setUp(self):
        super(YelpReviewFullTestCase, self).setUp()
        self.name = self.names[4]
        self.size = self.sizes[4]


class YahooAnswersTestCase(AgNewsTestCase):

    def setUp(self):
        super(YahooAnswersTestCase, self).setUp()
        self.name = self.names[5]
        self.size = self.sizes[5]


class AmazonReviewPolarityTestCase(AgNewsTestCase):

    def setUp(self):
        super(AmazonReviewPolarityTestCase, self).setUp()
        self.name = self.names[6]
        self.size = self.sizes[6]


class AmazonReviewFullTestCase(AgNewsTestCase):

    def setUp(self):
        super(AmazonReviewFullTestCase, self).setUp()
        self.name = self.names[7]
        self.size = self.sizes[7]
