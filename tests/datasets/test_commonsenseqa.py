import shutil
import tempfile
from unittest import TestCase, mock

from lineflow import download
from lineflow.datasets.commonsenseqa import CommonsenseQA, get_commonsenseqa


class CommonsenseQATestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.default_cache_root = download.get_cache_root()
        cls.temp_dir = tempfile.mkdtemp()
        download.set_cache_root(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        download.set_cache_root(cls.default_cache_root)
        shutil.rmtree(cls.temp_dir)

    def test_get_commonsenseqa(self):
        raw = get_commonsenseqa()
        self.assertIn("train", raw)
        self.assertIn("dev", raw)
        self.assertIn("test", raw)

    def test_get_commonsenseqa_twice(self):
        get_commonsenseqa()
        with mock.patch("lineflow.datasets.commonsenseqa.pickle", autospec=True) as mock_pickle:
            get_commonsenseqa()
        mock_pickle.dump.assert_not_called()
        self.assertEqual(mock_pickle.load.call_count, 1)

    def test_loads_each_split(self):
        train = CommonsenseQA(split="train")
        dev = CommonsenseQA(split="dev")
        test = CommonsenseQA(split="test")

        self.assertEqual(len(train), 9741)
        self.assertEqual(len(dev), 1221)
        self.assertEqual(len(test), 1140)

    def test_raises_value_error_with_invalid_split(self):
        with self.assertRaises(ValueError):
            CommonsenseQA(split="invalid_split")
