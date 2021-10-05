import shutil
import tempfile
from types import ClassMethodDescriptorType
from unittest import TestCase, mock

from lineflow import download
from lineflow.datasets.scitldr import get_scitldr


class SciTLDRTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default_cache_root = download.get_cache_root()
        cls.temp_dir = tempfile.mkdtemp()
        download.set_cache_root(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        download.set_cache_root(cls.default_cache_root)
        shutil.rmtree(cls.temp_dir)

    def test_get_scitldr(self):
        raw = get_scitldr(mode="a")
        for key in [
            "source",
            "source_labels",
            "rouge_scores",
            "paper_id",
            "target",
            "title",
        ]:
            with self.subTest(key=key):
                self.assertIn(key, raw['train'][0])

    def test_get_scitldr_twice(self):
        get_scitldr()
        with mock.patch(
            "lineflow.datasets.scitldr.pickle", autospec=True
        ) as mock_pickle:
            get_scitldr()
        mock_pickle.dump.assert_not_called()
        mock_pickle.load.assert_called_once()
