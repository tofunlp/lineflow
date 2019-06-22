from unittest import TestCase
import tempfile

from lineflow.torch import TextDataset


class TextDatasetTestCase(TestCase):

    def setUp(self):
        self.fp = tempfile.NamedTemporaryFile()
        self.n = 100
        for i in range(self.n):
            self.fp.write(f'line #{str(i).zfill(3)}\n'.encode('utf-8'))
        self.size = self.fp.tell()
        self.fp.seek(0)
        self.data = TextDataset(self.fp.name)

    def tearDown(self):
        self.fp.close()

    def test_dunder_init(self):
        self.assertEqual(self.data._path, self.fp.name)
        self.assertEqual(self.data._total_size, self.size)

    def test_dunder_iter(self):
        for x, i in zip(self.data, range(self.n)):
            self.assertEqual(x, f'line #{str(i).zfill(3)}')

    def test_pytorch_dataloader(self):
        from itertools import chain
        from torch.utils.data import DataLoader
        loader = DataLoader(self.data,
                            batch_size=16,
                            collate_fn=lambda x: x,
                            shuffle=False,
                            num_workers=2)
        for x, i in zip(sorted(chain.from_iterable(loader)), range(self.n)):
            self.assertEqual(x, f'line #{str(i).zfill(3)}')
