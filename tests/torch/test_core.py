from unittest import TestCase

from lineflow.torch import Dataset


class DatasetTestCase(TestCase):

    def setUp(self):
        self.base = range(100)
        self.data = Dataset.range(100)

    def test_map(self):
        def f(x): return x ** 2

        for x, y in zip(self.data.map(f), map(f, self.base)):
            self.assertEqual(x, y)

    def test_filter(self):
        def f(x): return x % 2 == 0

        for x, y in zip(self.data.filter(f), filter(f, self.base)):
            self.assertEqual(x, y)

    def test_flat_map(self):
        from itertools import chain
        def f(x): return [x] * 5

        for x, y in zip(self.data.flat_map(f), chain.from_iterable(map(f, self.base))):
            self.assertEqual(x, y)

    def test_shuffles_data_with_buffer(self):
        for x, y in zip(sorted(self.data.shuffle(3)), self.base):
            self.assertEqual(x, y)

    def test_shuffles_data_without_buffer(self):
        for x, y in zip(sorted(self.data.shuffle()), self.base):
            self.assertEqual(x, y)

    def test_all(self):
        self.assertListEqual(self.data.all(), list(self.base))

    def test_first(self):
        self.assertEqual(self.data.first(), self.base[0])

    def test_take(self):
        n = 50
        self.assertListEqual(self.data.take(n), list(self.base[:n]))

    def test_range_with_pytorch_dataloader(self):
        from itertools import chain
        from torch.utils.data import DataLoader
        loader = DataLoader(self.data,
                            batch_size=16,
                            collate_fn=lambda x: x,
                            shuffle=False,
                            num_workers=2)
        self.assertListEqual(
            list(sorted(chain.from_iterable(loader))),
            list(self.base)
        )
