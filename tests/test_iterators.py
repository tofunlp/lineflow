from unittest import TestCase

from lineflow import iterators


class IteratorsTestCase(TestCase):

    def setUp(self):
        self.data = range(100)

    def test_prefetch_iterator(self):
        it = iterators.PrefetchIterator(self.data, n_prefetch=5)
        repeat = 10

        for _ in range(repeat):
            for x, y in zip(self.data, it):
                self.assertEqual(x, y)
