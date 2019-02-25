from unittest import TestCase

from lineflow import reducers


class ReducersTestCase(TestCase):

    def setUp(self):
        self.data = range(100)

    def test_call(self):
        with self.assertRaises(NotImplementedError):
            reducers.Reducer()(self.data)

    def test_filter(self):
        reduced = reducers.Filter(lambda x: x % 2 == 0)(self.data)
        expected = [x for x in self.data if x % 2 == 0]
        for x, y in zip(reduced, expected):
            self.assertEqual(x, y)
        self.assertTrue(reduced._done)
        for x, y in zip(reduced, expected):
            self.assertEqual(x, y)

    def test_flat_map(self):
        reduced = reducers.FlatMap(lambda x: [x] * 3)(self.data)
        expected = [[x] * 3 for x in self.data]
        expected = [x for xs in expected for x in xs]
        for x, y in zip(reduced, expected):
            self.assertEqual(x, y)
        self.assertTrue(reduced._done)
        for x, y in zip(reduced, expected):
            self.assertEqual(x, y)

    def test_concat(self):
        reduced = reducers.Concat()(self.data, self.data)
        expected = list(self.data) * 2
        for x, y in zip(reduced, expected):
            self.assertEqual(x, y)
        self.assertTrue(reduced._done)
        for x, y in zip(reduced, expected):
            self.assertEqual(x, y)

    def test_zip(self):
        reduced = reducers.Zip()(self.data, self.data)
        expected = list(zip(self.data, self.data))
        for x, y in zip(reduced, expected):
            self.assertEqual(x, y)
        self.assertTrue(reduced._done)
        for x, y in zip(reduced, expected):
            self.assertEqual(x, y)
