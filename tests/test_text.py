from unittest import TestCase
import tempfile

import lineflow
from lineflow.core import RandomAccessConcat, RandomAccessZip
from lineflow import TextDataset, CsvDataset
from lineflow.text import RandomAccessText, RandomAccessCsv


class RandomAccessTextTestCase(TestCase):

    def setUp(self):
        self.length = 100

        fp = tempfile.NamedTemporaryFile()
        for i in range(self.length):
            fp.write(f'line #{i}\n'.encode('utf-8'))
        fp.seek(0)
        self.fp = fp

    def tearDown(self):
        self.fp.close()

    def test_init(self):
        text = RandomAccessText(self.fp.name)
        self.assertEqual(text._path, self.fp.name)
        self.assertEqual(text._length, None)

    def test_getitem(self):
        text = RandomAccessText(self.fp.name)
        for i in range(self.length):
            self.assertEqual(text[i], f'line #{i}')

    def test_raises_index_error_with_invalid_index(self):
        text = RandomAccessText(self.fp.name)
        with self.assertRaises(IndexError):
            text[-1]
            text[self._length]

    def test_len(self):
        text = RandomAccessText(self.fp.name)
        self.assertEqual(len(text), self.length)


class RandomAccessCsvTestCase(TestCase):

    def setUp(self):
        lines = ['en,ja',
                 'this is English .,this is Japanese .',
                 'this is also English .,this is also Japanese .']
        self.lines = lines
        fp = tempfile.NamedTemporaryFile()
        for x in lines:
            fp.write(f'{x}\n'.encode('utf-8'))
        fp.seek(0)
        self.fp = fp

    def tearDown(self):
        self.fp.close()

    def test_loads_csv_with_header(self):
        data = RandomAccessCsv(self.fp.name, header=True)
        self.assertListEqual(data._header, self.lines[0].split(','))

    def test_iterates_csv_with_header(self):
        from collections import OrderedDict

        data = RandomAccessCsv(self.fp.name, header=True)
        expected = [OrderedDict(zip(data._header, line.split(','))) for line in self.lines[1:]]
        self.assertSequenceEqual(data, expected)
        for x, y in zip(data, expected):
            self.assertEqual(x, y)

    def test_loads_csv_without_header(self):
        data = RandomAccessCsv(self.fp.name, header=False)
        self.assertIsNone(data._header)

    def test_iterates_csv_without_header(self):
        data = RandomAccessCsv(self.fp.name, header=False)
        expected = [line.split(',') for line in self.lines]
        self.assertSequenceEqual(data, expected)
        for x, y in zip(data, expected):
            self.assertEqual(x, y)


class TextDatasetTestCase(TestCase):

    def setUp(self):
        lines = ['This is a test .', 'That is also a test .']
        fp = tempfile.NamedTemporaryFile()
        for x in lines:
            fp.write(f'{x}\n'.encode('utf-8'))
        fp.seek(0)
        self.lines = lines
        self.fp = fp

    def tearDown(self):
        self.fp.close()

    def test_text(self):
        fp = self.fp
        lines = self.lines

        data = TextDataset(fp.name)
        for x, y in zip(data, lines):
            self.assertEqual(x, y)

        for i, y in enumerate(lines):
            self.assertEqual(data[i], y)

        self.assertEqual(data._length, None)
        self.assertEqual(len(data), len(lines))
        self.assertEqual(data._length, len(lines))
        # check if length is cached
        self.assertEqual(len(data), len(lines))

        self.assertIsInstance(data._dataset, RandomAccessText)

        data = data.map(str.split)

        for x, y in zip(data, lines):
            self.assertEqual(x, y.split())

        self.assertIsInstance(data, lineflow.core.MapDataset)
        self.assertIsInstance(data._dataset, RandomAccessText)

    def test_zips_multiple_files(self):
        fp = self.fp
        lines = self.lines

        data = TextDataset([fp.name, fp.name], mode='zip')
        for x, y in zip(data, lines):
            self.assertTupleEqual(x, (y, y))
        for j, y in enumerate(lines):
            self.assertTupleEqual(data[j], (y, y))
        self.assertEqual(len(data), len(lines))
        self.assertEqual(data._length, len(lines))
        self.assertIsInstance(data._dataset, RandomAccessZip)
        self.assertIsInstance(data.map(lambda x: x)._dataset, RandomAccessZip)

    def test_concats_multiple_files(self):
        fp = self.fp
        lines = self.lines

        data = TextDataset([fp.name, fp.name], mode='concat')
        for x, y in zip(data, lines + lines):
            self.assertEqual(x, y)
        for j, y in enumerate(lines + lines):
            self.assertEqual(data[j], y)
        self.assertEqual(len(data), len(lines) * 2)
        self.assertEqual(data._length, len(lines) * 2)

        self.assertEqual(data[len(data) - 1], lines[-1])
        self.assertIsInstance(data._dataset, RandomAccessConcat)
        self.assertIsInstance(data.map(lambda x: x)._dataset, RandomAccessConcat)

    def test_raises_value_error_with_invalid_mode(self):
        with self.assertRaises(ValueError):
            TextDataset([self.fp.name, self.fp.name], mode='invalid_mode')


class CsvDatasetTestCase(TestCase):

    def setUp(self):
        lines = ['en,ja',
                 'this is English .,this is Japanese .',
                 'this is also English .,this is also Japanese .']
        self.lines = lines
        fp = tempfile.NamedTemporaryFile()
        for x in lines:
            fp.write(f'{x}\n'.encode('utf-8'))
        fp.seek(0)
        self.fp = fp

    def tearDown(self):
        self.fp.close()

    def test_keeps_original_dataset(self):
        data = CsvDataset(self.fp.name, header=True)
        for i in range(100):
            data = data.map(lambda x: x)
            self.assertIsInstance(data._dataset, RandomAccessCsv)
            self.assertEqual(len(data._funcs), i + 1)

        data = CsvDataset(self.fp.name)
        for i in range(100):
            data = data.map(lambda x: x)
            self.assertIsInstance(data._dataset, RandomAccessCsv)
            self.assertEqual(len(data._funcs), i + 1)
