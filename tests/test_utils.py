import unittest

from lineflow import utils


class ApplyTestCase(unittest.TestCase):

    def test_apply_tuple(self):
        tuple_data = (0, 1, 2)

        @utils.apply(0)
        @utils.apply(1)
        @utils.apply(2)
        def to_str(x): return str(x)

        self.assertTupleEqual(to_str(tuple_data),
                              tuple(str(x) for x in tuple_data))

        self.assertTupleEqual(utils.apply(1)(str)(tuple_data),
                              tuple(str(x) if i == 1 else x
                                    for i, x in enumerate(tuple_data)))

    def test_apply_list(self):
        list_data = [0, 1, 2]

        @utils.apply(0)
        @utils.apply(1)
        @utils.apply(2)
        def to_str(x): return str(x)

        self.assertListEqual(to_str(list_data),
                             [str(x) for x in list_data])

        self.assertListEqual(utils.apply(1)(str)(list_data),
                             [str(x) if i == 1 else x
                              for i, x in enumerate(list_data)])

    def test_apply_dict(self):
        dict_data = {'a': 0, 'b': 1, 'c': 2}

        @utils.apply('a')
        @utils.apply('b')
        @utils.apply('c')
        def to_str(x): return str(x)

        self.assertDictEqual(to_str(dict_data),
                             {k: str(v) for k, v in dict_data.items()})

        self.assertDictEqual(utils.apply('b')(str)(dict_data),
                             {k: str(v) if k == 'b' else v
                              for k, v in dict_data.items()})

    def test_raises_type_error_with_byte_string(self):
        with self.assertRaises(TypeError):
            utils.apply(0)(str)(b'invalid')


class ApplyAllTestCase(unittest.TestCase):

    def test_apply_tuple(self):
        tuple_data = (0, 1, 2)

        @utils.apply_all()
        def to_str(x): return str(x)

        self.assertTupleEqual(to_str(tuple_data),
                              tuple(str(x) for x in tuple_data))

        self.assertTupleEqual(utils.apply_all(1)(str)(tuple_data),
                              tuple(x if i == 1 else str(x)
                                    for i, x in enumerate(tuple_data)))

    def test_apply_list(self):
        list_data = [0, 1, 2]

        @utils.apply_all()
        def to_str(x): return str(x)

        self.assertListEqual(to_str(list_data),
                             [str(x) for x in list_data])

        self.assertListEqual(utils.apply_all(1)(str)(list_data),
                             [x if i == 1 else str(x)
                              for i, x in enumerate(list_data)])

    def test_apply_dict(self):
        dict_data = {'a': 0, 'b': 1, 'c': 2}

        @utils.apply_all()
        def to_str(x): return str(x)

        self.assertDictEqual(to_str(dict_data),
                             {k: str(v) for k, v in dict_data.items()})

        self.assertDictEqual(utils.apply_all('b')(str)(dict_data),
                             {k: v if k == 'b' else str(v)
                              for k, v in dict_data.items()})

    def test_raises_type_error_with_byte_string(self):
        with self.assertRaises(TypeError):
            utils.apply_all(0)(str)(b'invalid')

    def test_raises_value_error_with_map_function(self):
        with self.assertRaises(ValueError):
            utils.apply_all()(utils.MapFunction(0, str))
