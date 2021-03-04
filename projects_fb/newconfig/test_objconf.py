import unittest

from detectron2.config.instantiate import instantiate

from newconfig import LazyCall as L


class TestClass:
    def __init__(self, int_arg, list_arg=None, dict_arg=None, extra_arg=None):
        self.int_arg = int_arg
        self.list_arg = list_arg
        self.dict_arg = dict_arg
        self.extra_arg = extra_arg

    def __call__(self, call_arg):
        return call_arg + self.int_arg


class TestConstruction(unittest.TestCase):
    def test_basic_construct(self):
        objconf = L(TestClass)(
            int_arg=3,
            list_arg=[10],
            dict_arg={},
            extra_arg=L(TestClass)(int_arg=4, list_arg="${..list_arg}"),
        )

        obj = instantiate(objconf)
        self.assertIsInstance(obj, TestClass)
        self.assertEqual(obj.int_arg, 3)
        self.assertEqual(obj.extra_arg.int_arg, 4)
        self.assertEqual(obj.extra_arg.list_arg, obj.list_arg)

        objconf.extra_arg.list_arg = [5]
        obj = instantiate(objconf)
        self.assertIsInstance(obj, TestClass)
        self.assertEqual(obj.extra_arg.list_arg, [5])

    def test_instantiate_other_obj(self):
        # do nothing for other obj
        self.assertEqual(instantiate(5), 5)
        x = [3, 4, 5]
        self.assertEqual(instantiate(x), x)
        x = TestClass(1)
        self.assertIs(instantiate(x), x)
        x = {"xx": "yy"}
        self.assertIs(instantiate(x), x)

    def test_instantiate_lazy_target(self):
        # _target_ is result of instantiate
        objconf = L(L(len)(int_arg=3))(call_arg=4)
        objconf._target_._target_ = TestClass
        self.assertEqual(instantiate(objconf), 7)
