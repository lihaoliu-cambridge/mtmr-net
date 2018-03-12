#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# todo:make thom real
from ur_file import func_1
import unittest

__author__ = 'LH Liu'


# todo: alter XXX & xxx
class TestXXX(unittest.TestCase):
    # add func
    def _list_equal(self, list_one, list_another):
        set_one = set(list_one)
        set_another = set(list_another)
        msg = '{} != {}'.format(list_one, list_another)
        if len(set_one) != len(set_another):
            raise self.failureException(msg)
        for element in set_one:
            if element not in set_another:
                raise self.failureException(msg)

    def test_func_1(self):
        ori_path = 'xxx'
        result = 'xxx'
        # todo: alter func_1 to real func from ur program
        self._list_equal(func_1(ori_path), result)


if __name__ == '__main__':
    unittest.main()
