#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from process import get_dirname_num
from unittest import TestCase


class Test_Process(TestCase):
    def test_get_dirname_num(self):
        pattern = r'test_([0-9]+)'
        print type(pattern)
        dirpath = "/Onepiece/project/PythonProject/Lazy/bmdiagnosis/logs/tf_logs/resnet/log4tb/test_1"
        self.assertEqual(1, get_dirname_num(dirpath, pattern))

        dirpath = "/Onepiece/project/PythonProject/Lazy/bmdiagnosis/logs/tf_logs/resnet/log4tb/test_2"
        self.assertEqual(2, get_dirname_num(dirpath, pattern))

        dirpath = "/Onepiece/project/PythonProject/Lazy/bmdiagnosis/logs/tf_logs/resnet/log4tb/"
        self.assertEqual(None, get_dirname_num(dirpath, pattern))



