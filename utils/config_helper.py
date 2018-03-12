#!/usr/bin/env python
# coding=utf-8
from __future__ import unicode_literals
import codecs
import yaml
import os


class ConfigHelper(object):
    @staticmethod
    def load_config(config_file_path):
        if not os.path.isfile(config_file_path):
            raise IOError('Config file {} not existed.'.format(config_file_path))

        with codecs.open(config_file_path, mode='r', encoding='utf-8') as config_file:
            config_dict = yaml.load(config_file)
            return config_dict
