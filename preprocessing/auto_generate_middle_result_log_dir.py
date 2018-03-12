#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import logging
import os
import re

__author__ = 'Liu Lihao'

logger = logging.getLogger()


def _get_dirname_num(dirpath_or_dirname, pattern):
    """
    Use re module to get dirname's num.

    :param dirpath_or_dirname:
    :param pattern:
    :return:
    """
    dirname = str(dirpath_or_dirname.split("/")[-1])

    match = re.match(pattern, dirname)

    if match:
        return int(match.group(1))
    else:
        return None


def _get_subdir(log_dir_path, self_increasing_mode, pattern):
    """
    This func is used to get a new subdir, so we can avoid overwriting the old model and tensorboard subdir.

    :param log_dir_path:
    :param self_increasing_mode:
    :param pattern:
    :return:
    """
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    dir_list = [dirname for dirname in os.listdir(log_dir_path)
                if (os.path.isdir(os.path.join(log_dir_path, dirname)) and pattern in dirname)]

    if not dir_list or dir_list is []:
        log_dir_path = os.path.join(log_dir_path, "{}_1".format(pattern))
    else:
        max_num = 1

        for dirname in dir_list:
            dirname_num = _get_dirname_num(dirname, r"{}_([0-9]+)".format(pattern))
            max_num = max(max_num, dirname_num)

        if self_increasing_mode:
            log_dir_path = os.path.join(log_dir_path, "{}_{}".format(pattern, int(max_num + 1)))
        else:
            log_dir_path = os.path.join(log_dir_path, "{}_{}".format(pattern, int(max_num)))

    return log_dir_path


def _synchronize_and_generate_subdir(tensorboard_dir, model_dir, note_dir, self_increasing_mode, pattern):
    """
    Make sure model and tensorboard subdir have the same dir num, and generate them.

    :param tensorboard_dir:
    :param model_dir:
    :param note_dir:
    :param pattern:
    :return:
    """
    tensorboard_dir_list = tensorboard_dir.split("/")
    tensorboard_dir_prefix = "/".join(tensorboard_dir_list[:-1])
    tensorboard_dirname = tensorboard_dir_list[-1]

    model_dir_list = model_dir.split("/")
    model_dir_prefix = "/".join(model_dir_list[:-1])
    model_dirname = model_dir_list[-1]

    note_dir_list = note_dir.split("/")
    note_dir_prefix = "/".join(note_dir_list[:-1])
    note_dirname = note_dir_list[-1]

    num_tensorboard_dir = _get_dirname_num(tensorboard_dirname, r"{}_([0-9]+)".format(pattern))
    num_model_dir = _get_dirname_num(model_dirname, r"{}_([0-9]+)".format(pattern))
    num_note_dir = _get_dirname_num(note_dirname, r"{}_([0-9]+)".format(pattern))

    if not num_tensorboard_dir or not num_model_dir or not num_note_dir:
        raise ValueError("Logs dirname is invalid.")

    dirname, num_dir = (tensorboard_dirname, num_tensorboard_dir) \
        if num_tensorboard_dir > num_model_dir else (model_dirname, num_model_dir)

    dirname, _ = (dirname, num_dir) \
        if num_dir > num_note_dir else (note_dirname, num_note_dir)

    tensorboard_dir = os.path.join(tensorboard_dir_prefix, dirname)
    model_dir = os.path.join(model_dir_prefix, dirname)
    note_dir = os.path.join(note_dir_prefix, dirname)

    if self_increasing_mode:
        os.system("rm -rf {}".format(tensorboard_dir))
        os.system("rm -rf {}".format(model_dir))
        os.system("rm -rf {}".format(note_dir))
    else:
        # os.system("rm -rf {}".format(tensorboard_dir))
        os.system("rm -rf {}".format(note_dir))

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(note_dir):
        os.makedirs(note_dir)

    return tensorboard_dir, model_dir, note_dir


def _save_params_to_note_dir(args_config_file_path, note_dir):
    os.system("cp {} {}/".format(args_config_file_path, note_dir))


def genarate_middle_result_log_dir(args_config_file_path, tensorboard_dir, model_dir, note_dir,
                                   self_increasing_mode, pattern):
    tensorboard_dir = _get_subdir(tensorboard_dir, self_increasing_mode, pattern)
    model_dir = _get_subdir(model_dir, self_increasing_mode, pattern)
    note_dir = _get_subdir(note_dir, self_increasing_mode, pattern)

    tensorboard_dir, model_dir, note_dir = _synchronize_and_generate_subdir(tensorboard_dir,
                                                                            model_dir,
                                                                            note_dir,
                                                                            self_increasing_mode,
                                                                            pattern)

    _save_params_to_note_dir(args_config_file_path, note_dir)

    return tensorboard_dir, model_dir
