#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from utils.config_helper import ConfigHelper
from utils.model_handler_2d_regression import ModelHandler
from preprocessing.auto_generate_middle_result_log_dir import genarate_middle_result_log_dir
import logging
import os

__author__ = 'Liu Lihao'

logger = logging.getLogger()


def process(args_config_file_path):
    args_config = ConfigHelper.load_config(args_config_file_path)

    # 1. get training or testing mode:
    is_training = args_config["is_training"]
    # 2. get middle result log dir params
    middle_result_log_dir_params = args_config["middle_result_log_dir_params"]
    # 3. get model params:
    model_params = args_config["model_params"]
    # 4. get running params from and run your model
    training_params = args_config["training_params"]
    testing_params = args_config["testing_params"]

    # step 1. get dirname pattern
    pattern, default_pattern = middle_result_log_dir_params["pattern"], middle_result_log_dir_params["default_pattern"]
    if pattern is None or len(pattern) == 0:
        pattern = default_pattern

    # step 2. get and generate middle result log dir
    if is_training:
        # get middle result log dir automatically from config args under training mode
        tensorboard_dir, model_dir = genarate_middle_result_log_dir(args_config_file_path,
                                                                    middle_result_log_dir_params["tensorboard_dir"],
                                                                    middle_result_log_dir_params["model_dir"],
                                                                    middle_result_log_dir_params["note_dir"],
                                                                    middle_result_log_dir_params["self_increasing_mode"]
                                                                    and is_training,
                                                                    pattern)

        training_params.update({"tensorboard_dir": tensorboard_dir})
        training_params.update({"model_dir": model_dir})

    # step 3. todo: train or test your model
    if is_training:
        set_up_visiable_gpu(training_params.get("gpu_num"), training_params.get("gpu_device_num"))

        model_handler = ModelHandler(model_params)
        model_handler.train(training_params)
    else:
        set_up_visiable_gpu(testing_params.get("gpu_num"), testing_params.get("gpu_device_num"))

        model_handler = ModelHandler(model_params)
        model_handler.test(testing_params)


def set_up_visiable_gpu(gpu_num, gpu_device_num):
    if gpu_device_num is not None and len(gpu_device_num) == gpu_num:
        visiable_gpu_device_num = gpu_device_num
    else:
        print "Gpu num is not the same as len of gpu_device_num."
        visiable_gpu_device_num = range(gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in visiable_gpu_device_num])


if __name__ == '__main__':
    process("./conf/args.yaml")
