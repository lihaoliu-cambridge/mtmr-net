#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from utils.config_helper import ConfigHelper
from process import process
import logging.config
import logging
import argparse
import os

__author__ = "Liu Lihao"


def set_logging(error_log_config_file_path, error_log_folder_path, error_log_level):
    if not os.path.exists(error_log_folder_path):
        os.makedirs(error_log_folder_path)

    # set log file path and level
    # log_config = ConfigHelper.load_config(os.path.join(os.path.dirname(__file__), "conf", "log.yaml"))
    log_config = ConfigHelper.load_config(error_log_config_file_path)
    log_config["handlers"]["file_handler"]["filename"] = os.path.join(error_log_folder_path, "logs")
    log_config["handlers"]["file_handler"]["level"] = error_log_level

    logging.config.dictConfig(log_config)


def main():
    # first read parameters
    parser = argparse.ArgumentParser()

    log_parameter = parser.add_argument_group("Error log parameters", "Parameters are used for error logging.")
    log_parameter.add_argument("--error_log_config_file_path",
                               default="./conf/log.yaml",
                               help="The Error log config file path.")
    log_parameter.add_argument("--error_log_folder_path",
                               default="./logs/error_logs/",
                               help="The error log output position.")
    log_parameter.add_argument("--error_log_level",
                               default="DEBUG",
                               choices=["DEBUG", "INFO", "ERROR", "WARNING", "CRITICAL"],
                               help="The error log level.")

    program_parameter = parser.add_argument_group("Initialization parameters",
                                                  "Parameters are used for initialize program.")
    program_parameter.add_argument("--args_config_file_path",
                                   default="./conf/args.yaml",
                                   help="The args config path which will be to used to init program.")

    args = parser.parse_args()

    # second set logging and get logger
    # print args.error_log_config_file_path, args.error_log_folder_path, args.error_log_level
    set_logging(args.error_log_config_file_path, args.error_log_folder_path, args.error_log_level)
    logger = logging.getLogger()

    # third start program
    logger.info("Begin program")
    try:
        # print args.args_config_file_path
        process(args.args_config_file_path)
    except Exception as e:
        logger.error(e.message)
        raise
    logger.info("Program end")


if __name__ == '__main__':
    main()
