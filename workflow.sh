#!/usr/bin/env bash

working_dir="~/Onepiece/project/PythonProject/all_feature_research"
error_log_config_file_path="./conf/log.yaml"
error_log_folder_path="./logs/error_logs"
error_log_level="DEBUG"
args_config_file_path="./conf/args.yaml"

cd ${working_dir}
# source ./environment_config.sh
# python_cmd_env = "/data/ssd/public/lhliu/env/anaconda2/bin/python"
python_cmd_env = "~/env/anaconda2/bin/python"
${python_cmd_env} ./main.py $@ --error_log_config_file_path=${error_log_config_file_path} --error_log_folder_path=${error_log_folder_path} --error_log_level=${error_log_level} --args_config_file_path=${args_config_file_path}
if [ $? != 0 ] ; then
    exit 1
fi
exit 0