# !/usr/bin/env python
# -*- coding: utf-8 -*-
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import torch as t
import time
import json


def get_model_averger_result(prob_1, prob_2):
    prob_1 = np.array(eval(prob_1))
    prob_2 = np.array(eval(prob_2))

    final_prob = np.add(prob_1, prob_2)
    output_label = np.argmax(final_prob)

    return output_label


def get_average_score(x):
    sum_0, sum_1, times = 0, 0, 0.0
    for i in x["probability"]:
        if isinstance(i, str):
            i = eval(i)
        sum_0 += i[0]
        sum_1 += i[1]
        times += 1
    return [sum_0 / times, sum_1 / times]


def get_average_label(x):
    sum_0, sum_1, times = 0, 0, 0.0
    for i in x["probability"]:
        i = eval(i)
        sum_0 += i[0]
        sum_1 += i[1]
        times += 1
    return np.argmax(np.array([sum_0 / times, sum_1 / times]))


def _get_attributes_score(input_file_info):
        input_file_info["internalStructure_average"] = input_file_info["internalStructure"].map(
            lambda attribute_score: _get_normalized_average_score_for_internal_structure(attribute_score))

        input_file_info["calcification_average"] = input_file_info["calcification"].map(
            lambda attribute_score: _get_normalized_average_score_for_calcification(attribute_score))

        input_file_info["subtlety_average"] = input_file_info["subtlety"].map(
            lambda attribute_score: _get_normalized_average_score(attribute_score))

        input_file_info["sphericity_average"] = input_file_info["sphericity"].map(
            lambda attribute_score: _get_normalized_average_score(attribute_score))

        input_file_info["margin_average"] = input_file_info["margin"].map(
            lambda attribute_score: _get_normalized_average_score(attribute_score))

        input_file_info["lobulation_average"] = input_file_info["lobulation"].map(
            lambda attribute_score: _get_normalized_average_score(attribute_score))

        input_file_info["spiculation_average"] = input_file_info["spiculation"].map(
            lambda attribute_score: _get_normalized_average_score(attribute_score))

        input_file_info["texture_average"] = input_file_info["texture"].map(
            lambda attribute_score: _get_normalized_average_score(attribute_score))

        input_file_info["malignancy_average"] = input_file_info["malignancy"].map(
            lambda attribute_score: _get_normalized_average_score(attribute_score))

        return input_file_info


def _get_normalized_average_score(attribute_score):
        min_value = 1
        max_value = 5

        attribute_score = json.loads(attribute_score)

        if isinstance(attribute_score, list) and len(attribute_score) != 0:
            score_list = np.array(attribute_score)
            x = np.mean(score_list)
            return x
        else:
            raise ValueError()


def _get_normalized_average_score_for_internal_structure(attribute_score):
        min_value = 1
        max_value = 4

        attribute_score = json.loads(attribute_score)

        if isinstance(attribute_score, list) and len(attribute_score) != 0:
            score_list = np.array(attribute_score)
            x = np.mean(score_list)
            return x
        else:
            raise ValueError()


def _get_normalized_average_score_for_calcification(attribute_score):
        min_value = 1
        max_value = 6

        attribute_score = json.loads(attribute_score)

        if isinstance(attribute_score, list) and len(attribute_score) != 0:
            score_list = np.array(attribute_score)
            x = np.mean(score_list)
            return x
        else:
            raise ValueError()


def get_average_score_for_p_val(x):
    sum_0 = 0
    for i in x:
        sum_0 += (i * 4 + 1)
    return sum_0 / len(x)


def get_average_score_for_p_val_2(x, name=None):
    z_list = []
    for z in x["coordZ"]:
        z_list.append(z)

    if len(z_list) == 4:
        z_list = sorted(z_list)[1:-1]
    elif len(z_list) > 6:
        z_list = sorted(z_list)[3:-3]
    else:
        z_list = z_list

    sum_0, sum_1, times = 0, 0, 0.0
    for idx, i in x[["probability", "coordZ"]].iterrows():
        if i["coordZ"] in z_list:
            i_name = i[name]
            sum_0 += 2 * (i_name[0] * 4 + 1)
            times += 2
        else:
            i_name = i[name]
            sum_0 += (i_name[0] * 4 + 1)
            times += 1
    return sum_0 / times


def get_average_score_for_p_val_for_cal(x):
    sum_0 = 0
    for i in x:
        sum_0 += (i * 5 + 1)
    return sum_0 / len(x)


def get_average_score_for_p_val_for_cal_2(x, name=None):
    z_list = []
    for z in x["coordZ"]:
        z_list.append(z)

    if len(z_list) == 4:
        z_list = sorted(z_list)[1:-1]
    elif len(z_list) > 6:
        z_list = sorted(z_list)[3:-3]
    else:
        z_list = z_list

    sum_0, sum_1, times = 0, 0, 0.0
    for idx, i in x[["probability", "coordZ"]].iterrows():
        if i["coordZ"] in z_list:
            i_name = i[name]
            sum_0 += 2 * (i_name[0] * 5 + 1)
            times += 2
        else:
            i_name = i[name]
            sum_0 += (i_name[0] * 5 + 1)
            times += 1
    return sum_0 / times


def get_average_score_for_p_val_for_is(x):
    sum_0 = 0
    for i in x:
        sum_0 += (i * 3 + 1)
    return sum_0 / len(x)


def get_average_score_for_p_val_for_is_2(x, name=None):
    z_list = []
    for z in x["coordZ"]:
        z_list.append(z)

    if len(z_list) == 4:
        z_list = sorted(z_list)[1:-1]
    elif len(z_list) > 6:
        z_list = sorted(z_list)[3:-3]
    else:
        z_list = z_list

    sum_0, sum_1, times = 0, 0, 0.0
    for idx, i in x[["probability", "coordZ"]].iterrows():
        if i["coordZ"] in z_list:
            i_name = i[name]
            sum_0 += 2 * (i_name[0] * 3 + 1)
            times += 2
        else:
            i_name = i[name]
            sum_0 += (i_name[0] * 3 + 1)
            times += 1
    return sum_0 / times


filelist = [
        "/home/lhliu/Onepiece/project/PythonProjects/MTMR-net/data/output_data/all_example/ranking_resnet_1/test/0"
        "/ranking_resnet_1_50.csv",
        "/home/lhliu/Onepiece/project/PythonProjects/MTMR-net/data/output_data/all_example/ranking_resnet_1/test/0"
        "/ranking_resnet_1_150.csv "
    ]


def read_pd_and_set_main_key(filepath):
    result = pd.read_csv(filepath)

    result["main_key"] = result.apply(
        lambda x: str(x["seriesuid"]) + "_" + str(x["coordX"]) + "_" + str(x["coordY"]), axis=1)
    result = _get_attributes_score(result)
    # print \"".join(result.columns.values)

    return result


def print_origin_accuracy(result):
    result["output_label"] = result.apply(lambda x: np.argmax(np.array(eval(x["probability"]))), axis=1)

    print "SLICE Accuracy:", \
        1.0 * result[result["label"] == result["output_label"].tolist()].shape[0] / result.shape[0]


def main_calculation():
    for filepath in filelist:
        print "~~~~~~~~~~~~", filepath

        result = read_pd_and_set_main_key(filepath)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Probability ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print_origin_accuracy(result)

        nodule_result = result.groupby('main_key').apply(get_average_score).to_frame()
        result_with_nodule_score = result.set_index('main_key').join(nodule_result)
        result_with_nodule_score.rename(columns={result_with_nodule_score.columns[-1]: "nodule_score"}, inplace=True)
        result_with_nodule_score.reset_index(inplace=True)

        result_with_nodule_score["main_key"] = result_with_nodule_score.apply(
            lambda x: str(x["seriesuid"]) + "_" + str(x["coordX"]) + "_" + str(x["coordY"]), axis=1)
        one_copy = result_with_nodule_score.iloc[result_with_nodule_score['main_key'].drop_duplicates().index.values]
        one_copy["output_label_x"] = one_copy.apply(lambda x: np.argmax(np.array(x["nodule_score"])), axis=1)
        print "Nodule Accuracy:", 1.0 * one_copy[one_copy["label"] == one_copy["output_label_x"].tolist()].shape[0] / \
                                 one_copy.shape[0]

        one_copy["probability"] = one_copy["nodule_score"]
        one_copy = one_copy[[
            "seriesuid", "coordX", "coordY", "coordZ", "diameter_mm", "subtlety", "internalStructure",
             "calcification", "sphericity", "margin", "lobulation", "spiculation", "texture", "malignancy",
             "preprocessed_filename", "label", "probability", "subtlety_average", "internalStructure_average",
             "calcification_average", "sphericity_average", "margin_average", "lobulation_average",
             "spiculation_average", "texture_average", "subtlety_output_score", "internalStructure_output_score",
             "calcification_output_score", "sphericity_output_score", "margin_output_score",
             "lobulation_output_score", "spiculation_output_score", "texture_output_score"]]
        one_copy.to_csv(str(filepath[:-4]) + "_dropduplicate.csv", index=False)

        print one_copy[one_copy["diameter_mm"] > 10][one_copy["label"] == 0.0]


def attributes():
    for i in filelist:
        print "~~~~~~~~~~~~", i
        # result = pd.read_csv(i)
        result = read_pd_and_set_main_key(i)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Attribute ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for item in ['texture', 'subtlety', 'spiculation', 'sphericity',
                     'margin', 'lobulation', 'internalStructure', 'calcification']:
            if item == 'calcification':
                nodule_result = result.groupby('main_key').apply(
                    get_average_score_for_p_val_for_cal_2, args=("{}_output_score".format(item))).to_frame()
            elif item == 'internalStructure':
                nodule_result = result.groupby('main_key').apply(
                    get_average_score_for_p_val_for_is_2, args=("{}_output_score".format(item))).to_frame()
            else:
                nodule_result = result.groupby('main_key').apply(
                    get_average_score_for_p_val_2, args=("{}_output_score".format(item))).to_frame()

            nodule_result.rename(columns={nodule_result.columns[0]: "nodule_{}_score".format(item)}, inplace=True)
            result_with_nodule_score = result.set_index('main_key').join(nodule_result)
            result_with_nodule_score.reset_index(inplace=True)
            print result

            result_with_nodule_score["main_key"] = result_with_nodule_score.apply(
                lambda x: str(x["seriesuid"]) + "_" + str(x["coordX"]) + "_" + str(x["coordY"]), axis=1)
            one_copy = result_with_nodule_score.iloc[
                result_with_nodule_score['main_key'].drop_duplicates().index.values]
            _, p_val = spearmanr(one_copy["{}_average".format(item)], one_copy["nodule_{}_score".format(item)])
            # print one_copy["{}_average".format(item)], one_copy["{}_output_score".format(item)]
            # print "model_nodule_{}_p_val:".format(item), p_val
            print "Nodule Absolute Distance of {}:".format(item), np.mean(
                np.abs(np.subtract(one_copy["{}_average".format(item)],
                                   one_copy["nodule_{}_score".format(item)])))
            one_copy["{}_output_score".format(item)] = one_copy["nodule_{}_score".format(item)]


def calulate_nodule_score(result):
        result["main_key"] = result.apply(
            lambda x: str(x["seriesuid"]) + "_" + str(x["coordX"]) + "_" + str(x["coordY"]), axis=1)
        result = _get_attributes_score(result)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Probability ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # print_origin_accuracy(result)

        nodule_result = result.groupby('main_key').apply(get_average_score).to_frame()
        result_with_nodule_score = result.set_index('main_key').join(nodule_result)
        result_with_nodule_score.rename(columns={result_with_nodule_score.columns[-1]: "nodule_score"}, inplace=True)
        result_with_nodule_score.reset_index(inplace=True)

        result_with_nodule_score["main_key"] = result_with_nodule_score.apply(
            lambda x: str(x["seriesuid"]) + "_" + str(x["coordX"]) + "_" + str(x["coordY"]), axis=1)
        one_copy = result_with_nodule_score.iloc[result_with_nodule_score['main_key'].drop_duplicates().index.values]
        one_copy["output_label_x"] = one_copy.apply(lambda x: np.argmax(np.array(x["nodule_score"])), axis=1)
        print "Nodule Accuracy:", 1.0 * one_copy[one_copy["label"] == one_copy["output_label_x"].tolist()].shape[0] / \
                                 one_copy.shape[0]
        return 1.0 * one_copy[one_copy["label"] == one_copy["output_label_x"].tolist()].shape[0] / \
                                 one_copy.shape[0]


if __name__ == '__main__':
    main_calculation()
    # attributes()
