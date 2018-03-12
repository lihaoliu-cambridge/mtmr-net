# !/usr/bin/env python
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from skimage import io, transform
from glob import glob
import matplotlib.pyplot as plt
import multiprocessing as mp
import SimpleITK as sitk
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


def main_accuracy():
    filelist = [
        "/home/lhliu/Onepiece/project/PythonProjects/MTMR-net/data/output_data/all_example/ranking_resnet_1/test/0"
        "/ranking_resnet_1_50_dropduplicate.csv",
        "/home/lhliu/Onepiece/project/PythonProjects/MTMR-net/data/output_data/all_example/ranking_resnet_1/test/0"
        "/ranking_resnet_1_150_dropduplicate.csv "
    ]

    for i in range(len(filelist)):
        for j in range(i + 1, len(filelist)):
            print filelist[i], filelist[j]
            result = pd.read_csv(filelist[i])
            result["main_key"] = result.apply(lambda x:
                                              str(x["seriesuid"]) + str(x["coordX"]) + str(x["coordY"]) + str(
                                                  x["coordZ"]),
                                              axis=1)

            tmp = pd.read_csv(filelist[j])
            tmp["main_key"] = tmp.apply(lambda x:
                                        str(x["seriesuid"]) + str(x["coordX"]) + str(x["coordY"]) + str(x["coordZ"]),
                                        axis=1)

            result = pd.merge(result, tmp, on="main_key")
            # print result.columns.values

            need_result = result[["main_key", "probability_x", "probability_y", "label_x", "label_y"]]
            need_result["output_label_x"] = need_result.apply(lambda x: np.argmax(np.array(eval(x["probability_x"]))),
                                                              axis=1)
            need_result["output_label_y"] = need_result.apply(lambda x: np.argmax(np.array(eval(x["probability_y"]))),
                                                              axis=1)
            need_result["output_label"] = need_result.apply(lambda x:
                                                            get_model_averger_result(x["probability_x"],
                                                                                     x["probability_y"]),
                                                            axis=1)
            if need_result[need_result["label_x"] != need_result["label_y"]].shape[0] != 0:
                raise ValueError

            print "model_1:", 1.0 * need_result[need_result["label_x"] == need_result["output_label_x"].tolist()].shape[
                0] / \
                              need_result.shape[0]
            print "model_2:", 1.0 * need_result[need_result["label_y"] == need_result["output_label_y"].tolist()].shape[
                0] / \
                              need_result.shape[0]
            print "model_average:", 1.0 * \
                                    need_result[need_result["label_x"] == need_result["output_label"].tolist()].shape[
                                        0] / \
                                    need_result.shape[0]


def get_model_averger_result_2(prob_1, prob_2):
    final_prob = np.add(prob_1, prob_2)

    return final_prob[-1] / 2


def main_auc():
    filelist = [
        "/home/lhliu/Onepiece/project/PythonProjects/MTMR-net/data/output_data/all_example/ranking_resnet_1/test/0"
        "/ranking_resnet_1_50_dropduplicate.csv",
        "/home/lhliu/Onepiece/project/PythonProjects/MTMR-net/data/output_data/all_example/ranking_resnet_1/test/0"
        "/ranking_resnet_1_150_dropduplicate.csv "
    ]

    for i in range(len(filelist)):
        for j in range(i + 1, len(filelist)):
            print filelist[i], filelist[j]
            result = pd.read_csv(filelist[i])
            result["main_key"] = result.apply(lambda x:
                                              str(x["seriesuid"]) + str(x["coordX"]) + str(x["coordY"]) + str(
                                                  x["coordZ"]),
                                              axis=1)

            tmp = pd.read_csv(filelist[j])
            tmp["main_key"] = tmp.apply(lambda x:
                                        str(x["seriesuid"]) + str(x["coordX"]) + str(x["coordY"]) + str(x["coordZ"]),
                                        axis=1)

            result = pd.merge(result, tmp, on="main_key")
            # print result.columns.values

            need_result = result
            need_result["total_prob"] = need_result.apply(lambda x:
                                                            get_model_averger_result_2(
                                                                np.array(eval(x["probability_x"])),
                                                                np.array(eval(x["probability_y"]))),
                                                            axis=1)
            need_result["output_label"] = need_result.apply(lambda x:
                                                            get_model_averger_result(x["probability_x"],
                                                                                     x["probability_y"]),
                                                            axis=1)

            from sklearn import metrics

            print "sensitivity:", metrics.recall_score(need_result["label_x"], need_result["total_prob"] > 0.5)

            tn, fp, fn, tp = metrics.confusion_matrix(need_result["label_x"], need_result["total_prob"] > 0.5).ravel()
            # print tn, fp, fn, tp
            specificity = 1.0 * tn / (tn + fp)
            print "specificity:", specificity
            print "accuracy:", metrics.accuracy_score(need_result["label_x"], need_result["total_prob"] > 0.5)
            print "precision:", metrics.precision_score(need_result["label_x"], need_result["total_prob"] > 0.5)
            print "recall:", metrics.recall_score(need_result["label_x"], need_result["total_prob"] > 0.5)
            print "auc:", metrics.roc_auc_score(need_result["label_x"], need_result["total_prob"])
            print "kappa", metrics.cohen_kappa_score(need_result["label_x"], need_result["total_prob"] > 0.5)


if __name__ == '__main__':
    main_auc()
