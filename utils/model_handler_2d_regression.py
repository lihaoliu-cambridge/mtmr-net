# !/usr/bin/env python
# coding=utf-8
from __future__ import unicode_literals
from preprocessing.data_handler_2d_regression import LungNodule
from torch.utils.data import DataLoader
from torchnet import meter
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from scipy.stats import spearmanr
import torch.nn as nn
import torch.optim as optim
import torch as t
import pandas as pd
import numpy as np
import model as models
import logging
import scipy
import math
import time
import copy
import os

logger = logging.getLogger()

attribute_idx_dict = {1: "subtlety", 2: "internalStructure", 3: "calcification", 4: "sphericity",
                      5: "margin", 6: "lobulation", 7: "spiculation", 8: "texture"}


class ModelHandler(object):
    def __init__(self, model_params):
        # self.model = getattr(models, model_params.get("model_name"))(**model_params).eval()
        self.model = getattr(models, model_params.get("model_name"))().eval()

        self.model_initial_params = copy.deepcopy(self.model.state_dict())

        self.initial_features_params = ["features." + str(item) for item in self.model.features.state_dict().keys()]
        self.initial_classifier_params = ["classifier." + str(item) for item in
                                          self.model.classifier.state_dict().keys()]

    # ~~~~~~~~~~ TRAINING PROCESS TOOLS ~~~~~~~~~~
    # step 0: get cross validation folder
    def _get_folders_info_from_dict(self, training_params):
        k_folder = training_params["k_folder"]
        is_cross_validation = training_params["is_cross_validation"]

        if is_cross_validation:
            folders = list(reversed(range(k_folder)))
        else:
            folders = 0

        return k_folder, folders

    def _generate_log_dir_and_writer(self, training_params, ith_folder):
        tb_dir_i = os.path.join(training_params.get("tensorboard_dir"), str(ith_folder))
        os.system("rm -rf {}".format(tb_dir_i))
        if not os.path.exists(tb_dir_i):
            os.makedirs(tb_dir_i)

        writer = SummaryWriter(tb_dir_i)
        json_filepath = os.path.join(tb_dir_i, "all_scalars.json")

        return writer, json_filepath

    # step 1: get dataloader
    def _get_dataloader_info_from_dict(self, training_params):
        image_dir = training_params.get("multi_thread_data_dir")
        data_pattern = training_params.get("data_pattern")
        annotation_file_path = training_params.get("annotation_file_path")
        hard_example_mining_file_path = training_params.get("hard_example_mining_file_path")
        repeat_times = training_params.get("repeat_times")
        batch_size = training_params.get("batch_size")
        thread_num = training_params.get("thread_num")
        gpu_num = training_params.get("gpu_num")

        print "Image Dir:", image_dir

        return image_dir, data_pattern, annotation_file_path, \
               hard_example_mining_file_path, repeat_times, batch_size, gpu_num, thread_num

    def _get_train_and_validation_dataloader(self, training_params, folder_num, ith_folder):
        image_dir, data_pattern, annotation_file_path, hem_file_path, repeat_times, batch_size, gpu_num, thread_num = \
            self._get_dataloader_info_from_dict(training_params)

        if isinstance(hem_file_path, list):
            hem_file_path_i = hem_file_path[ith_folder]
        else:
            hem_file_path_i = hem_file_path

        train_data = LungNodule(ith_folder, folder_num, image_dir, data_pattern, annotation_file_path,
                                is_training=True,
                                is_testing=False,
                                hard_example_mining_file_path=hem_file_path_i,
                                repeat_times=repeat_times)
        val_data = LungNodule(ith_folder, folder_num, image_dir, data_pattern, annotation_file_path,
                              is_training=False,
                              is_testing=False)
        # val_data = LungNodule(image_dir, data_pattern, annotation_file_path, is_training=False, is_testing=False)
        train_dataloader = DataLoader(train_data, batch_size * gpu_num, shuffle=True, num_workers=thread_num)
        val_dataloader = DataLoader(val_data, batch_size * gpu_num, shuffle=False, num_workers=thread_num)

        return train_dataloader, val_dataloader, len(train_data)

    # step 2(1): configure model
    def _get_pretrained_model_info_from_dict(self, training_params):
        pretrained_model_path = training_params.get("pretrained_model_path")
        another_pretrained_model_path = training_params.get("another_pretrained_model_path")
        unnecessary_keys = training_params.get("unnecessary_keys")
        special_model_load_method = training_params.get("special_model_load_method")

        return pretrained_model_path, another_pretrained_model_path, unnecessary_keys, special_model_load_method

    def _load_pretrained_model(self, pretrained_model_path,
                               another_pretrained_model_path, unnecessary_keys):
        if pretrained_model_path is not None and str(pretrained_model_path) != "":
            pretrained_dict = t.load(pretrained_model_path)
            model_dict = copy.deepcopy(self.model_initial_params)

            # 1. filter out unnecessary keys
            useful_pretrained_dict = {}
            unused_pretrained_dict = []
            for key, value in pretrained_dict.items():
                if key in model_dict and key not in unnecessary_keys and value.shape == model_dict[key].shape:
                    useful_pretrained_dict.update({key: value})
                else:
                    unused_pretrained_dict.append(key)
            print "Unused pretrained layer:", unused_pretrained_dict

            # 2. overwrite entries in the existing state dict
            model_dict.update(useful_pretrained_dict)
        else:
            model_dict = copy.deepcopy(self.model_initial_params)

        self.model.load_state_dict(model_dict)

    def _load_pretrained_model_2(self, pretrained_model_path,
                                 another_pretrained_model_path, unnecessary_keys):
        if pretrained_model_path is not None and str(pretrained_model_path) != "":
            model_dict = copy.deepcopy(self.model_initial_params)

            # 1. filter out unnecessary keys
            pretrained_dict = t.load(pretrained_model_path)
            useful_pretrained_dict = {}
            unused_pretrained_dict = {}
            for original_key, value in pretrained_dict.items():
                if original_key in model_dict:
                    key = original_key
                elif "features." + str(original_key) in model_dict:
                    key = "features." + str(original_key)
                else:
                    key = original_key

                if key in model_dict and key not in unnecessary_keys and value.shape == model_dict[key].shape:
                    if key in self.initial_features_params:
                        useful_pretrained_dict.update({key: value})
                        # elif key in self.initial_classifier_malignacy_params:
                        #     useful_pretrained_dict.update({key: value})
                        # elif key in self.initial_classifier_attribute_params:
                        #     useful_pretrained_dict.update({key: value})
                        # else:
                        #     raise
                else:
                    unused_pretrained_dict.update({key: value})
            model_dict.update(useful_pretrained_dict)
            print "First model used pretrained layer:", len(useful_pretrained_dict.keys())
            print "First model unused pretrained layer:", len(unused_pretrained_dict.keys())
        else:
            model_dict = copy.deepcopy(self.model_initial_params)

        # 3. load the new state dict
        self.model.load_state_dict(model_dict)

    def _load_pretrained_model_from_video_project(self, pretrained_model_path,
                                                  another_pretrained_model_path, unnecessary_keys):
        if pretrained_model_path is not None and str(pretrained_model_path) != "":
            pretrained_dict = t.load(pretrained_model_path)
            pretrained_dict = {key.split("module.", 1)[-1]: value
                               for key, value in pretrained_dict["state_dict"].items()}
            model_dict = copy.deepcopy(self.model_initial_params)

            # 1. filter out unnecessary keys
            useful_pretrained_dict = {}
            unused_pretrained_dict = []
            for key, value in pretrained_dict.items():
                if key == "conv1.weight":
                    tmp_value = value.cpu().numpy()
                    tmp_value = np.mean(tmp_value, axis=1)
                    tmp_value = np.expand_dims(tmp_value, 1)
                    mean_value = t.from_numpy(tmp_value).cuda()
                    useful_pretrained_dict.update({key: mean_value})
                else:
                    if key in model_dict and key not in unnecessary_keys and value.shape == model_dict[key].shape:
                        # print "pretrained layer"
                        useful_pretrained_dict.update({key: value})
                    else:
                        unused_pretrained_dict.append(key)
            print "Used pretrained layer:\n", len(useful_pretrained_dict.keys())
            print "Unused pretrained layer:\n", len(unused_pretrained_dict)

            # 2. overwrite entries in the existing state dict
            model_dict.update(useful_pretrained_dict)
        else:
            model_dict = copy.deepcopy(self.model_initial_params)

        self.model.load_state_dict(model_dict)

    def _set_up_model(self, training_params, ith_folder):
        pretrained_model_path, another_pretrained_model_path, unnecessary_keys, special_model_load_method = \
            self._get_pretrained_model_info_from_dict(training_params)

        if isinstance(pretrained_model_path, list) and isinstance(another_pretrained_model_path, list) and \
                        len(unnecessary_keys) != 0 and isinstance(unnecessary_keys[0], list):
            pretrained_model_path = pretrained_model_path[ith_folder]
            another_pretrained_model_path = another_pretrained_model_path[ith_folder]
            unnecessary_keys = unnecessary_keys[ith_folder]

        print "\nPretrained Model Path:", pretrained_model_path
        print "Another pretrained Model Path:", another_pretrained_model_path

        if special_model_load_method == 2:
            self._load_pretrained_model_2(pretrained_model_path=pretrained_model_path,
                                          another_pretrained_model_path=another_pretrained_model_path,
                                          unnecessary_keys=unnecessary_keys)
        else:
            self._load_pretrained_model(pretrained_model_path=pretrained_model_path,
                                        another_pretrained_model_path=another_pretrained_model_path,
                                        unnecessary_keys=unnecessary_keys)

    # step 2(2): map to gpu
    def _get_gpu_info_from_dict(self, training_params):
        gpu_num = training_params.get("gpu_num")
        gpu_device_num = training_params.get("gpu_device_num")

        return gpu_num, gpu_device_num

    def _config_gpu_info(self, training_params):
        gpu_num, gpu_device_num = self._get_gpu_info_from_dict(training_params)

        if gpu_device_num is not None and len(gpu_device_num) == gpu_num:
            visiable_gpu_device_num = range(gpu_num)
        else:
            print "Gpu num is not the same as len of gpu_device_num."
            visiable_gpu_device_num = range(gpu_num)

        return visiable_gpu_device_num

    def _transfer_data_to_gpu(self, var, visiable_gpu_device_num):
        var_class = str(var.__class__)
        if "model" in var_class and visiable_gpu_device_num is not None and len(visiable_gpu_device_num) != 0:
            var = nn.DataParallel(var, device_ids=visiable_gpu_device_num)
            var.cuda()
        elif "torch.autograd.variable.Variable" in var_class:
            if visiable_gpu_device_num is not None and len(visiable_gpu_device_num) != 0:
                return var.cuda()
            else:
                return var
        elif "torch" in var_class and "Tensor" in var_class:
            if visiable_gpu_device_num is not None and len(visiable_gpu_device_num) != 0:
                return Variable(var.cuda())
            else:
                return Variable(var)
        else:
            raise TypeError("No matched var class for transferring to GPU.")

    # step 3: criterion and optimizer
    def _get_algorithm_info_from_dict(self, i, training_params):
        loss_function_name = training_params.get("loss_function_name")
        weight_decay = training_params.get("weight_decay")

        optimizer_name = training_params.get("optimizer_name")
        momentum = training_params.get("momentum")
        if isinstance(training_params.get("learning_rate"), list):
            lr = training_params.get("learning_rate")[i]
        else:
            lr = training_params.get("learning_rate")
        print "learning rate: {}\n".format(lr)
        different_learning_rate = training_params.get("different_learning_rate")

        learning_rate_decay = training_params.get("learning_rate_decay")
        learning_rate_decay_step = training_params.get("learning_rate_decay_step")

        feature_part_decay_rate = training_params.get("feature_part_decay_rate")

        return loss_function_name, weight_decay, \
               optimizer_name, momentum, lr, different_learning_rate, \
               learning_rate_decay, learning_rate_decay_step, \
               feature_part_decay_rate

    def _get_criterion_from_loss_function(self, loss_function_name):
        return eval(loss_function_name)

    def _get_optimizer(self, optimizer_name, **kwargs):
        different_learning_rate = kwargs.get("different_learning_rate")
        learning_rate = float(kwargs.get("lr"))

        feature_part_decay_rate = float(kwargs.get("feature_part_decay_rate"))

        if different_learning_rate is None:
            params = self.model.parameters()
        else:
            # todo: change this when model changes
            params = [{"params": self.model.features.parameters(), "lr": learning_rate * feature_part_decay_rate},
                      {"params": self.model.classifier.parameters()}]
            for layer in [self.model.attribute_feature_fc,
                          self.model.attribute_subtlety_score_fc, self.model.attribute_internalStructure_score_fc,
                          self.model.attribute_calcification_score_fc, self.model.attribute_sphericity_score_fc,
                          self.model.attribute_margin_score_fc, self.model.attribute_lobulation_score_fc,
                          self.model.attribute_spiculation_score_fc, self.model.attribute_texture_score_fc]:
                params.append({"params": layer.parameters()})

        optimizer = eval(optimizer_name)
        if "Adam" in optimizer_name:
            optimizer = optimizer(params,
                                  lr=learning_rate,
                                  weight_decay=float(kwargs.get("weight_decay")))
        else:
            optimizer = optimizer(params,
                                  momentum=kwargs.get("momentum"),
                                  lr=learning_rate,
                                  weight_decay=float(kwargs.get("weight_decay")))

        # for param_group in optimizer.param_groups:
        #     print param_group.keys(), param_group["lr"], len(param_group["params"])

        return optimizer

    def _get_criterion_and_optimizer(self, i, training_params):
        loss_function_name, weight_decay, \
        optimizer_name, momentum, lr, different_learning_rate, \
        learning_rate_decay, learning_rate_decay_step, \
        feature_part_decay_rate = self._get_algorithm_info_from_dict(i, training_params)

        criterion = self._get_criterion_from_loss_function(loss_function_name)

        # todo: finish later
        optimizer = self._get_optimizer(optimizer_name,
                                        different_learning_rate=different_learning_rate,
                                        momentum=momentum,
                                        lr=lr,
                                        weight_decay=weight_decay,
                                        feature_part_decay_rate=feature_part_decay_rate)

        return criterion, optimizer, learning_rate_decay, learning_rate_decay_step

    # step 4(1): meters for measure network
    def _get_network_measure_tools(self, training_params):
        loss_meter = meter.AverageValueMeter()
        confusion_matrix = meter.ConfusionMeter(training_params.get("classes_num"))
        previous_loss = 1e100

        return loss_meter, confusion_matrix, previous_loss

    # step 4(2): other training info
    def _get_other_training_info_from_dict(self, training_params):
        max_epoch = training_params.get("max_epoch")
        standard_accuracy = training_params.get("standard_accuracy")
        save_name = training_params.get("save_name")
        save_frequency = training_params.get("save_frequency")
        print_freq = training_params.get("print_freq")

        batch_size = training_params.get("batch_size")
        gpu_num = training_params.get("gpu_num")

        model_dir = training_params.get("model_dir")

        return max_epoch, standard_accuracy, save_name, save_frequency, print_freq, batch_size, gpu_num, model_dir

    # ~~~~~~~~~~ TESING PROCESS TOOLS ~~~~~~~~~~
    def _get_output_related_info_for_validation(self, training_params):
        model_dir = training_params.get("model_dir")
        ith_folder = training_params.get("ith_folder")
        ith_epoch = training_params.get("ith_epoch")

        image_dir = training_params.get("multi_thread_data_dir")

        output_data_path = training_params.get("output_data_path")
        result_file = training_params.get("result_file")
        retrain_sample_file = training_params.get("retrain_sample_file")
        all_sample_file = training_params.get("all_sample_file")

        model_number = model_dir.strip().strip("/").split("/")[-1]
        result_dir = os.path.join(output_data_path, "result_analysis", model_number, "val", str(ith_folder))
        if not os.path.exists(result_dir) or not os.path.isdir(result_dir):
            os.makedirs(result_dir)

        return model_dir, ith_folder, ith_epoch, \
               image_dir, \
               output_data_path, result_file, retrain_sample_file, all_sample_file, \
               model_number, result_dir

    # ~~~~~~~~~~ TRAINING PROCESS ~~~~~~~~~~
    def training_transfer_data_to_gpu(self, image, gt_label, gt_attribute_score, actually_needed_gpu):
        input_cuda = self._transfer_data_to_gpu(image.float(),
                                                visiable_gpu_device_num=actually_needed_gpu)

        gt_label_cuda = self._transfer_data_to_gpu(gt_label.long(),
                                                   visiable_gpu_device_num=actually_needed_gpu)

        gt_attribute_score_cuda = []
        for item in gt_attribute_score:
            gt_attribute_score_cuda.append(self._transfer_data_to_gpu(
                item.float(), visiable_gpu_device_num=actually_needed_gpu))

        return input_cuda, gt_label_cuda, gt_attribute_score_cuda

    def trainging_get_loss(self,
                           output_score_1,
                           output_attribute_score_1,
                           gt_score_1,
                           gt_attribute_score_1):
        # basic loss
        xcentloss_func_1 = nn.CrossEntropyLoss()
        xcentloss_1 = xcentloss_func_1(output_score_1, gt_score_1)

        ranking_loss_sum = 0
        half_size_of_output_score = output_score_1.size()[0] / 2
        for i in range(half_size_of_output_score):
            tmp_output_1 = output_score_1[i]
            tmp_output_2 = output_score_1[i + half_size_of_output_score]
            tmp_gt_score_1 = gt_score_1[i]
            tmp_gt_score_2 = gt_score_1[i + half_size_of_output_score]

            if tmp_gt_score_1.data[0] > tmp_gt_score_2.data[0]:
                # print ">", gt_score_1, gt_score_2
                rankingloss_func = nn.MarginRankingLoss(margin=0.1)
                target = t.ones(1) * -1
                ranking_loss_sum += rankingloss_func(tmp_output_1, tmp_output_2, Variable(target.cuda()))
            elif tmp_gt_score_1.data[0] < tmp_gt_score_2.data[0]:
                # print "<", gt_score_1, gt_score_2
                rankingloss_func = nn.MarginRankingLoss(margin=0.1)
                target = t.ones(1)
                ranking_loss_sum += rankingloss_func(tmp_output_1, tmp_output_2, Variable(target.cuda()))
            else:
                continue

        ranking_loss = ranking_loss_sum / half_size_of_output_score

        # attribute loss
        attribute_mseloss_func_1 = nn.MSELoss()
        attribute_mseloss_1 = attribute_mseloss_func_1(output_attribute_score_1[0], gt_attribute_score_1[0])
        attribute_mseloss_sum = attribute_mseloss_1
        for i in range(1, len(output_attribute_score_1)):
            attribute_mseloss_func = nn.MSELoss()
            attribute_mseloss = attribute_mseloss_func(output_attribute_score_1[i], gt_attribute_score_1[i])
            if i in [5, 6]:
                attribute_mseloss_sum += (2 * attribute_mseloss)
            else:
                attribute_mseloss_sum += attribute_mseloss
        attribute_mseloss_mean = attribute_mseloss_sum

        return xcentloss_1, ranking_loss, attribute_mseloss_mean

    def training_get_accuracy(self, confusion_matrix, classes_num):
        cm_value = confusion_matrix.value()

        correct_sum = 0.0
        for i in range(classes_num):
            correct_sum += cm_value[i][i]
        if cm_value.sum() != 0:
            accuracy = 1.0 * correct_sum / cm_value.sum()
        else:
            accuracy = 0.0

        return accuracy

    def trainging_get_actually_needed_gpu(self, visiable_gpu_device_num, image):
        actually_needed_gpu = range(image.size()[0] / 2) \
            if len(visiable_gpu_device_num) > (image.size()[0] / 2) else visiable_gpu_device_num

        return actually_needed_gpu

    def train(self, training_params):
        folder_num, folder_list = self._get_folders_info_from_dict(training_params)
        for i in folder_list:
            print "Folder ~~~~~~~~~~~~~~~~~~~~~~~~", i

            writer, json_filepath = self._generate_log_dir_and_writer(training_params, i)

            # step 1: get dataloader
            train_dataloader, val_dataloader, length_of_train_data = \
                self._get_train_and_validation_dataloader(training_params, folder_num, i)

            # step 2: configure model and map to gpu
            self._set_up_model(training_params, i)

            visiable_gpu_device_num = self._config_gpu_info(training_params)
            print "\nVisiable gpu number:{}\n".format(t.cuda.device_count())
            self._transfer_data_to_gpu(self.model, visiable_gpu_device_num=visiable_gpu_device_num)

            # step 3: criterion and optimizer
            criterion, optimizer, learning_rate_decay, learning_rate_decay_step = \
                self._get_criterion_and_optimizer(i, training_params)

            # step 4: meters for measure network and other training info
            loss_meter, confusion_matrix, _ = self._get_network_measure_tools(training_params)
            max_epoch, standard_accuracy, save_name, save_frequency, print_freq, batch_size, gpu_num, model_dir = \
                self._get_other_training_info_from_dict(training_params)

            # step 5: train
            self.model.train()

            # todo: (note) TRAIN NOW
            highest_accuracy = copy.deepcopy(standard_accuracy)
            for epoch in range(max_epoch):
                loss_meter.reset()
                confusion_matrix.reset()

                for ii, (image, gt_score, gt_label,
                         gt_subtlety_score, gt_internalStructure_score,
                         gt_calcification_score, gt_sphericity_score,
                         gt_margin_score, gt_lobulation_score,
                         gt_spiculation_score, gt_texture_score,
                         _, _) in enumerate(train_dataloader):
                    # step A: get input and ground_truth_data
                    actually_needed_gpu = self.trainging_get_actually_needed_gpu(visiable_gpu_device_num, image)

                    gt_attribute_score = [gt_subtlety_score, gt_internalStructure_score,
                                          gt_calcification_score, gt_sphericity_score,
                                          gt_margin_score, gt_lobulation_score,
                                          gt_spiculation_score, gt_texture_score]

                    input_cuda, gt_label_cuda, gt_attribute_score_cuda = \
                        self.training_transfer_data_to_gpu(image, gt_label, gt_attribute_score, actually_needed_gpu)

                    # step B: IMPORTANT clear grads
                    optimizer.zero_grad()
                    # if ii == 1:
                    #     break

                    # step C: get output_data
                    # todo: put data to network
                    output_score_cuda, \
                    cat_subtlety_score, cat_internalStructure_score, cat_calcification_score, cat_sphericity_score, \
                    cat_margin_score, cat_lobulation_score, cat_spiculation_score, cat_texture_score = \
                        nn.parallel.data_parallel(self.model, input_cuda, device_ids=actually_needed_gpu)

                    cat_subtlety_score = t.nn.functional.sigmoid(cat_subtlety_score)
                    cat_internalStructure_score = t.nn.functional.sigmoid(cat_internalStructure_score)
                    cat_calcification_score = t.nn.functional.sigmoid(cat_calcification_score)
                    cat_sphericity_score = t.nn.functional.sigmoid(cat_sphericity_score)
                    cat_margin_score = t.nn.functional.sigmoid(cat_margin_score)
                    cat_lobulation_score = t.nn.functional.sigmoid(cat_lobulation_score)
                    cat_spiculation_score = t.nn.functional.sigmoid(cat_spiculation_score)
                    cat_texture_score = t.nn.functional.sigmoid(cat_texture_score)

                    output_attribute_score = [cat_subtlety_score, cat_internalStructure_score,
                                              cat_calcification_score, cat_sphericity_score,
                                              cat_margin_score, cat_lobulation_score,
                                              cat_spiculation_score, cat_texture_score]

                    # step D: get score and update weights
                    loss1, rankingloss, loss2 = self.trainging_get_loss(output_score_cuda,
                                                                        output_attribute_score,
                                                                        gt_label_cuda,
                                                                        gt_attribute_score_cuda)

                    loss = 0.4 * loss1 + 0.2 * rankingloss + 0.4 * loss2
                    loss.backward()
                    optimizer.step()

                    print "Epoch {}, Step {}, " \
                          "Losses are: {:.5f}, {:.5f}, {:.5f}, " \
                          "Total loss is {:.5f}\t\t\t\t\t Sample Num: {}".format(epoch, ii,
                                                                                 float(loss1.data[0]),
                                                                                 float(rankingloss) if isinstance(
                                                                                     rankingloss, long) else float(
                                                                                     rankingloss.data[0]),
                                                                                 float(loss2.data[0]),
                                                                                 float(loss.data[0]),
                                                                                 image.shape[0])

                    # meters update and visualize
                    loss_meter.add(loss.data[0])
                    # confusion_matrix.add(score.max(dim=1)[1].data, target.data)

                    if ii % training_params.get("print_freq") == training_params.get("print_freq") - 1:
                        training_n_iter = epoch * (int(math.ceil(1.0 * length_of_train_data /
                                                                 (batch_size * gpu_num)))) + ii + 1
                        writer.add_scalars("data/loss", {"train": float(loss.data[0])}, training_n_iter)

                # step E: get score and update weights validation and visualization
                training_params["ith_folder"] = str(i)
                training_params["ith_epoch"] = str(epoch + 1)
                val_n_iter = (epoch + 1) * (int(math.ceil(1.0 * length_of_train_data) / (batch_size * gpu_num)))
                # todo: (note) validation after each epoch
                val_cm, val_accuracy, val_loss = self.val(val_dataloader, training_params, criterion)

                print "val_loss", val_loss, 'val_accuracy', val_accuracy
                print "auc\n", val_cm.value()[0], "\n", val_cm.value()[1]
                writer.add_scalars("data/loss", {"validation": float(val_loss)}, val_n_iter)
                writer.add_scalars("data/accuracy", {"validation": float(val_accuracy)}, val_n_iter)

                if (epoch + 1) % learning_rate_decay_step == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * float(learning_rate_decay)

                if int(epoch + 1) % save_frequency == 0:
                    new_model_dir = copy.deepcopy(model_dir)
                    new_model_dir = os.path.join(new_model_dir, str(i))
                    if not os.path.exists(new_model_dir):
                        os.makedirs(new_model_dir)

                    prefix = os.path.join(new_model_dir,
                                          "{}_{}.pth".format(save_name, epoch+1))
                    name = prefix
                    t.save(self.model.state_dict(), name)

                writer.export_scalars_to_json(json_filepath)

    # ~~~~~~~~~~ TESTING PROCESS ~~~~~~~~~~
    def testing_transfer_data_to_gpu(self, image, gt_label, gt_attribute_score, actually_needed_gpu):
        input_cuda = Variable(image.float(), volatile=True).cuda()

        gt_label_cuda = self._transfer_data_to_gpu(gt_label.long(),
                                                   visiable_gpu_device_num=actually_needed_gpu)

        gt_attribute_score_cuda = []
        for item in gt_attribute_score:
            gt_attribute_score_cuda.append(self._transfer_data_to_gpu(
                item.float(), visiable_gpu_device_num=actually_needed_gpu))

        return input_cuda, gt_label_cuda, gt_attribute_score_cuda

    def testing_get_p_val(self, output_score, gt_attribute_score):
        new_list = zip(output_score, gt_attribute_score)
        new_list = sorted(new_list, key=lambda x: x[1])
        _, p_val = spearmanr(list(zip(*new_list)[1]), list(zip(*new_list)[0]))

        if np.isnan(p_val):
            p_val = np.float64(0)
        return p_val

    # todo: alter these funcs
    def val(self, dataloader, training_params, criterion):
        # config gpu
        visiable_gpu_device_num = self._config_gpu_info(training_params)

        model_dir, ith_folder, ith_epoch, \
        image_dir, \
        output_data_path, result_file, retrain_sample_file, all_sample_file, \
        model_number, result_dir = self._get_output_related_info_for_validation(training_params)

        self.model.eval()

        loss_list_1, rankingloss_list, loss_list_2, results = [], [], [], []
        confusion_matrix = meter.ConfusionMeter(training_params.get("classes_num"))
        # all_nodule_list = []

        for ii, (image, gt_score, gt_label,
                 gt_subtlety_score, gt_internalStructure_score,
                 gt_calcification_score, gt_sphericity_score,
                 gt_margin_score, gt_lobulation_score,
                 gt_spiculation_score, gt_texture_score,
                 other_info_name, other_info) in enumerate(dataloader):
            # if ii == 3:
            #     break

            actually_needed_gpu = self.trainging_get_actually_needed_gpu(visiable_gpu_device_num, image)

            gt_attribute_score = [gt_subtlety_score, gt_internalStructure_score,
                                  gt_calcification_score, gt_sphericity_score,
                                  gt_margin_score, gt_lobulation_score,
                                  gt_spiculation_score, gt_texture_score]

            input_cuda, gt_label_cuda, gt_attribute_score_cuda = \
                self.testing_transfer_data_to_gpu(image, gt_label, gt_attribute_score, actually_needed_gpu)

            # step C: get output_data
            # todo: put data to network
            output_score_cuda, \
            cat_subtlety_score, cat_internalStructure_score, cat_calcification_score, cat_sphericity_score, \
            cat_margin_score, cat_lobulation_score, cat_spiculation_score, cat_texture_score = \
                nn.parallel.data_parallel(self.model, input_cuda, device_ids=actually_needed_gpu)

            cat_subtlety_score = t.nn.functional.sigmoid(cat_subtlety_score)
            cat_internalStructure_score = t.nn.functional.sigmoid(cat_internalStructure_score)
            cat_calcification_score = t.nn.functional.sigmoid(cat_calcification_score)
            cat_sphericity_score = t.nn.functional.sigmoid(cat_sphericity_score)
            cat_margin_score = t.nn.functional.sigmoid(cat_margin_score)
            cat_lobulation_score = t.nn.functional.sigmoid(cat_lobulation_score)
            cat_spiculation_score = t.nn.functional.sigmoid(cat_spiculation_score)
            cat_texture_score = t.nn.functional.sigmoid(cat_texture_score)

            output_attribute_score = [cat_subtlety_score, cat_internalStructure_score,
                                      cat_calcification_score, cat_sphericity_score,
                                      cat_margin_score, cat_lobulation_score,
                                      cat_spiculation_score, cat_texture_score]


            # test loss
            loss1, rankingloss, loss2 = self.trainging_get_loss(output_score_cuda,
                                                                output_attribute_score,
                                                                gt_label_cuda,
                                                                gt_attribute_score_cuda)

            if isinstance(loss_list_1, float):
                loss_list_1.append(loss1)
            else:
                loss_list_1.append(float(loss1.data[0]))

            if isinstance(rankingloss, float):
                rankingloss_list.append(rankingloss)
            else:
                rankingloss_list.append(float(rankingloss) if isinstance(rankingloss, long) else float(rankingloss.data[0]))

            if isinstance(loss_list_2, float):
                loss_list_2.append(loss2)
            else:
                loss_list_2.append(float(loss2.data[0]))

            confusion_matrix.add(output_score_cuda.max(dim=1)[1].data, gt_label)

            accuracy_0 = self.testing_get_p_val(np.squeeze(cat_subtlety_score.cpu().data.numpy(), axis=1).tolist(),
                                                gt_subtlety_score.cpu().numpy().tolist())
            accuracy_1 = self.testing_get_p_val(np.squeeze(cat_internalStructure_score.cpu().data.numpy(), axis=1).tolist(),
                                                gt_internalStructure_score.cpu().numpy().tolist())
            accuracy_2 = self.testing_get_p_val(np.squeeze(cat_calcification_score.cpu().data.numpy(), axis=1).tolist(),
                                                gt_calcification_score.cpu().numpy().tolist())
            accuracy_3 = self.testing_get_p_val(np.squeeze(cat_sphericity_score.cpu().data.numpy(), axis=1).tolist(),
                                                gt_sphericity_score.cpu().numpy().tolist())
            accuracy_4 = self.testing_get_p_val(np.squeeze(cat_margin_score.cpu().data.numpy(), axis=1).tolist(),
                                                gt_margin_score.cpu().numpy().tolist())
            accuracy_5 = self.testing_get_p_val(np.squeeze(cat_lobulation_score.cpu().data.numpy(), axis=1).tolist(),
                                                gt_lobulation_score.cpu().numpy().tolist())
            accuracy_6 = self.testing_get_p_val(np.squeeze(cat_spiculation_score.cpu().data.numpy(), axis=1).tolist(),
                                                gt_spiculation_score.cpu().numpy().tolist())
            accuracy_7 = self.testing_get_p_val(np.squeeze(cat_texture_score.cpu().data.numpy(), axis=1).tolist(),
                                                gt_texture_score.cpu().numpy().tolist())

        self.model.train()

        # Accuracy 2
        mean_loss_1 = sum(loss_list_1) / float(len(loss_list_1))
        mean_rankingloss = sum(rankingloss_list) / float(len(rankingloss_list))
        mean_loss_2 = sum(loss_list_2) / float(len(loss_list_2))
        print "Three val losses are {:.5f}, {:.5f}, {:.5f}".format(float(mean_loss_1),
                                                                   float(mean_rankingloss),
                                                                   float(mean_loss_2))
        accuracy = self.training_get_accuracy(confusion_matrix, training_params.get("classes_num"))
        print "attribute p_vals are {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}".format(
            accuracy_0, accuracy_1, accuracy_2, accuracy_3, accuracy_4, accuracy_5, accuracy_6, accuracy_7)

        return confusion_matrix, accuracy, 0.4 * mean_loss_1 + 0.2 * mean_rankingloss + 0.4 * mean_loss_2
        # return confusion_matrix, accuracy, mean_loss_1

    # ~~~~~~~~~~ TESTING PROCESS ~~~~~~~~~~
    def test(self, testing_params):
        # _mean_then_sofemax
        model_accuracy = []
        k_folder = testing_params["k_folder"]
        for i in reversed(range(k_folder)):
            print i, "~~~~~~~~~~~~~~~~~~~~~~~~"

            # gpu config
            gpu_num = testing_params.get("gpu_num")
            gpu_device_num = testing_params.get("gpu_device_num")
            if gpu_device_num is not None and len(gpu_device_num) == gpu_num:
                visiable_gpu_device_num = range(gpu_num)
            else:
                print "Gpu num is not the same as len of gpu_device_num."
                visiable_gpu_device_num = range(gpu_num)
            print "Visiable gpu number:", t.cuda.device_count(), ", numbers: ", gpu_device_num

            # model --> gpu
            self._transfer_data_to_gpu(self.model, visiable_gpu_device_num=[0])

            # set output config
            model_dir = testing_params.get("pretrained_model_path")
            image_dir = testing_params.get("multi_thread_data_dir")

            output_data_path = testing_params.get("output_data_path")
            result_file = testing_params.get("result_file")
            retrain_sample_file = testing_params.get("retrain_sample_file")
            all_sample_file = testing_params.get("all_sample_file")

            train_or_test = "train" if testing_params.get("flag_training_dataset") else "test"
            model_number = model_dir.strip().strip("/").split("/")[-1]
            result_dir = os.path.join(output_data_path, "result_analysis", model_number, train_or_test, str(i))
            if not os.path.exists(result_dir) or not os.path.isdir(result_dir):
                os.makedirs(result_dir)
            with open(os.path.join(result_dir, result_file), 'w') as f:
                f.write(image_dir)
                f.write("\n")
            # print "Image Dir:", image_dir

            data_pattern = testing_params.get("data_pattern")
            annotation_file_path = testing_params.get("annotation_file_path")
            flag_training_dataset = testing_params.get("flag_training_dataset")
            flag_testing_dataset = testing_params.get("flag_testing_dataset")
            test_data = LungNodule(i, k_folder, image_dir, data_pattern, annotation_file_path,
                                   is_training=flag_training_dataset, is_testing=flag_testing_dataset)
            test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False,
                                         num_workers=testing_params.get("thread_num"))

            # run each model
            model_path_list = []
            if os.path.isdir(model_dir):
                for root, dirs, files in os.walk(model_dir):
                    for filename in files:
                        if filename.strip().endswith(".pth"):
                            model_path_list.append(os.path.join(root, filename))
            else:
                model_path_list = [os.path.abspath(model_dir)]

            if len(model_path_list) == 0:
                continue

            highest_accuracy = {"score": 0.0, "model": None, "auc": None}
            for model_path in model_path_list:
                # print model_path
                if "/{}/".format(str(i)) not in model_path:
                    continue

                print "~~~~~~~~", model_path
                self.model.load_state_dict(t.load(model_path))

                self.model.eval()

                confusion_matrix = meter.ConfusionMeter(testing_params.get("classes_num"))

                results = []
                accuracy_num, total_num = 0.0, 0
                not_good_enough_list, all_nodule_list = [], []

                for ii, (data, gt_score, ground_truth,
                         gt_subtlety_score, gt_internalStructure_score,
                         gt_calcification_score, gt_sphericity_score,
                         gt_margin_score, gt_lobulation_score,
                         gt_spiculation_score, gt_texture_score,
                         other_info_name, other_info) in enumerate(test_dataloader):
                    other_info_name_2 = copy.deepcopy(other_info_name)
                    other_info_2 = copy.deepcopy(other_info)

                    colume_name = list(zip(*other_info_name)[0])
                    other_info_pd = pd.DataFrame(zip(*other_info), columns=colume_name)

                    if data.size()[0] != 1:
                        print "Error."

                    # print "len is", int(math.ceil(1.0 * data.size()[1] / gpu_num))
                    actually_needed_gpu = self.trainging_get_actually_needed_gpu(visiable_gpu_device_num, data)

                    gt_attribute_score = [gt_subtlety_score, gt_internalStructure_score,
                                          gt_calcification_score, gt_sphericity_score,
                                          gt_margin_score, gt_lobulation_score,
                                          gt_spiculation_score, gt_texture_score]

                    input_cuda, gt_label_cuda, gt_attribute_score_cuda = \
                        self.testing_transfer_data_to_gpu(data, ground_truth, gt_attribute_score, [0])

                    # step C: get output_data
                    # todo: put data to network
                    output_score_cuda, \
                    cat_subtlety_score, cat_internalStructure_score,\
                    cat_calcification_score, cat_sphericity_score, \
                    cat_margin_score, cat_lobulation_score, \
                    cat_spiculation_score, cat_texture_score = self.model(input_cuda)

                    output_attribute_score = [cat_subtlety_score, cat_internalStructure_score,
                                              cat_calcification_score, cat_sphericity_score,
                                              cat_margin_score, cat_lobulation_score,
                                              cat_spiculation_score, cat_texture_score]

                    total_score = t.unsqueeze(output_score_cuda.mean(dim=0), 0)
                    cat_subtlety_score = t.nn.functional.sigmoid(cat_subtlety_score)
                    cat_internalStructure_score = t.nn.functional.sigmoid(cat_internalStructure_score)
                    cat_calcification_score = t.nn.functional.sigmoid(cat_calcification_score)
                    cat_sphericity_score = t.nn.functional.sigmoid(cat_sphericity_score)
                    cat_margin_score = t.nn.functional.sigmoid(cat_margin_score)
                    cat_lobulation_score = t.nn.functional.sigmoid(cat_lobulation_score)
                    cat_spiculation_score = t.nn.functional.sigmoid(cat_spiculation_score)
                    cat_texture_score = t.nn.functional.sigmoid(cat_texture_score)

                    total_probability = t.nn.functional.softmax(total_score)

                    label = total_probability.max(dim=1)[1].data.tolist()

                    # HARD EXAMPLE MINING
                    probability = total_probability.data.tolist()
                    if ground_truth.numpy()[0] == 1 and probability[0][1] < 0.50:
                        not_good_enough_list.append(other_info_pd.iloc[0])
                        # print "Label ~~~~~~~~~~~~~", label
                        # print type(other_info_pd.iloc[0])
                        # print other_info_pd.iloc[0]
                        # print "\n"
                        # print probability[0], " and ", path.numpy()[0]
                    elif ground_truth.numpy()[0] == 0 and probability[0][0] < 0.50:
                        not_good_enough_list.append(other_info_pd.iloc[0])
                        # print "Label ~~~~~~~~~~~~~", label
                        # print type(other_info_pd.iloc[0])
                        # print other_info_pd.iloc[0]
                        # print "\n"
                        # print probability[0], " and ", path.numpy()[0]

                    colume_name_2 = list(zip(*other_info_name_2)[0])

                    colume_name_2.append("probability")
                    other_info_2.append((probability[0],))

                    colume_name_2.append("subtlety_average")
                    other_info_2.append((gt_subtlety_score.tolist()[0],))
                    colume_name_2.append("internalStructure_average")
                    other_info_2.append((gt_internalStructure_score.tolist()[0],))
                    colume_name_2.append("calcification_average")
                    other_info_2.append((gt_calcification_score.tolist()[0],))
                    colume_name_2.append("sphericity_average")
                    other_info_2.append((gt_sphericity_score.tolist()[0],))
                    colume_name_2.append("margin_average")
                    other_info_2.append((gt_margin_score.tolist()[0],))
                    colume_name_2.append("lobulation_average")
                    other_info_2.append((gt_lobulation_score.tolist()[0],))
                    colume_name_2.append("spiculation_average")
                    other_info_2.append((gt_spiculation_score.tolist()[0],))
                    colume_name_2.append("texture_average")
                    other_info_2.append((gt_texture_score.tolist()[0],))

                    colume_name_2.append("subtlety_output_score")
                    other_info_2.append((cat_subtlety_score.data.tolist()[0][0],))
                    colume_name_2.append("internalStructure_output_score")
                    other_info_2.append((cat_internalStructure_score.data.tolist()[0][0],))
                    colume_name_2.append("calcification_output_score")
                    other_info_2.append((cat_calcification_score.data.tolist()[0][0],))
                    colume_name_2.append("sphericity_output_score")
                    other_info_2.append((cat_sphericity_score.data.tolist()[0][0],))
                    colume_name_2.append("margin_output_score")
                    other_info_2.append((cat_margin_score.data.tolist()[0][0],))
                    colume_name_2.append("lobulation_output_score")
                    other_info_2.append((cat_lobulation_score.data.tolist()[0][0],))
                    colume_name_2.append("spiculation_output_score")
                    other_info_2.append((cat_spiculation_score.data.tolist()[0][0],))
                    colume_name_2.append("texture_output_score")
                    other_info_2.append((cat_texture_score.data.tolist()[0][0],))

                    other_info_pd_2 = pd.DataFrame(zip(*other_info_2), columns=colume_name_2)
                    all_nodule_list.append(other_info_pd_2.iloc[0])

                    batch_results = [(ground_truth_, label_, probability_) for ground_truth_, label_, probability_
                                     in
                                     zip(ground_truth, label, probability)]

                    accuracy_num += np.sum(np.equal(ground_truth.numpy(), np.array(label)).astype(int))
                    total_num += data.size()[0]

                    results += batch_results
                    confusion_matrix.add(total_score.max(dim=1)[1].data, ground_truth)

                if highest_accuracy["score"] < 1.0 * accuracy_num / total_num:
                    highest_accuracy = {"score": 1.0 * accuracy_num / total_num,
                                        "model": model_path,
                                        "auc": str(confusion_matrix.value())}

                train_or_test = "train" if flag_training_dataset else "test"
                model_number_for_hard_example = model_dir.strip().strip("/").split("/")[-1] + "/" \
                                                + train_or_test + "/" + str(i) + "/" \
                                                + "_".join(model_path.split("/")[-1].split("_")[:2]) \
                                                + "_" + model_path.split("/")[-1].split("_")[-1].split(".")[0]

                # generate retrained sample csv file
                hard_example_filepath = os.path.join(output_data_path,
                                                     retrain_sample_file.format(model_number_for_hard_example))
                print "Hard_Example_Filepath:", hard_example_filepath
                print "Result_Analysis_Filepath:", os.path.join(result_dir, result_file)
                hard_example_dir = "/".join(hard_example_filepath.strip().split("/")[:-1])
                if not os.path.exists(hard_example_dir):
                    os.makedirs(hard_example_dir)
                if len(not_good_enough_list) > 0:
                    not_good_enough_pd = pd.concat(not_good_enough_list, axis=1, ignore_index=True).T
                    not_good_enough_pd.to_csv(hard_example_filepath, index=False)

                # generate all nodule info
                all_example_filepath = os.path.join(output_data_path,
                                                    all_sample_file.format(model_number_for_hard_example))
                all_example_dir = "/".join(all_example_filepath.strip().split("/")[:-1])
                if not os.path.exists(all_example_dir):
                    os.makedirs(all_example_dir)
                if len(all_nodule_list) > 0:
                    all_nodule_pd = pd.concat(all_nodule_list, axis=1, ignore_index=True).T
                    all_nodule_pd.to_csv(all_example_filepath, index=False)

                with open(os.path.join(result_dir, result_file), 'a') as f:
                    # output and save result
                    str_model_path = "Model: {}\n".format(model_path)
                    str_test_accuracy = "Test Accuracy = {} / {} = {}\n".format(
                        accuracy_num, total_num, 1.0 * accuracy_num / total_num)
                    str_auc = "AUC:\n {}\n".format(str(confusion_matrix.value()))
                    str_hem = "Length of not good enough examples: {}".format(str(len(not_good_enough_list)))

                    f.write(image_dir)
                    f.write("\n")

                    f.writelines(str_model_path)
                    f.writelines(str_test_accuracy)
                    f.writelines(str_auc)
                    f.writelines(str_hem)
                    f.writelines("\n\n")

                    str_model_path = "Model: {}".format(model_path)
                    str_test_accuracy = "Test Accuracy = {} / {} = {}".format(
                        accuracy_num, total_num, 1.0 * accuracy_num / total_num)
                    str_auc = "AUC:\n {}".format(str(confusion_matrix.value()))
                    str_hem = "Length of not good enough examples: {}".format(str(len(not_good_enough_list)))

                    print str_model_path
                    print str_test_accuracy
                    print str_auc
                    print str_hem
                    print "\n"

            print "Highest Accuracy's Model: {}\nAccuracy: {}\nAUC: {}".format(highest_accuracy["score"],
                                                                               highest_accuracy["model"],
                                                                               highest_accuracy["auc"])
            model_accuracy.append(highest_accuracy)

            with open(os.path.join(output_data_path,
                                   "result_analysis",
                                   model_number,
                                   train_or_test,
                                   "all_model_" + result_file), 'w') as f_all:
                for item in model_accuracy:
                    str_model_accuracy = "Model: {}\nHighest Accuracy: {}\nAUC:\n{}".format(item["model"],
                                                                                            item["score"],
                                                                                            item["auc"])

                    f_all.write(str_model_accuracy)
                    f_all.write("\n\n")
                    print str_model_accuracy
