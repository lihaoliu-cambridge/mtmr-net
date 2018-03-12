# !/usr/bin/env python
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from skimage import io, transform
from sklearn.utils import shuffle
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import multiprocessing as mp
import SimpleITK as sitk
import pandas as pd
import numpy as np
import torch as t
import time
import json

image_size = [256, 256]
half_image_size = [int(i / 2) for i in image_size]

fixed_origin = [0., 0., 0.]
fixed_spacing = [1., 1., 1.]
fixed_direction = [1., 0., 0., 0., 1., 0., 0., 0., 1.]


class LungNodule(Dataset):
    def __init__(self, k, k_folders,
                 image_dir, data_pattern="*.mhd", annotation_file_path=None,
                 transforms=None,
                 is_training=True, is_testing=False,
                 hard_example_mining_file_path=None, repeat_times=None):
        """
        step 0. get image file path list
        step 1. get annotation info and other info from csv file
        step 2. merge filepath_list and input_file_info, and clean merged_info to valid_input_info
        from step 0, 1, 2 ======> get the valid input info as self.valid_input_info

        step 3. get training or testing mode
        step 4. get transformation for training data

        :param image_dir:
        :param data_pattern:
        :param annotation_file_path:
        :param transforms:
        :param is_training:
        :param is_testing:
        """
        # step 1. get image file path list
        image_dir = image_dir[:-1] if image_dir[-1] == "/" else image_dir
        data_pattern = data_pattern[1:] if data_pattern[0] == "/" else data_pattern
        # filepath_list = glob(image_dir + "/*/" + data_pattern)
        filepath_list = glob(image_dir + "/" + data_pattern)
        if len(filepath_list) == 0:
            raise ValueError("Can't find any file in the dir: {}".format(image_dir))

        # step 2. get annotation info and other info from csv file
        annotation_file_list = annotation_file_path.strip().split("/")
        if is_training:
            annotation_file_path = "/".join(annotation_file_list[:-1]) + "/train.csv"
        else:
            annotation_file_path = "/".join(annotation_file_list[:-1]) + "/test.csv"
        print annotation_file_path

        input_file_info = pd.read_csv(annotation_file_path)

        input_file_info = input_file_info[input_file_info["diameter_mm"] >= 3.0]

        # step 3. merge filepath_list and input_file_info to valid_input_info
        input_file_info["preprocessed_filename"] = input_file_info[["seriesuid", "coordX", "coordY", "coordZ"]].apply(
            lambda x: "_".join(str(item) for item in x), axis=1)

        input_file_info["label"] = input_file_info["malignancy"].map(
            lambda malignancy_score: self._get_label_from_malignancy_score_2(malignancy_score, 3, 3))

        self.keep_colume_list = list(input_file_info.columns.values)

        input_file_info = self._get_attributes_score(input_file_info)
        input_file_info = self._get_attributes_label(input_file_info)

        input_file_info = input_file_info.dropna()
        input_file_info.index = np.array(range(input_file_info.shape[0]))

        # step 3. get training, validating or testing mode
        if is_training and not is_testing:
            # training mode
            self.valid_input_info = input_file_info

            if hard_example_mining_file_path is not None:
                print "Note: training data with hard example mining"

                hard_example = pd.read_csv(hard_example_mining_file_path).dropna()
                hard_example = self._get_attributes_score(hard_example)
                hard_example = self._get_attributes_label(hard_example)
                hard_example = hard_example.dropna()
                for _ in range(repeat_times):
                    self.valid_input_info = self.valid_input_info.append(hard_example)

                self.valid_input_info.index = np.array(range(self.valid_input_info.shape[0]))
            else:
                print "Note: training data"

            if int(self.valid_input_info.shape[0]) % 2 != 0:
                self.valid_input_info = self.valid_input_info.append(self.valid_input_info.iloc[-1])
                print "Data number is not even."
        else:
            self.valid_input_info = input_file_info

        self.valid_input_info["filepath"] = self.valid_input_info["preprocessed_filename"].map(
            lambda incomplete_filename: self._get_filepath_from_filename(filepath_list, str(incomplete_filename)))

        if len(self.valid_input_info) == 0:
            raise ValueError("Empty Valid Info DataFrame.")

        for i in set(self.valid_input_info["label"].tolist()):
            print "Class {}, Number {}.".format(i,
                                                self.valid_input_info["label"][
                                                    self.valid_input_info["label"] == i
                                                ].shape[0])

        # step 4. get transformation for training data
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            if is_testing or not is_training:
                self.transforms = T.Compose([
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    @staticmethod
    def _get_filepath_from_filename(exist_filepath_list, incomplete_filename):
        """

        :param exist_filepath_list:
        :param incomplete_filename:
        :return:
        """
        for filepath in exist_filepath_list:
            if str(incomplete_filename) in filepath:
                return str(filepath)

    @staticmethod
    def _get_label_from_malignancy_score(malignacy_score, threshold_1=2.5, threshold_2=3.5):
        malignacy_score = json.loads(malignacy_score)

        if isinstance(malignacy_score, list) and len(malignacy_score) != 0:
            if 0 <= (sum(malignacy_score) / float(len(malignacy_score))) <= threshold_1:
                return 0
            elif threshold_2 <= (sum(malignacy_score) / float(len(malignacy_score))) <= 5:
                return 1
            else:
                # print malignacy_score
                return None
        else:
            # print threshold
            return None

    @staticmethod
    def _get_label_from_malignancy_score_2(malignacy_score, threshold_1=3, threshold_2=3):
        malignacy_score = json.loads(malignacy_score)
        # malignancy_score = eval(malignancy_score)

        if isinstance(malignacy_score, list) and len(malignacy_score) != 0:
            if 0 <= (sum(malignacy_score) / float(len(malignacy_score))) < threshold_1:
                return 0
            elif threshold_2 < (sum(malignacy_score) / float(len(malignacy_score))) <= 5:
                return 1
            else:
                # print malignacy_score
                return None
        else:
            # print threshold
            return None

    @staticmethod
    def _remove_many(malignacy_score, threshold_1=3, threshold_2=3):
        malignacy_score = json.loads(malignacy_score)

        if malignacy_score.count(3) > (len(malignacy_score) - malignacy_score.count(3)):
            # print malignacy_score
            return None
        else:
            if isinstance(malignacy_score, list) and len(malignacy_score) != 0:
                if 0 <= (sum(malignacy_score) / float(len(malignacy_score))) < threshold_1:
                    return 0
                elif threshold_2 < (sum(malignacy_score) / float(len(malignacy_score))) <= 5:
                    return 1
                else:
                    # print malignacy_score
                    return None
            else:
                # print threshold
                return None

    def _get_attributes_score(self, input_file_info):
        input_file_info["internalStructure_average"] = input_file_info["internalStructure"].map(
            lambda attribute_score: self._get_normalized_average_score_for_internal_structure(attribute_score))

        input_file_info["calcification_average"] = input_file_info["calcification"].map(
            lambda attribute_score: self._get_normalized_average_score_for_calcification(attribute_score))

        input_file_info["subtlety_average"] = input_file_info["subtlety"].map(
            lambda attribute_score: self._get_normalized_average_score(attribute_score))

        input_file_info["sphericity_average"] = input_file_info["sphericity"].map(
            lambda attribute_score: self._get_normalized_average_score(attribute_score))

        input_file_info["margin_average"] = input_file_info["margin"].map(
            lambda attribute_score: self._get_normalized_average_score(attribute_score))

        input_file_info["lobulation_average"] = input_file_info["lobulation"].map(
            lambda attribute_score: self._get_normalized_average_score(attribute_score))

        input_file_info["spiculation_average"] = input_file_info["spiculation"].map(
            lambda attribute_score: self._get_normalized_average_score(attribute_score))

        input_file_info["texture_average"] = input_file_info["texture"].map(
            lambda attribute_score: self._get_normalized_average_score(attribute_score))

        input_file_info["malignancy_average"] = input_file_info["malignancy"].map(
            lambda attribute_score: self._get_normalized_average_score(attribute_score))

        return input_file_info

    @staticmethod
    def _get_normalized_average_score(attribute_score):
        min_value = 1
        max_value = 5

        attribute_score = json.loads(attribute_score)

        if isinstance(attribute_score, list) and len(attribute_score) != 0:
            score_list = np.array(attribute_score)
            x = np.mean((1.0 * score_list - min_value) / (max_value - min_value))
            return x
        else:
            raise ValueError()

    @staticmethod
    def _get_normalized_average_score_for_internal_structure(attribute_score):
        min_value = 1
        max_value = 4

        attribute_score = json.loads(attribute_score)

        if isinstance(attribute_score, list) and len(attribute_score) != 0:
            score_list = np.array(attribute_score)
            x = np.mean((1.0 * score_list - min_value) / (max_value - min_value))
            return x
        else:
            raise ValueError()

    @staticmethod
    def _get_normalized_average_score_for_calcification(attribute_score):
        min_value = 1
        max_value = 6

        attribute_score = json.loads(attribute_score)

        if isinstance(attribute_score, list) and len(attribute_score) != 0:
            score_list = np.array(attribute_score)
            x = np.mean((1.0 * score_list - min_value) / (max_value - min_value))
            return x
        else:
            raise ValueError()

    def _get_attributes_label(self, input_file_info):
        input_file_info["internalStructure_label"] = input_file_info["internalStructure"].map(
            lambda attribute_label: self._get_normalized_average_label_for_internal_structure(attribute_label))

        input_file_info["calcification_label" ] = input_file_info["calcification"].map(
            lambda attribute_label: self._get_normalized_average_label_for_calcification(attribute_label))

        input_file_info["subtlety_label" ] = input_file_info["subtlety"].map(
            lambda attribute_label: self._get_normalized_average_label(attribute_label))

        input_file_info["sphericity_label" ] = input_file_info["sphericity"].map(
            lambda attribute_label: self._get_normalized_average_label(attribute_label))

        input_file_info["margin_label" ] = input_file_info["margin"].map(
            lambda attribute_label: self._get_normalized_average_label(attribute_label))

        input_file_info["lobulation_label" ] = input_file_info["lobulation"].map(
            lambda attribute_label: self._get_normalized_average_label(attribute_label))

        input_file_info["spiculation_label" ] = input_file_info["spiculation"].map(
            lambda attribute_label: self._get_normalized_average_label(attribute_label))

        input_file_info["texture_label" ] = input_file_info["texture"].map(
            lambda attribute_label: self._get_normalized_average_label(attribute_label))

        input_file_info["malignancy_label" ] = input_file_info["malignancy"].map(
            lambda attribute_label: self._get_normalized_average_label(attribute_label))

        return input_file_info

    @staticmethod
    def _get_normalized_average_label(attribute_label):
        attribute_label = json.loads(attribute_label)

        if isinstance(attribute_label, list) and len(attribute_label) != 0:
            if 1 <= (sum(attribute_label) / float(len(attribute_label))) <= 2:
                return 0
            elif 2 < (sum(attribute_label) / float(len(attribute_label))) <= 3:
                return 1
            elif 3 < (sum(attribute_label) / float(len(attribute_label))) <= 4:
                return 2
            elif 4 < (sum(attribute_label) / float(len(attribute_label))) <= 5:
                return 3
            else:
                return None
        else:
            # print threshold
            return ValueError

    @staticmethod
    def _get_normalized_average_label_for_internal_structure(attribute_label):
        attribute_label = json.loads(attribute_label)

        if isinstance(attribute_label, list) and len(attribute_label) != 0:
            if 1 <= (sum(attribute_label) / float(len(attribute_label))) <= 2:
                return 0
            elif 2 < (sum(attribute_label) / float(len(attribute_label))) <= 3:
                return 1
            elif 3 < (sum(attribute_label) / float(len(attribute_label))) <= 4:
                return 2
            else:
                return None
        else:
            # print threshold
            return ValueError

    @staticmethod
    def _get_normalized_average_label_for_calcification(attribute_label):
        attribute_label = json.loads(attribute_label)

        if isinstance(attribute_label, list) and len(attribute_label) != 0:
            if 1 <= (sum(attribute_label) / float(len(attribute_label))) <= 2:
                return 0
            elif 2 < (sum(attribute_label) / float(len(attribute_label))) <= 3:
                return 1
            elif 3 < (sum(attribute_label) / float(len(attribute_label))) <= 4:
                return 2
            elif 4 < (sum(attribute_label) / float(len(attribute_label))) <= 5:
                return 3
            elif 5 < (sum(attribute_label) / float(len(attribute_label))) <= 6:
                return 4
            else:
                return None
        else:
            # print threshold
            return ValueError

    @staticmethod
    def _dense_to_one_hot(labels_dense):
        """
        Convert class labels from scalars to one-hot vectors.
        :param labels_dense: numpy arrry, continuous and start from 0.
        :return: list
        """
        labels_dense = labels_dense.astype(np.int32)
        num_classes = len(set(labels_dense.tolist()))

        if num_classes <= 1:
            raise ValueError("There are less than 2 classes.")

        if labels_dense.min() != 0 or num_classes != labels_dense.max() - labels_dense.min() + 1:
            raise ValueError("Labels are not continous. Or labels are not from 0 to {}".format(labels_dense.max()))

        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

    @staticmethod
    def make_class_num_even(input_file_info):
        max_num = 0
        for class_num in set(input_file_info["label"].tolist()):
            max_num = max(max_num, input_file_info["label"][input_file_info["label"] == class_num].count())
        # print "max", max_num

        for class_num in set(input_file_info["label"].tolist()):
            input_file_info_for_this_class = input_file_info[input_file_info["label"] == class_num]
            input_file_info_size = input_file_info["label"][input_file_info["label"] == class_num].count()

            rest_num = max_num - input_file_info_size

            while rest_num >= input_file_info_size:
                input_file_info = input_file_info.append(input_file_info_for_this_class)

                input_file_info_size = input_file_info["label"][input_file_info["label"] == class_num].count()
                rest_num = max_num - input_file_info_size

            if rest_num != 0:
                input_file_info = input_file_info.append(
                    input_file_info[input_file_info["label"] == class_num].sample(rest_num))

        input_file_info = input_file_info.reset_index(drop=True)

        return input_file_info

    @staticmethod
    def make_class_num_even_for_malignancy(input_file_info):
        max_num = 0
        for class_num in set(input_file_info["label"].tolist()):
            max_num = max(max_num, input_file_info["label"][input_file_info["label"] == class_num].count())
        # print "max", max_num

        for class_num in set(input_file_info["label"].tolist()):
            input_file_info_for_this_class = input_file_info[input_file_info["label"] == class_num]
            input_file_info_size = input_file_info["label"][input_file_info["label"] == class_num].count()

            if max_num == input_file_info_size:
                continue

            rest_num = max_num

            while rest_num >= input_file_info_size:
                input_file_info = input_file_info.append(input_file_info_for_this_class)

                rest_num = rest_num - input_file_info_size
                input_file_info_size = input_file_info["label"][input_file_info["label"] == class_num].count()

            if rest_num != 0:
                input_file_info = input_file_info.append(
                    input_file_info[input_file_info["label"] == class_num].sample(rest_num))

        input_file_info = input_file_info.reset_index(drop=True)

        return input_file_info

    @staticmethod
    def _get_chuchk_data(input_file_info, idx, chuck_size_list, printing=False):
        if idx == len(chuck_size_list) - 1:
            ith_chuck_data = input_file_info[
                chuck_size_list[idx] <= input_file_info["malignancy_average"]][
                input_file_info["malignancy_average"] <= chuck_size_list[idx] + 1.0 / len(chuck_size_list)]
        else:
            ith_chuck_data = input_file_info[
                chuck_size_list[idx] <= input_file_info["malignancy_average"]][
                input_file_info["malignancy_average"] < chuck_size_list[idx] + 1.0 / len(chuck_size_list)]

        if printing:
            print "Number of {} - {}: Number {}.".format(chuck_size_list[idx],
                                                         chuck_size_list[idx] + 1.0 / len(chuck_size_list),
                                                         ith_chuck_data.shape[0])

        return ith_chuck_data

    def make_class_num_even_for_regression(self, input_file_info):
        # get even size number
        chuck_low_bound_list = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]

        chuck_size_list = []
        for idx, low_bound in enumerate(chuck_low_bound_list):
            ith_chuck_data = self._get_chuchk_data(input_file_info, idx, chuck_low_bound_list, True)
            ith_chuck_data_size = ith_chuck_data.shape[0]

            chuck_size_list.append(ith_chuck_data_size)

        max_size = max(chuck_size_list)
        mean_size = int(1.0 * sum(chuck_size_list) / len(chuck_size_list))
        best_chuck_size = int(max_size + mean_size) / 2
        # print max_size, mean_size, best_chuck_size

        # add more uneven data
        for idx, low_bound in enumerate(chuck_low_bound_list):
            ith_chuck_data = self._get_chuchk_data(input_file_info, idx, chuck_low_bound_list)
            ith_chuck_data_size = ith_chuck_data.shape[0]

            if ith_chuck_data.shape[0] <= 0 or best_chuck_size <= ith_chuck_data_size:
                continue

            total_chuck_time = best_chuck_size / ith_chuck_data_size - 1
            rest_num = best_chuck_size % ith_chuck_data_size
            while total_chuck_time:
                input_file_info = input_file_info.append(ith_chuck_data)
                total_chuck_time -= 1

            # todo:
            # print input_file_info.shape
            if rest_num != 0:
                input_file_info = input_file_info.append(ith_chuck_data.sample(rest_num))

            input_file_info = input_file_info.reset_index(drop=True)

        input_file_info = shuffle(input_file_info)

        return input_file_info

    def __len__(self):
        # print "Length of data: {}".format(self.valid_input_info.shape[0])
        return self.valid_input_info.shape[0]

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        one_input_info_1 = self.valid_input_info.iloc[index]

        # one_input_info_1
        image_filepath = one_input_info_1["filepath"]
        img_array = io.imread(image_filepath)
        if list(img_array.shape) != image_size:
            raise ValueError("Image's shape {} is not {}".format(str(img_array.shape), str(image_size)))

        img_array = np.stack([img_array, img_array, img_array], axis=2)
        pil_image = Image.fromarray(img_array)

        img_array = self.transforms(pil_image)

        return img_array, \
               one_input_info_1["malignancy_average"], int(one_input_info_1["label"]),\
               one_input_info_1["subtlety_average"], one_input_info_1["internalStructure_average"],\
               one_input_info_1["calcification_average"], one_input_info_1["sphericity_average"],\
               one_input_info_1["margin_average"], one_input_info_1["lobulation_average"],\
               one_input_info_1["spiculation_average"], one_input_info_1["texture_average"],\
               list(self.valid_input_info[self.keep_colume_list].columns.values),\
               one_input_info_1[self.keep_colume_list].tolist()
        # t.from_numpy(np.array(one_input_info_1[["subtlety_average",
        #                                         "internalStructure_average", "calcification_average",
        #                                         "sphericity_average", "margin_average", "lobulation_average",
        #                                         "spiculation_average", "texture_average"]].tolist())), \


class AddChannel(object):
    """

    Add channel dimension.
    Convert ndarrays in sample to Tensors.
    """
    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, image):
        if isinstance(image, list):
            image = np.stack(image, self.dim)
            # print image.shape
        else:
            image = np.expand_dims(image, self.dim)
        return image


def plot_image(img_array_1, img_array_2, img_array_3, output_image_filepath=None):
    plt.figure(figsize=(12, 4))
    plt.title(u"result")

    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.imshow(img_array_1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Original")
    plt.imshow(img_array_2, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Output")
    plt.imshow(img_array_3)
    plt.axis('off')

    if not output_image_filepath:
        plt.show()
    else:
        plt.savefig(output_image_filepath, dpi=100)


def main():
    # test load dataset with dataloader
    lung_nodule_dataset = LungNodule(# "../data/luna2016/preprocessed_data_for_pytorch/benign/256x256x256_3D/train",
        4, 5,
        "/home/lhliu/Onepiece/share/lhliu/big_data/data/datas/LIDC-IDRI/"
        "preprocessed_data_for_pytorch/benign_2d/new_OA_256x256x256_2D_2",
        "*.jpg",
        "../data/LIDC-IDRI/CSVFILES/train.csv",
        is_training=False,
        is_testing=True)
    # print len(lung_nodule_dataset)
    lung_nodule_dataloader = DataLoader(lung_nodule_dataset, batch_size=2, shuffle=False, num_workers=4)

    # test directly load dataset
    start_time = time.time()
    for i in range(len(lung_nodule_dataset)):
        image_1, m_score_1, attribute_score_1, label_1,  _, _, _, _, _, _, _, _, _, = \
            lung_nodule_dataset[i]
        # print image_1, label_1, attribute_score_1, image_2, label_2, attribute_score_2
        plot_image(image_1.numpy()[0], image_1.numpy()[1],image_1.numpy()[2], "")

        if i == 1:
            break
    end_time = time.time()
    print "Time is: {}".format(end_time - start_time)

    # test 4
    start_time = time.time()
    for i, (image_1, m_score_1, attribute_score_1, label_1, _, _, _, _, _, _, _,
            other_info_name, other_info_1) in enumerate(lung_nodule_dataloader):
        print image_1.shape
        colume_name = list(zip(*other_info_name)[0])
        print "colume_name", colume_name

        a = pd.DataFrame(zip(*other_info_1), columns=colume_name)
        print a.shape

        if i == 0:
            break
    end_time = time.time()
    print "Time is {}".format(end_time - start_time)


if __name__ == '__main__':
    main()
