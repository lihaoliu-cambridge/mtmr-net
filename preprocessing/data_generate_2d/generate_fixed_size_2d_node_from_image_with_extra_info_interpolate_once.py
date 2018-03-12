# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import defaultdict
from skimage import io, transform
from glob import glob
from math import ceil
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd
import numpy as np
from scipy.ndimage.interpolation import zoom
from scipy.misc import toimage
import pickle
import shutil
import json
import copy
import os

try:
    # long waits are not fun
    from tqdm import tqdm
except ImportError as e:
    print "TQDM does make much nicer wait bars..."
    tqdm = lambda x: x

__author__ = 'Liu Lihao'


original_data_path = "/home/lhliu/Onepiece/share/lhliu/big_data/data/datas/LIDC-IDRI/"
original_csvfiles_path = "/home/lhliu/Onepiece/share/lhliu/big_data/data/datas/LIDC-IDRI/"
processed_data_path_for_oa = "/home/lhliu/Onepiece/share/lhliu/big_data/data/datas/LIDC-IDRI/" \
                             "preprocessed_data_for_pytorch/benign_2d/new_OA_256x256x256_2D_2/"
processed_data_path_for_hs = "/home/lhliu/Onepiece/share/lhliu/big_data/data/datas/LIDC-IDRI/" \
                             "preprocessed_data_for_pytorch/benign_2d/new_HS_256x256x256_2D_2/"
processed_data_path_for_hvv = "/home/lhliu/Onepiece/share/lhliu/big_data/data/datas/LIDC-IDRI/" \
                              "preprocessed_data_for_pytorch/benign_2d/new_HVV_256x256x256_2D_2/"
file_list = glob(original_data_path + "Images_3/" + "*.mhd")

image_num = 100

first_size_x = 48
first_size_y = 48
first_size_z = 48
half_first_size_x = first_size_x / 2
half_first_size_y = first_size_y / 2
half_first_size_z = int(first_size_z / 2)
first_spacing_x = float(1.0)
first_spacing_y = float(1.0)
first_spacing_z = float(1.0)

final_size_x = 256
final_size_y = 256
final_size_z = 256
half_final_size_x = final_size_x / 2
half_final_size_y = final_size_y / 2
half_final_size_z = int(final_size_z / 2)

final_offset_x = 64
final_offset_y = 64
final_offset_z = 64
half_final_offset_x = final_offset_x / 2
half_final_offset_y = final_offset_x / 2
half_final_offset_z = final_offset_x / 2
final_spacing_x = float(1.0 * first_size_x / final_size_x)
final_spacing_y = float(1.0 * first_size_x / final_size_x)
final_spacing_z = float(1.0 * first_size_x / final_size_x)

fixed_origin = [0., 0., 0.]
fixed_spacing = [1., 1., 1.]
fixed_direction = [1., 0., 0., 0., 1., 0., 0., 0., 1.]


def get_info_from_csv_and_pkl(to_be_name_prefix):
    csv_info = pd.read_csv(os.path.join(original_csvfiles_path, "CSVFILES", "annotations_xml_voxel.csv"))
    csv_info["filepath"] = csv_info["seriesuid"].map(
        lambda incomplete_filename: get_complete_filepath_from_filename(file_list, str(incomplete_filename)))
    csv_info["main_key"] = csv_info.apply(
        lambda x: str(x["seriesuid"]) + "_" + str(x["coordX"]) + "_" + str(x["coordY"]), axis=1)
    csv_info["to_be_image_name"] = csv_info.apply(
        lambda x: to_be_name_prefix + str(x["seriesuid"]) + "_" + str(x["coordX"]) + "_" + str(x["coordY"]), axis=1)

    # f = open(os.path.join(original_csvfiles_path, "important_files", "LIDC_mask_all.pkl"))
    f = open("/home/lhliu/Onepiece/project/PythonProjects/bmdiagnosis_research/data/"
             "output_data/preprocessing_related/LIDC_mask_all.pkl")
    pkl_info = pickle.load(f)

    return csv_info, pkl_info


def get_complete_filepath_from_filename(exist_filepath_list, incomplete_filename):
    """

    :param exist_filepath_list:
    :param incomplete_filename:
    :return:
    """
    for filepath in exist_filepath_list:
        if str(incomplete_filename) in filepath:
            return str(filepath)


def load_image_zyx(filepath):
    itk_img = sitk.ReadImage(filepath)
    img_array = sitk.GetArrayFromImage(itk_img)
    origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(itk_img.GetSpacing())  # x,y,z  spacing of voxels in world coor. (mm)
    direction = np.array(itk_img.GetDirection())

    return img_array, origin, spacing, direction


def load_image_xyz(filepath):
    itk_img = sitk.ReadImage(filepath)
    img_array = sitk.GetArrayFromImage(itk_img)
    img_array = img_array.swapaxes(0, 2).astype(np.float32)
    origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(itk_img.GetSpacing())  # x,y,z  spacing of voxels in world coor. (mm)
    direction = np.array(itk_img.GetDirection())

    return img_array, origin, spacing, direction


def normalization(img):
    max_hu = 1200.
    min_hu = -400.
    mean_value = 0.33
    std = 0.33

    img[img > max_hu] = max_hu
    img[img < min_hu] = min_hu
    img_normalized = (img - min_hu) / (max_hu - min_hu)
    img_normalized = (img_normalized - mean_value) / std

    # for i in range(0, img_normalized.shape[2]):
    #     pass
    #     print img_normalized[0, 0, i]

    return img_normalized


def interpolation_without_transpose(img, new_shape, new_spacing, old_origin, old_spacing, old_direction, spline=False):
    sitk_img = sitk.GetImageFromArray(img)
    sitk_img.SetOrigin(old_origin)
    sitk_img.SetSpacing(old_spacing)
    sitk_img.SetDirection(old_direction)

    resample = sitk.ResampleImageFilter()
    if spline:
        resample.SetInterpolator(sitk.sitkBSpline)
    resample.SetOutputDirection(sitk_img.GetDirection())
    resample.SetOutputOrigin(sitk_img.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_shape)

    new_image = resample.Execute(sitk_img)
    img_array = sitk.GetArrayFromImage(new_image)
    img_array = img_array.astype(np.float32)

    return img_array, new_image


def interpolation_without_transpose_2(img, new_shape, new_spacing,
                                      old_origin, old_spacing, old_direction,
                                      offset_x, offset_y, offset_z, spline=False):
    sitk_img = sitk.GetImageFromArray(img)
    sitk_img.SetOrigin(old_origin)
    sitk_img.SetSpacing(old_spacing)
    sitk_img.SetDirection(old_direction)

    resample = sitk.ResampleImageFilter()
    if spline:
        resample.SetInterpolator(sitk.sitkBSpline)
    resample.SetOutputDirection(sitk_img.GetDirection())
    resample.SetOutputOrigin(sitk_img.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_shape)

    new_image = resample.Execute(sitk_img)
    img_array = sitk.GetArrayFromImage(new_image)
    img_array = img_array[offset_z:-offset_z, offset_y:-offset_y, offset_x:-offset_x]
    new_image = sitk.GetImageFromArray(img_array)
    img_array = img_array.astype(np.float32)
    return img_array, new_image


def get_new_pixel_coord(pixel_coord, origin, spacing, new_origin, new_spacing):
    world_coord = (1.0 * pixel_coord * spacing) + origin
    new_pixel_coord = 1.0 * (world_coord - new_origin) / new_spacing

    return int(new_pixel_coord)


def get_required_region_3(img_array,
                          x_min, y_min, z_min,
                          x_max, y_max, z_max,
                          size_x, size_y, size_z):
    """

    :param img_array: in z y x coord.
    :param x_min:
    :param y_min:
    :param z_min:
    :param x_max:
    :param y_max:
    :param z_max:
    :param size_x:
    :param size_y:
    :param size_z:
    :return:
    """
    img_shape = img_array.shape

    node_x_min = max(x_min, 0)
    node_y_min = max(y_min, 0)
    node_z_min = max(z_min, 0)
    node_x_max = node_x_min + size_x
    node_y_max = node_y_min + size_y
    node_z_max = node_z_min + size_z

    if node_x_max <= img_shape[2] and node_y_max <= img_shape[1] and node_z_max <= img_shape[0]:
        node_patch = img_array.swapaxes(0, 2)[node_x_min:node_x_max, node_y_min:node_y_max, z_min]
        # node_patch = img_array[node_z_min:node_z_max, node_y_min:node_y_max, node_x_min:node_x_max]
    else:
        node_x_max = min(x_max, img_shape[2])
        node_y_max = min(y_max, img_shape[1])
        node_z_max = min(z_max, img_shape[0])
        node_x_min = node_x_max - size_x
        node_y_min = node_y_max - size_y
        node_z_min = node_z_max - size_z

        if node_x_min >= 0 and node_y_min >= 0 and node_z_min >= 0:
            node_patch = img_array.swapaxes(0, 2)[node_x_min:node_x_max, node_y_min:node_y_max, z_min]
        else:
            raise ValueError("Node_patch doesn't match.")

    # print node_x_max, node_x_min, node_y_max, node_y_min, node_z_max, node_z_min
    return node_patch


def plot_image(img_array, image_size_x, image_size_y, middle_x, middle_y, middle_z):
    fig, ax = plt.subplots(2, 2, figsize=[image_size_x, image_size_y])
    ax[0, 0].imshow(img_array[middle_x, :, :], cmap='gray')
    ax[0, 1].imshow(img_array[:, middle_y, :], cmap='gray')
    ax[1, 0].imshow(img_array[:, :, middle_z], cmap='gray')
    plt.show()


# save image and lable as bin, npy or tfrecord
def write_image(node_patch, filepath):
    sitk.WriteImage(node_patch, str(filepath))


def write_hs(output_data):
    csv_info, pkl_info = get_info_from_csv_and_pkl("HS_")
    # get image_path first
    image_path_and_info_dict = defaultdict(list)
    for i in range(len(pkl_info)):
        image_path_and_info_dict.setdefault(str(pkl_info[i]["image_path"]), []).append(pkl_info[i])
    print len(image_path_and_info_dict)

    # read each image only once
    for key in sorted(image_path_and_info_dict.keys()):
        pkl_info_list = image_path_and_info_dict[key]

        # get image_path
        first_nodule_csv_info = csv_info[csv_info["main_key"] == pkl_info_list[0]["main_key"]]
        if first_nodule_csv_info.shape[0] != 1:
            raise ValueError
        image_path = first_nodule_csv_info["filepath"].iloc[0]
        # print image_path

        if "LIDC-IDRI-0037_" not in image_path:
            continue
        # if "LIDC-IDRI-0487_1.3.6.1.4.1.14519.5.2.1.6279.6001.300270516469599170290456821227" not in image_path:
        #     continue

        # load image
        img_array, origin, spacing, direction = load_image_zyx(str(image_path))

        # get hs feature
        hs_img_array = copy.deepcopy(img_array)
        for i in range(len(pkl_info_list)):
            mask_list = pkl_info_list[i]["pixels"]
            for zyx in mask_list:
                z, y, x = zyx.tolist()
                hs_img_array[z, y, x] = 1

        for i in range(len(pkl_info_list)):
            try:
                # get csv info about this nodule
                one_nodule_csv_info = csv_info[csv_info["main_key"] == pkl_info_list[i]["main_key"]]
                if one_nodule_csv_info.shape[0] != 1:
                    raise ValueError
                print one_nodule_csv_info["main_key"]

                # crop the nodule area
                z_min, y_min, x_min, z_max, y_max, x_max = pkl_info_list[i]["field"]
                if z_max - z_min == 0:
                    z_max = z_min + 1
                z_len = z_max - z_min
                y_len = y_max - y_min
                x_len = x_max - x_min
                z_len_world = 1.0 * z_len * spacing[2]
                y_len_world = 1.0 * y_len * spacing[1]
                x_len_world = 1.0 * x_len * spacing[0]
                print z_min, y_min, x_min, z_max, y_max, x_max

                max_zyx = int(1.25 * max(max(z_len_world, y_len_world), x_len_world))
                # print "\nz: {}\n y: {}\n x: {}\n 1.25 * max_xyz: {}".format(z_len, y_len, x_len, max_zyx)

                new_z_len = int(max_zyx / spacing[2])
                new_y_len = int(max_zyx / spacing[1])
                new_x_len = int(max_zyx / spacing[0])
                new_z_min = z_min - ((new_z_len - z_len) / 2)
                new_z_max = new_z_min + new_z_len
                new_y_min = y_min - ((new_y_len - y_len) / 2)
                new_y_max = new_y_min + new_y_len
                new_x_min = x_min - ((new_x_len - x_len) / 2)
                new_x_max = new_x_min + new_x_len

                nodule_area = get_required_region_3(hs_img_array,
                                                    new_x_min, new_y_min, new_z_min,
                                                    new_x_max, new_y_max, new_z_max,
                                                    new_x_len, new_y_len, new_z_len)

                nodule_area_z, nodule_area_y, nodule_area_x = nodule_area.shape
                final_shape = [final_size_x + final_offset_x,
                               final_size_y + final_offset_y,
                               final_size_z + final_offset_z]
                final_spacing = [(1.0 * nodule_area_x * spacing[0]) / final_shape[0],
                                 (1.0 * nodule_area_y * spacing[1]) / final_shape[1],
                                 (1.0 * nodule_area_z * spacing[2]) / final_shape[2]]

                nodule_area_in_fixed_size, _ = interpolation_without_transpose_2(nodule_area, final_shape, final_spacing,
                                                                                 origin, spacing, direction,
                                                                                 half_final_offset_x,
                                                                                 half_final_offset_y,
                                                                                 half_final_offset_z)

                new_image = sitk.GetImageFromArray(nodule_area_in_fixed_size)
                write_image(new_image, os.path.join(output_data, one_nodule_csv_info["to_be_image_name"].iloc[0]))
            except Exception as e:
                print pkl_info_list[i]["main_key"], "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                print e.message


def write_oa(output_data):
    csv_info, pkl_info = get_info_from_csv_and_pkl("OA_")

    # get image_path first
    image_path_and_info_dict = defaultdict(list)
    for i in range(len(pkl_info)):
        image_path_and_info_dict.setdefault(str(pkl_info[i]["image_path"]), []).append(pkl_info[i])
    print len(image_path_and_info_dict)

    # read each image only once
    for key in sorted(image_path_and_info_dict.keys()):
        pkl_info_list = image_path_and_info_dict[key]

        # get image_path
        first_nodule_csv_info = csv_info[csv_info["main_key"] == pkl_info_list[0]["main_key"]]
        if first_nodule_csv_info.shape[0] != 1:
            raise ValueError
        image_path = first_nodule_csv_info["filepath"].iloc[0]

        # load image
        img_array, origin, spacing, direction = load_image_zyx(str(image_path))
        oa_img_array = copy.deepcopy(img_array)

        # process each nodule
        for i in range(len(pkl_info_list)):
            # get csv info about this nodule
            one_nodule_csv_info = csv_info[csv_info["main_key"] == pkl_info_list[i]["main_key"]]
            if one_nodule_csv_info.shape[0] != 1:
                raise ValueError

            # crop the nodule area
            z_min, y_min, x_min, z_max, y_max, x_max = pkl_info_list[i]["field"]

            if z_max - z_min < 0:
                print one_nodule_csv_info["main_key"].tolist()[0]
                continue
            z_list = range(z_min, z_max + 1, 1)

            y_len = y_max - y_min
            x_len = x_max - x_min
            # print z_min, y_min, x_min, z_max, y_max, x_max

            max_yz = int(1.25 * max(x_len, y_len))
            # print "\nz: {}\n y: {}\n x: {}\n 1.25 * max_xyz: {}".format(z_len, y_len, x_len, max_zyx)

            new_y_len = max_yz
            new_x_len = max_yz
            new_y_min = y_min - ((new_y_len - y_len) / 2)
            new_y_max = new_y_min + new_y_len
            new_x_min = x_min - ((new_x_len - x_len) / 2)
            new_x_max = new_x_min + new_x_len

            for z in z_list:
                try:
                    nodule_area = get_required_region_3(oa_img_array,
                                                        new_x_min, new_y_min, z,
                                                        new_x_max, new_y_max, z + 1,
                                                        new_x_len, new_y_len, 1)

                    nodule_area_in_fixed_size = transform.resize(nodule_area, (final_size_x + final_offset_x,
                                                                               final_size_y + final_offset_y))
                    f_x_min = final_offset_x / 2
                    f_x_max = final_offset_x / 2 + final_size_x
                    f_y_min = final_offset_y / 2
                    f_y_max = final_offset_y / 2 + final_size_y
                    nodule_area_in_fixed_size = nodule_area_in_fixed_size[f_x_min:f_x_max, f_y_min:f_y_max]

                    output_path = os.path.join(output_data, one_nodule_csv_info["to_be_image_name"].iloc[0] + "_" + str(z) + ".jpg")

                    toimage(nodule_area_in_fixed_size).save(output_path)
                    # plt.imshow(nodule_area_in_fixed_size, cmap="gray")
                    # plt.show()

                    # new_image = sitk.GetImageFromArray(nodule_area_in_fixed_size)
                    # write_image(new_image, os.path.join(output_data, one_nodule_csv_info["to_be_image_name"].iloc[0]))
                except Exception as e:
                    print pkl_info_list[i]["main_key"], "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", z
                    print e.message


def write_shuffled_csv():
    csv_info, pkl_info = get_info_from_csv_and_pkl("OA_")

    # get image_path first
    image_path_and_info_dict = defaultdict(list)
    for i in range(len(pkl_info)):
        image_path_and_info_dict.setdefault(str(pkl_info[i]["image_path"]), []).append(pkl_info[i])
    print len(image_path_and_info_dict)

    z_slice_series_list = []

    # read each image only once
    import random
    key_list = image_path_and_info_dict.keys()
    random.shuffle(key_list)

    for key in key_list:
        pkl_info_list = image_path_and_info_dict[key]

        # get image_path
        first_nodule_csv_info = csv_info[csv_info["main_key"] == pkl_info_list[0]["main_key"]]
        if first_nodule_csv_info.shape[0] != 1:
            raise ValueError
        image_path = first_nodule_csv_info["filepath"].iloc[0]

        # if "LIDC-IDRI-0001" not in image_path:
        #    continue

        # process each nodule
        for i in range(len(pkl_info_list)):
            # get csv info about this nodule
            one_nodule_csv_info = csv_info[csv_info["main_key"] == pkl_info_list[i]["main_key"]]
            if one_nodule_csv_info.shape[0] != 1:
                raise ValueError

            # crop the nodule area
            z_min, y_min, x_min, z_max, y_max, x_max = pkl_info_list[i]["field"]

            if z_max - z_min < 0:
                print one_nodule_csv_info["main_key"].tolist()[0]
                continue
            z_list = range(z_min, z_max + 1, 1)

            for z in z_list:
                z_slice = copy.deepcopy(one_nodule_csv_info.iloc[0])
                z_slice["coordZ"] = z

                z_slice_series_list.append(z_slice)

    # by nodule
    df = pd.DataFrame(z_slice_series_list)
    df = df.set_index("main_key").loc[np.random.permutation(df.main_key.unique())].reset_index()

    header = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm',
                      'subtlety', 'internalStructure', 'calcification', 'sphericity',
                      'margin', 'lobulation', 'spiculation', 'texture', 'malignancy']
    df = df[header]
    csvfile_output_path = os.path.join(original_csvfiles_path, "CSVFILES",
                                       "new_shuffled_2_by_nodule_annotations_xml_voxel_2d.csv")
    df.to_csv(csvfile_output_path, header=header, index=False)

if __name__ == '__main__':
    write_oa(processed_data_path_for_oa)
