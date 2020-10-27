# MTMR-Net
Code for 《Multi-Task Deep Model with Margin Ranking Loss for Lung Nodule Analysis》 Project.

****
|Author|LIU Lihao|
|---|---
|E-mail|lhliu1994@gmail.com
****


## Introduction

We can output a more robust benign-malignant classification result with persuasive semantic feature scores compared to other CAD techniques which can only output classification results, as shown in the figures.
  ![image](https://github.com/CaptainWilliam/MTMR-net/blob/master/data/github_image/fig_1.png)

## Requirement

Python 2.7.13

PyTorch == 0.3.0

tensorboardX == 0.9

numpy == 1.14.3

## Installation

pytorch: http://pytorch.org/

tensorboardX: https://github.com/lanpa/tensorboard-pytorch

Download and unzip this project: MTMR-net-master.zip.


Download resnet50 model into "./logs/middle_result_logs/imagenet/" folder from pytorch website:<br>https://download.pytorch.org/models/resnet50-19c8e357.pth

## Dataset

Original LIDC-IDRI dataset can be found in the official website: 
<br>https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI

Download and data into "./data/" folder.

The preprocessing methods can be found in below 2 links: 
<br>https://github.com/zhwhong/lidc_nodule_detection
<br>https://github.com/jcausey-astate/NoduleX_code


## Todos

 - Modify the [args.yaml](https://github.com/CaptainWilliam/MTMR-net/blob/master/conf/args.yaml), add the parameters your deep learning model need under the "running_params" item. Details are shown in another project: https://github.com/CaptainWilliam/Deep-Learning-Model-Saving-Helper
 - Pass the running_params (a python dict which contains the running parameters) to you own model.
 - The first parameter "is_training" is True for training mode, "is_training" is False for test mode.
 - Finish you mode(training or test), and run it.
 
```sh
$ cd MTMR-NET-master
$ python main.py
```

## Acknowledgement

:kissing_smiling_eyes:Thanks my dearest brother yong for this beautiful figure.
