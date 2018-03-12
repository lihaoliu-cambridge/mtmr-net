# MTMR-net
Code for 《MTMR-net: Multi-Task Deep Learning with Margin Ranking Loss for Lung Nodule Analysis Project》

****
|Author|LIU Lihao|
|---|---
|E-mail|lhliu1994@gmail.com
****


## Introduction

This is the pytorch implementation for 《MTMR-net: Multi-Task Deep Learning with Margin Ranking Loss for Lung Nodule Analysis Project》.(Paper Coming Soon)

We can output a more robust benign-malignant classification result with persuasive semantic feature scores compared to other CAD techniques which can only output classification results, as shown in the figures.
  ![image](https://github.com/CaptainWilliam/MTMR-net/blob/master/data/github_image/fig_1.png)

## Installation

pytorch: http://pytorch.org/

tensorboardX: https://github.com/lanpa/tensorboard-pytorch

Download and unzip this project: MTMR-net-master.zip.

Download and unzip preprocessed data into "./data/" folder：<br>https://drive.google.com/open?id=1xFRQBzuQLv4fO5ecsyKnPc5u2D2N9e76

Download resnet50 model into "./logs/middle_result_logs/imagenet/" folder from pytorch website:<br>https://download.pytorch.org/models/resnet50-19c8e357.pth

## Dataset:

1.Original Dataset:

LIDC-IDRI dataset can be found in the official website: 
https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI

The processing method can be found in this 2 links: 
<br>https://github.com/zhwhong/lidc_nodule_detection
<br>https://github.com/jcausey-astate/NoduleX_code


2.Preprocessed Data:
Find the preprocessed data(2d slices) which can be used directly in the code from Installation section.



## Todos

 - Modify the [args.yaml](https://github.com/CaptainWilliam/Colorization/blob/master/conf/args.yaml), add the parameters your deep learning model need under the "running_params" item. Details are shown in another project: https://github.com/CaptainWilliam/Deep-Learning-Model-Saving-Helper
 - Pass the running_params (a python dict which contains the running parameters) to you own model.
 - The first parameter "is_training" is True for training mode, "is_training" is False for test mode.
 - Finish you mode(training or test), and run it.
 
```sh
$ cd MTMR-net-master
$ python main.py
```
