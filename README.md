# MTMR-Net
Code for [Multi-Task Deep Model with Margin Ranking Loss for Lung Nodule Analysis](https://ieeexplore.ieee.org/document/8794587) on IEEE Transactions on Medical Imaging (TMI).


## Introduction

This repository provides the PyTorch implementation for our TMI paper [Multi-Task Deep Model with Margin Ranking Loss for Lung Nodule Analysis](https://ieeexplore.ieee.org/document/8794587). Our model can output a more robust benign-malignant classification result with persuasive semantic feature scores compared to other CAD techniques which can only output classification results.
  ![image](https://github.com/CaptainWilliam/MTMR-net/blob/master/data/github_image/fig_1.png)


## Requirement

Python == 2.7.13  
PyTorch == 0.3.0  
tensorboardX == 0.9  
numpy == 1.14.3


## Installation

Download and unzip this project: 

   ```shell
   git clone https://github.com/lihaoliu-cambridge/mtmr-net.git
   cd mtmr-net
   ```
   
Download [resnet50.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth) into `./logs/middle_result_logs/imagenet/` folder.


## Dataset

Download and [original LIDC-IDRI dataset](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) into `./data/` folder

The preprocessing methods can be found in below two links:  
https://github.com/zhwhong/lidc_nodule_detection  
https://github.com/jcausey-astate/NoduleX_code


## Todos

 - Modify the [args.yaml](https://github.com/CaptainWilliam/MTMR-net/blob/master/conf/args.yaml), add the parameters your deep learning model need under the "running_params" item. Details are shown in another project: https://github.com/CaptainWilliam/Deep-Learning-Model-Saving-Helper
 - Pass the running_params (a python dict which contains the running parameters) to you own model.
 - The first parameter "is_training" is True for training mode, "is_training" is False for test mode.
 - Finish you mode(training or test), and run it.
 
   ```shell
   $ cd mtmr-net
   $ python main.py
   ```

## Acknowledgement

:kissing_smiling_eyes:Thanks my dearest brother yong for this beautiful figure.
