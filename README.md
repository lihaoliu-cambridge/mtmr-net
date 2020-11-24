# MTMR-Net
Code for [Multi-Task Deep Model with Margin Ranking Loss for Lung Nodule Analysis](https://ieeexplore.ieee.org/document/8794587) on IEEE Transactions on Medical Imaging (TMI).  


## Introduction

This repository provides the PyTorch implementation for our TMI paper "Multi-Task Deep Model with Margin Ranking Loss for Lung Nodule Analysis". Our model can output a more robust benign-malignant classification result with persuasive semantic feature scores compared to other CAD techniques which can only output classification results.
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

Download the original [LIDC-IDRI dataset](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) into `./data/` folder

The preprocessing methods can be found in below two links:  
https://github.com/zhwhong/lidc_nodule_detection  
https://github.com/jcausey-astate/NoduleX_code  


## Running

 - Modify the [args.yaml](https://github.com/CaptainWilliam/MTMR-net/blob/master/conf/args.yaml), add the parameters of your deep learning model under the "running_params" item. Details are shown in [deep-learning-model-saving-helper](https://github.com/lihaoliu-cambridge/deep-learning-model-saving-helper) project.  
 - Pass the running_params (a python dict which contains the running parameters) to you own model.  
 - The first parameter "is_training" is True for training mode, "is_training" is False for test mode.  
 - Finish you mode(training or test), and run it.  
 
   ```shell
   cd mtmr-net
   python main.py
   ```  
   
   
## Citation

If you use our code for your research, please cite our paper:

```
@article{liu2019multi,
  title={Multi-task deep model with margin ranking loss for lung nodule analysis},
  author={Liu, Lihao and Dou, Qi and Chen, Hao and Qin, Jing and Heng, Pheng-Ann},
  journal={IEEE transactions on medical imaging},
  volume={39},
  number={3},
  pages={718--728},
  year={2019},
  publisher={IEEE}
}
```

## Question

Please open an issue or email 'lhliu1994@gmail.com' for any question.


## Acknowledgement

:kissing_smiling_eyes:Thanks my dearest brother Yong for this beautiful figure.
