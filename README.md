# Learning Spatial Common Sense with Geometry-Aware Recurrent Networks

This repository contains the source codes of the CVPR 2019 paper
[Learning Spatial Common Sense with
Geometry-Aware Recurrent Networks](http://openaccess.thecvf.com/content_CVPR_2019/html/Tung_Learning_Spatial_Common_Sense_With_Geometry-Aware_Recurrent_Networks_CVPR_2019_paper.html)
by [Hsiao-Yu Fish Tung](https://sfish0101.bitbucket.io/), [Ricson Cheng](https://ricsonc.github.io/), and [Katerina Fragkiadaki](https://www.cs.cmu.edu/~katef/)


## Installation
### Requirement
Current codes are tested and run with anaconda python 3.6 with tensorflow 1.13.1.
Please install libraries from requirements.txt.

### Dataset
Download the Shapenet dataset from 
```
https://www.dropbox.com/s/7fyu2s384w27lo7/all_tfrs.tar?dl=0
```
and put it in the top level of the repository.


Download rooms_ring_camera and shepard_metzler_7_parts from
```
https://console.cloud.google.com/storage/browser/gqn-dataset?pli=1
```
and put it into a folder named gqn-dataset in the top level of the repository.

### Pretrained Models
Download grnn_checkpoints.tar from
```https://www.dropbox.com/s/25wyz9rzsgp0dpv/grnn_checkpoints.tar?dl=0```
and extract into top level of the repository. 

## Running experiments
### Training view prediction models
Train the proposed models (GRNNs) with
```
python main.py grnn_shapenet_train
python main.py grnn_rooms_train
python main.py grnn_metzler_train
```

Train the baselines (tower model in [Generative Query Network](https://deepmind.com/blog/neural-scene-representation-and-rendering/))
with
```
python main.py tower_shapenet_train
python main.py tower_rooms_train
python main.py tower_metzler_train

```

### Running evaluation and visualization on the (pre)trained models

Run test on the trained models with
```
python main.py grnn_shapenet_eval
python main.py grnn_rooms_eval
python main.py grnn_metzler_eval
```

and

```
python main.py tower_shapenet_eval
python main.py tower_rooms_eval
python main.py tower_metzler_eval

```

## Citations
If you used this repository or the Shapenet dataset, please cite the paper.

@InProceedings{Tung_2019_CVPR,
author = {Fish Tung, Hsiao-Yu and Cheng, Ricson and Fragkiadaki, Katerina},
title = {Learning Spatial Common Sense With Geometry-Aware Recurrent Networks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
