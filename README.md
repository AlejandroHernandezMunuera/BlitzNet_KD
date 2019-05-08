# BlitzNet_Knowledge_Distillation

Knowledge distillation is introduced in a multi-task neural network. The NN used in this set up are: BlitzNet(student), Faster R-CNN(object detection tutor) and Mask R-CNN(semantic segmentation tutor)

## Introduction

This repository is a variation of the BlitzNet implementation(https://github.com/dvornikita/blitznet), adapted to process the corresponding soft output of the detection and segmentation tutors.

Access the BlitzNet GitHub to follow the setup steps and for better understanding. Examples of the tutor data used in this project are available in Extra/tutors_data_examples and the file datasets.py describes the structure of the tutor-dataset used. The tutors used to generate the data were: Faster R-CNN (https://github.com/endernewton/tf-faster-rcnn) and Mask R-CNN (https://github.com/matterport/Mask_RCNN). Changes were introduced in both projects in order to keep their soft output instead of the final one.

The tutors' code is not included in order to have different projects generating the tutor data and facilitate new results and conclusions.


