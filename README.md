# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation

# Introduction

![PoET_small](./figures/PoET_scaled.svg)

This repository is the official implementation of the paper [PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation](https://www.aau.at/wp-content/uploads/2022/09/jantos_poet.pdf).

PoET is a transformer-based framework that takes a single RGB-image as input to simultaneously estimate the 6D pose, namely translation and rotation, for every object present in the image. It takes the detections and feature maps of an object detector backbone and feeds this additional information into an attention-based transformer. Our framework can be trained on top of any object detector framework. Any additional information that is not contained in the raw RGB image, e.g. depth maps or 3D models, are not required. We achieve state-of-the-art-results on challenging 6D object pose estimation datasets. Moreover, PoET can be utilized as a pose sensor in 6D localization tasks.

![network_architecture](./figures/network_architecture.png)

**Abstract:** Accurate 6D object pose estimation is an important task for a variety of robotic applications such as grasping or localization. It is a challenging task due to object symmetries, clutter, occlusion and different scenes, but it becomes even more challenging when additional information, such as depth and 3D models, is not provided. We present a transformer-based approach that takes an RGB image as input and predicts a 6D pose for each object in the image. Besides the image, our network does not require any additional information such as depth maps or 3D object models. First, the image is passed through an object detector to generate feature maps and to detect objects. Second, these feature maps are fed into a transformer while the detected bounding boxes are provided as additional information. Afterwards, the output object queries are processed by a separate translation and rotation head. We achieve state-of-the-art results for RGB-only approaches on the challenging YCB-V dataset. We illustrate the suitability of the resulting model as pose sensor for a 6-DoF state estimation task.

# License
This software is made available to the public to use (_source-available_), licensed under the terms of the BSD-2-Clause-License with no commercial use allowed, the full terms of which are made available in the [LICENSE](./LICENSE) file. No license in patents is granted.

# Citing PoET

If you use PoET for academic research, please cite the corresponding paper and consult the [LICENSE](./LICENSE) file for a detailed explanation.

```latex
@inproceedings{jantos2022poet,
  title={{PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation}},
  author={Jantos, Thomas and Hamdad, Mohamed and Granig, Wolfgang and Weiss, Stephan and Steinbrener, Jan},
  booktitle={6th Annual Conference on Robot Learning (CoRL 2022)}
}
```

# Getting Started

## Requirements

PoET was tested with the following setup

* Linux 20.04
* CUDA 11.4
* Python 3.8.8
* PyTorch 1.9
* other standard packages: numpy, scipy, cv2, cython
* other non-standard packages: [mish-cuda](https://github.com/JunnYu/mish-cuda), [deformable_attention](https://github.com/fundamentalvision/Deformable-DETR/tree/main/models/ops)

## Docker 

We recommend to use PoET within a Docker container. We provide a prebuild and tested Docker image with all the required packages. The Docker image can be pulled with the following command:

```ssh
docker pull gitlab.aau.at:5050/aau-cns-docker/docker_registry/poet:latest
```

PoET can then be run inside the docker container in combination with command line arguments. An example is:

```ssh
docker run --entrypoint= -v /path/to/code/poet:/opt/project -v /path/to/data:/data -v /path/to/output:/output --rm --gpus all gitlab.aau.at:5050/aau-cns-docker/docker_registry/poet:latest python -u ../opt/project/main.py --epochs 50 --batch_size 16 --enc_layers 5 --dec_layers 5 --n_heads 16
```

## Scaled-YOLOv4 Backbone

This repository allows the user to run PoET with a Mask R-CNN object detector backbone. However, if you would like to reproduce the state-of-the-art results presented in our [paper](https://www.aau.at/wp-content/uploads/2022/09/jantos_poet.pdf) you can download our wrapper for the Scaled-YOLOv4 object detector backbone from our [Github](https://github.com/aau-cns/yolov4) (License: GPL 3.0). The backbone can be integrated easily into PoET by placing the code into the models directory.

## Model Zoo

Pretrained models and corresponding hyperparameter configurations can be downloaded from our [website](https://www.aau.at/en/smart-systems-technologies/control-of-networked-systems/datasets/poet-pose-estimation-transformer-for-single-view-multi-object-6d-pose-estimation/).

## BOP Datasets

For both the YCB-V as well as the LM-O datasets, we use the dataset as provided by the [BOP Challenge webpage](https://bop.felk.cvut.cz/home). However, we take the annotations provided by the BOP challenge and transform them to a format inspired by the [COCO annotaiton format](https://cocodataset.org/#home). This means that each. The corresponding script can be found in `data_utils`. The general format of the `annotation.json` file is
* images <dict>: list of images in the dataset with the following information 
	*  file_name <str>: path to file
	*  id <int>: unique image ID
	*  width <int>: width of the image
	*  height <int>: height of the image
	*  intrinsics <array>: array containing the camera intrinsics (3x3 matrix)
	*  type <str>: indicator whether the image is real ("real"), synthetically generated by projecting the 3D model ("synth") or generated by photo-realistic simulation ("pbr").

* categories <dict>: list of categories where each entry contains the name and the ID of the class. Note: class 0 is the `background` class.
* annotations <dict>: list of all annotated objects across all images
	*  id <int>: unique annotation ID
	*  image_id <int>: refers to the image this annotation belongs to
	*  relative_pose <dict>
		* position  <list>: relative translation of the object with respect to the camera (x, y, z)
		* rotation <list>: relative rotation of the object with respect to the camera (3x3 rotation matrix)
	* bbox <list>: upper left corner (x1, y1) and width and height of the bounding box in absolute pixel value
	* category_id <int>: ID to wich category the object belongs to.

We use a dataloader that requires this specific data structure and thus it might be necessary to adapt the dataloader.

