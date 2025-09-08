# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation

# Introduction

![PoET_small](./figures/PoET_scaled.svg)

This repository is the official implementation of the paper [PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation](https://www.aau.at/wp-content/uploads/2022/09/jantos_poet.pdf).

PoET is a transformer-based framework that takes a single RGB-image as input to simultaneously estimate the 6D pose, namely translation and rotation, for every object present in the image. It takes the detections and feature maps of an object detector backbone and feeds this additional information into an attention-based transformer. Our framework can be trained on top of any object detector framework. Any additional information that is not contained in the raw RGB image, e.g. depth maps or 3D models, is not required. We achieve state-of-the-art-results on challenging 6D object pose estimation datasets. Moreover, PoET can be utilized as a pose sensor in 6D localization tasks.

![network_architecture](./figures/network_architecture.png)

**Abstract:** Accurate 6D object pose estimation is an important task for a variety of robotic applications such as grasping or localization. It is a challenging task due to object symmetries, clutter, occlusion and different scenes, but it becomes even more challenging when additional information, such as depth and 3D models, is not provided. We present a transformer-based approach that takes an RGB image as input and predicts a 6D pose for each object in the image. Besides the image, our network does not require any additional information such as depth maps or 3D object models. First, the image is passed through an object detector to generate feature maps and to detect objects. Second, these feature maps are fed into a transformer while the detected bounding boxes are provided as additional information. Afterwards, the output object queries are processed by a separate translation and rotation head. We achieve state-of-the-art results for RGB-only approaches on the challenging YCB-V dataset. We illustrate the suitability of the resulting model as pose sensor for a 6-DoF state estimation task.

# License
This software is made available to the public to use (_source-available_), licensed under the terms of the BSD-2-Clause-License with no commercial use allowed, the full terms of which are made available in the [LICENSE](./LICENSE) file. No license in patents is granted.

# Citing PoET

If you use PoET for academic research, please cite the corresponding paper and consult the [LICENSE](./LICENSE) file for a detailed explanation.

```latex
@inproceedings{jantos2023poet,
  title={PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation},
  author={Jantos, Thomas Georg and Hamdad, Mohamed Amin and Granig, Wolfgang and Weiss, Stephan and Steinbrener, Jan},
  booktitle={Conference on Robot Learning},
  pages={1060--1070},
  year={2023},
  organization={PMLR}
}
```
# Showcases
Here we present showcases of PoET:
* [AI-Based Multi-Object Relative State Estimation with Self-Calibration Capabilities](https://arxiv.org/pdf/2303.00371): In this work, PoET was used to provide object-relative pose measurements between a mobile robot's camera and objects of interest. These measurements are fused with IMU measurements in an Extended Kalman Filter (EKF) to perform object-relative localization.
* [AIVIO: Closed-loop, Object-relative Navigation of UAVs with AI-aided Visual Inertial Odometry](https://arxiv.org/pdf/2410.05996): In this work, a real-time capable UAV system for closed-loop, object-relative navigation is presented. PoET is providing object-relative pose measurements for the Extended Kalman Filter (EKF). The UAV is equipped with an NVIDIA Jetson AGX Orin DevKit and PoET is optimized with TensorRT, achieving up to 50 FPS. Check out the [video of an example flight](https://www.youtube.com/watch?v=0LaYPmUwezg).
* [Aleatoric Uncertainty from AI-based 6D Object Pose Predictors for Object-relative State Estimation](https://arxiv.org/abs/2509.01583): In this work, PoET is adapted for aleatoric uncertainty estimation. Adding dedicated aleatoric uncertainty heads for the translation and rotation makes extending any (pre-trained) network for the uncertainty estimation task possible. The predicted uncertainties are modeled to be Gaussian distributed, thus allowing for a straightforward integration in an EKF for, e.g., object-relative state estimation. The experiments show that the predicted aleatoric uncertainty captures the error characteristics of PoET, even indicating challenging situations through heightened uncertainty values. This enables aleatoric uncertainty-based outlier rejection (AOR). 
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
docker pull aaucns/poet:latest
```

PoET can then be run inside the docker container in combination with command line arguments. An example is:

```ssh
docker run --entrypoint= -v /path/to/code/poet:/opt/project -v /path/to/data:/data -v /path/to/output:/output --rm --gpus all aaucns/poet:latest python -u ../opt/project/main.py --epochs 50 --batch_size 16 --enc_layers 5 --dec_layers 5 --n_heads 16
```

## Evaluation & Inference
The code also allows to evaluate a pre-trained PoET model on a pose dataset, containing ground truth information, or to perform inference on a custom dataset, where only the images are available.

In evaluation PoET processes all images, predicts the 6D pose for all detected objects and calculates the in the paper described evaluation metrics basd on the provided ground truth. The parameter ```--eval``` allows to evaluate for the ADD, ADD-S & ADD(-S) and calculates the translation and rotation error.  On the other hand, ```--eval_bop``` stores the results of PoET in BOP format such that it can be used with the BOP toolbox to evaluate for the metrics of the [BOP Challenge](https://bop.felk.cvut.cz/home). To run PoET in evalaution mode in the Docker container:

```ssh
docker run --entrypoint= -v /path/to/code/poet:/opt/project -v /path/to/data:/data -v /path/to/output:/output --rm --gpus all aaucns/poet:latest python -u ../opt/project/main.py --eval_batch_size 16 --enc_layers 5 --dec_layers 5 --n_heads 16 --resume /path/to/model/checkpoint0049.pth --eval
```
Please remeber to set the ```--eval_set``` parameter correctly.

In a lot of cases we want to perform inference with PoET on data that has no ground truth annotation. For this we provide our [inference_tools](./inference_tools). Currently it contains a simple script to load a custom dataset, processes every image and stores the 6D pose predcitions in a JSON file. The inference mode can be simply activated by using the ```--inference``` flag and setting the parameters correctly. To run PoET in inference mode in the Docker container:

```ssh
docker run --entrypoint= -v /path/to/code/poet:/opt/project -v /path/to/data:/data -v /path/to/output:/output --rm --gpus all aaucns/poet:latest python -u ../opt/project/main.py --enc_layers 5 --dec_layers 5 --n_heads 16 --resume /path/to/model/checkpoint0049.pth --inference --inference_path /path/to/inference/data --inference_output /path/to/output/dir
```

## Distributed Training
If you have multiple GPUs it is possible to train PoET with this [script](./launch_distributed.py). To launch distributed training, run

```ssh
python launch_distributed.py --train_arg_1 --traing_arg_2
```

So for example, if you run single GPU training using 

```ssh
python main.py --epochs 100 --resume output/checkpoint.pth --num_workers 6
```

It would then be

```ssh
python launch_distributed.py --epochs 100 --resume output/checkpoint.pth --num_workers 6
```

Please checkout the runtime arguments in the [launch_distributed.py](./launch_distributed.py) and [main.py](./main.py) scripts and adapt them to your scenario (e.g. number of GPUs). The distributed training also works in the provided docker container, however it requires an additional runtime argument:

```ssh
docker run --entrypoint= -v /path/to/code/poet:/opt/project -v /path/to/data:/data -v /path/to/output:/output --rm --ipc=host --gpus all aaucns/poet:latest python -u ../opt/project/launch_distributed.py --train_arg_1 --traing_arg_2
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

