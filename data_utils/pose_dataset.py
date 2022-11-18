# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE_DEFORMABLE_DETR in the LICENSES folder for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Build a dataset for the pose estimation task. This includes loading the images and annotations consisting of
class, bounding box, relative pose and absolute poses. Moreover, data augmentation and bounding box pertubation is possible.
"""
import copy
from pathlib import Path

import torch
import torch.utils.data
import numpy as np
import random
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection
from util.misc import get_local_rank, get_local_size
from util.quaternion_ops import quat2rot, rot2quat
import data_utils.transforms as T
from scipy.stats import truncnorm, uniform


class PoseDataset(CocoDetection):
    """
    Pose Estimation Dataset. Returns samples consisting of images and the target containing the class, bounding box and
    the pose.
    """
    def __init__(self, img_folder, ann_file, synthetic_background, transforms, return_masks, jitter=False,
                 jitter_probability=0.5, std=0.02, cache_mode=False, local_rank=0, local_size=1):
        """
        Args:
            img_folder (string): path to the directory containing the images
            ann_file (string): path to the file containing the annotations
            synthetic_background (string): path to the directory containing the background images for synthetic images
            transforms (callable): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
            return_masks (bool): Whether to include the segmentation mask
            jitter (bool): Apply jitter to the bounding box
            jitter_probability (float): Probability with which jitter is applied to the bounding box
            std (float): standard deviation of the jitter.
        """
        super(PoseDataset, self).__init__(img_folder, ann_file, synthetic_background,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ProcessPoseData(return_masks)
        self.jitter = jitter
        self.jitter_probability = jitter_probability
        self.std = std

    def __getitem__(self, idx):
        img, target = super(PoseDataset, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        if self.jitter:
            # For the bounding box center we sample from a truncated normal distribution limited by the bounding box
            # width and height for x and y respectively. For the width and height jitter we assume a maximal error of
            # 10% and sample from this error range uniformly.
            jitter_boxes = copy.deepcopy(target["boxes"])
            for box in jitter_boxes:
                # Apply bounding box jitter only with probability
                if random.random() < self.jitter_probability:
                    cxa, cxb = -box[2] / (2 * self.std), box[2] / (2 * self.std)
                    cya, cyb = -box[3] / (2 * self.std), box[3] / (2 * self.std)
                    wa, wb = -0.3 / self.std, 0.3 / self.std
                    ha, hb = -0.3 / self.std, 0.3 / self.std

                    box[0] = truncnorm.rvs(cxa, cxb, loc=box[0], scale=self.std)
                    box[1] = truncnorm.rvs(cya, cyb, loc=box[1], scale=self.std)
                    box[2] = box[2] * (1 + truncnorm.rvs(wa, wb, loc=0, scale=self.std))
                    box[3] = box[3] * (1 + truncnorm.rvs(ha, hb, loc=0, scale=self.std))

            target["jitter_boxes"] = jitter_boxes

        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ProcessPoseData(object):
    """
    Processes the annotation file and brings it in the right format for the pose estimation task.
    """
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        # Load absolute camera pose
        # Only need to store the global camera pose from the first annotated object as it is the same for each object
        cam_position = None
        cam_rotation = None
        # TODO: Implement if rotation stored as quaternions
        if 'camera_pose' in anno[0]:
            if 'position' in anno[0]['camera_pose']:
                cam_position = anno[0]['camera_pose']['position']
                cam_position = torch.tensor(cam_position, dtype=torch.float32)
            if 'rotation' in anno[0]['camera_pose']:
                cam_rotation = anno[0]['camera_pose']['rotation']
                cam_rotation = torch.tensor(cam_rotation, dtype=torch.float32)
                cam_rotation = torch.reshape(cam_rotation, (3, 3))

        # Load absolute object pose
        obj_position = None
        obj_rotation = None
        if 'object_pose' in anno[0]:
            if 'position' in anno[0]['object_pose']:
                obj_position = [obj['object_pose']['position'] for obj in anno]
                obj_position = torch.tensor(obj_position, dtype=torch.float32)
            if 'rotation' in anno[0]['object_pose']:
                obj_rotation = [obj['object_pose']['rotation'] for obj in anno]
                obj_rotation = torch.tensor(obj_rotation, dtype=torch.float32)
                obj_rotation = torch.reshape(obj_rotation, (-1, 3, 3))

        # Load relative pose between camera and object
        rel_position = None
        rel_quaternion = None
        rel_rotation = None
        if 'relative_pose' in anno[0]:
            if 'position' in anno[0]['relative_pose']:
                rel_position = [obj["relative_pose"]['position'] for obj in anno]
                rel_position = torch.tensor(rel_position, dtype=torch.float32)
            if 'quaternions' in anno[0]['relative_pose']:
                rel_quaternion = [obj["relative_pose"]['quaternions'] for obj in anno]
                rel_quaternion = torch.tensor(rel_quaternion, dtype=torch.float32)
            if 'rotation' in anno[0]['relative_pose']:
                rel_rotation = [obj["relative_pose"]['rotation'] for obj in anno]
                rel_rotation = torch.tensor(rel_rotation, dtype=torch.float32)
                if rel_rotation.shape[1] == 9:
                    rel_rotation = torch.reshape(rel_rotation, (-1, 3, 3))
                rel_quaternion = rot2quat(rel_rotation)
                rel_quaternion = torch.tensor(rel_quaternion, dtype=torch.float32)
            else:
                q = np.array([obj["relative_pose"]['quaternions'] for obj in anno])
                rel_rotation = quat2rot(q)
                rel_rotation = torch.tensor(rel_rotation, dtype=torch.float32)

        intrinsics = None
        if 'intrinsics' in anno[0]:
            intrinsics = [obj['intrinsics'] for obj in anno]
            intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        if obj_position is not None:
            obj_position = obj_position[keep]
        if obj_rotation is not None:
            obj_rotation = obj_rotation[keep]
        if rel_position is not None:
            rel_position = rel_position[keep]
        if rel_quaternion is not None:
            rel_quaternion = rel_quaternion[keep]
        if rel_rotation is not None:
            rel_rotation = rel_rotation[keep]
        if intrinsics is not None:
            intrinsics = intrinsics[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints
        if cam_position is not None:
            target["camera_position_w"] = cam_position
        if cam_rotation is not None:
            target["camera_rotation_w"] = cam_rotation
        if obj_position is not None:
            target["object_position_w"] = obj_position
        if obj_rotation is not None:
            target["object_rotation_w"] = obj_rotation
        if rel_position is not None:
            target["relative_position"] = rel_position
        if rel_quaternion is not None:
            target["relative_quaternions"] = rel_quaternion
        if rel_rotation is not None:
            target["relative_rotation"] = rel_rotation
        if intrinsics is not None:
            target["intrinsics"] = intrinsics

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_pose_estimation_transform(image_set, use_rgb_augmentation=False, use_grayscale=False):
    """
    Apply transformations to the images and targets for the pose estimation task depending on the data split.
    """
    # TODO: Add proper data augmentation for pose estimation

    if use_grayscale and image_set not in ['keyframes', 'keyframes_bop', 'test']:
        normalize = T.Compose([
            T.GrayScale(),
            T.ToTensor(),
            T.To3DImage(),
            T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        ])
    else:
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        ])

    rgb_augmentation = T.Compose([T.Blur(),
                                  T.Sharpness(),
                                  T.Contrast(),
                                  T.Brightness(),
                                  T.Color()])

    if image_set == 'train':
        if use_rgb_augmentation:
            return T.Compose([rgb_augmentation, normalize, ])
        else:
            return T.Compose([normalize, ])

    if image_set == 'train_synt':
        if use_rgb_augmentation:
            return T.Compose([rgb_augmentation, normalize, ])
        else:
            return T.Compose([normalize, ])

    if image_set == 'train_pbr':
        if use_rgb_augmentation:
            return T.Compose([rgb_augmentation, normalize, ])
        else:
            return T.Compose([normalize, ])

    if image_set == 'val':
        return T.Compose([
            normalize,
        ])

    if image_set == 'test':
        return T.Compose([
            normalize,
        ])

    if image_set in ['keyframes', 'keyframes_bop']:
        return T.Compose([
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.dataset_path)
    assert root.exists(), f'provided dataset path {root} does not exist'
    PATHS = {
        "train": (root / "train", root / "annotations" / f'train.json'),
        "train_synt": (root / "train", root / "annotations" / f'train_synt.json'),
        "train_pbr": (root / "train", root / "annotations" / f'train_pbr.json'),
        "test": (root / "test_all", root / "annotations" / f'test.json'),
        "keyframes": (root / "test_all", root / "annotations" / f'keyframes.json'),
        "keyframes_bop": (root / "test_all", root / "annotations"/ f'keyframes_bop.json'),
        "val": (root / "val", root / "annotations" / f'val.json'),
    }

    img_folder, ann_file = PATHS[image_set]

    # TODO: Replace 'transforms' by a proper data augmentation function suitable for pose estimation. Currently only
    #  image level augmentation possible (e.g. color augmentation, noise).
    if args.bbox_mode == 'jitter':
        jitter = True
    else:
        jitter = False
    dataset = PoseDataset(img_folder, ann_file, synthetic_background=args.synt_background,
                          transforms=make_pose_estimation_transform(image_set, args.rgb_augmentation, args.grayscale),
                          return_masks=False, jitter=jitter, jitter_probability=args.jitter_probability,
                          cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset
