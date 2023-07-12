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

import re
import torchvision.transforms.functional as F
from torchvision.datasets.vision import VisionDataset
from pathlib import Path
from PIL import Image
from os import listdir
from os.path import isfile, join


class InferenceDataset(VisionDataset):
    """
    Dataset for PoET Inference
    Only Images are loaded as no annotation / GT data is provided for the inference case
    Load the images from the provided path and sort them alphabetically.
    """
    def __init__(self, root, transform=None, target_transform=None, transforms=None):
        super(InferenceDataset, self).__init__(root, transforms, transform, target_transform)

        self.root = root
        self.image_paths = [f for f in listdir(root) if isfile(join(root, f))]
        self.image_paths.sort(key=lambda f: int(re.sub('\D', '', f)))

    def get_image(self, path, mode='RGB'):
        return Image.open(join(self.root, path)).convert(mode)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the image to be returned
        Returns:
            image
        """
        img_path = self.image_paths[index]
        # print(img_path)
        img = self.get_image(img_path, mode="RGB")
        img = F.to_tensor(img)
        return img, None

    def __len__(self):
        return len(self.image_paths)


def build_dataset(args):
    root = Path(args.inference_path)
    dataset = InferenceDataset(root)
    return dataset

