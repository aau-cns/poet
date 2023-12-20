# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE_DEFORMABLE_DETR in the LICENSES folder for details]
# ------------------------------------------------------------------------
# Modified from torchvision
# ------------------------------------------------------------------------

"""
Copy-Paste from torchvision, but add utility of caching images on memory
Extend for adding backgrounds to synthetic images during loading.
"""
import PIL.Image
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO
import random
import copy


class CocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        synthetic_background (string, optional): Path to the directory containing background images for synthetic images.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, synthetic_background=None, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        if cache_mode:
            self.cache = {}
            self.cache_images()

        # For Pose Estimation: Directory containing background images to sample from for synthetic data
        if synthetic_background is not None:
            self.synthetic_background = [synthetic_background + f for f in os.listdir(synthetic_background) if os.path.isfile(os.path.join(synthetic_background, f))]
        else:
            self.synthetic_background = None

        # For Pose Estimation: Check whether camera intrinsic is provided
        self.intrinsics = False
        if "intrinsics" in self.coco.imgs[0]:
            self.intrinsics = True

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.coco.loadImgs(img_id)[0]['file_name']
            with open(os.path.join(self.root, path), 'rb') as f:
                self.cache[path] = f.read()

    def get_image(self, path, mode='RGB'):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert(mode)
        return Image.open(os.path.join(self.root, path)).convert(mode)

    def get_background(self, target_size):
        n_background_images = len(self.synthetic_background)
        background_id = random.randint(0, n_background_images-1)
        path = self.synthetic_background[background_id]
        background_image = Image.open(path).convert('RGB')
        w, h = background_image.size
        # Perform random flipping
        if random.random() < 0.5:
            # Horizontal flip
            background_image = background_image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        elif random.random() < 0.5:
            # Vertical flipping
            background_image = background_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        # Perform random cropping
        if random.random() < 0.5:
            left = random.randint(0, w)
            top = random.randint(0, h)
            right = random.randint(left, w)
            bottom = random.randint(top, h)
            background_image = background_image.crop((left, top, right, bottom))
        background_image = background_image.resize(target_size)
        return background_image

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = copy.deepcopy(coco.loadAnns(ann_ids))

        # Load the intrinsics into the target annotation
        path = coco.loadImgs(img_id)[0]['file_name']
        if self.intrinsics:
            intrinsics = coco.imgs[img_id]['intrinsics']
            for tgt in target:
                tgt['intrinsics'] = intrinsics

        # Check whether image is synthetic
        synthetic = False
        mode = "RGB"
        if "type" in coco.imgs[img_id]:
            if coco.imgs[img_id]["type"] == "synt":
                synthetic = True
                mode = "RGBA"

        img = self.get_image(path, mode)
        # Load a random background image if image is synthetic
        if synthetic:
            if self.synthetic_background is None:
                print("DataLoader tries to load a synthetic background, but none is provided. Skipping this step.")
            else:
                background_img = self.get_background(img.size)
                background_img.paste(img, (0, 0), img)
                img = background_img.copy()

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
