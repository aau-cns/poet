# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE_DEFORMABLE_DETR for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Backbone modules.
"""

import torch
from torch import nn
from typing import List
from util.misc import NestedTensor

from .position_encoding import build_position_encoding
from .backbone_maskrcnn import build_maskrcnn


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        # TODO: Dirty, fix it
        # TODO: Currently the Object detector backbone has to be pretrained. Extend code to make object detectors
        #  trainable.
        if self[0].train_backbone:
            raise NotImplementedError
        else:
            self[0].eval()
            predictions, xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos, predictions


def build_backbone(args):
    # Build the positional embedding
    position_embedding = build_position_encoding(args)

    # Build the object detector backbone
    if args.backbone in ["maskrcnn", "fasterrcnn"]:
        backbone = build_maskrcnn(args)
    else:
        raise NotImplementedError
    model = Joiner(backbone, position_embedding)
    return model
