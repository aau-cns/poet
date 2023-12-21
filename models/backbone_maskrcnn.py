# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator, concat_box_prediction_layers
from typing import Dict, List
import yaml
from pathlib import Path


from util.misc import NestedTensor


class MaskRCNNBackbone(MaskRCNN):
    """
    MaskRCNN with ResNet50 Backbone as Object Detector Backbone for the Pose Estimation Transformer.
    Returns detected/predicted objects (class, bounding box) and the feature maps.
    """
    def __init__(self, input_resize=(240, 320), n_classes=8, backbone_str='resnet50-fpn', train_backbone=False,
                 return_interm_layers=True, dataset='lmo',
                 anchor_sizes=((32, ), (64, ), (128, ), (256, ), (512, ))):

        assert backbone_str == 'resnet50-fpn'
        backbone = resnet_fpn_backbone('resnet50', pretrained=False)

        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        super().__init__(backbone=backbone, num_classes=n_classes, rpn_anchor_generator=rpn_anchor_generator,
                         max_size=max(input_resize), min_size=min(input_resize))

        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            self.return_layers = ['2', '3', 'pool']
            # Might be wrong
            self.strides = [8, 16, 32]
            self.num_channels = [256, 256, 256]
        else:
            self.return_layers = ['pool']
            self.strides = [256]
            self.num_channels = [256]

        # Freeze backbone if it should not be trained
        self.train_backbone = train_backbone
        if not train_backbone:
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(False)

        # For the LMO set we need to map the object ids correctly.
        self.obj_id_map = None
        if dataset == 'lmo':
            self.obj_id_map = {1: 1, 5: 2, 6: 3, 8: 4, 9: 5, 10: 6, 11: 7, 12: 8}

    def forward(self, tensor_list: NestedTensor):
        image_sizes = [img.shape[-2:] for img in tensor_list.tensors]
        # xs = self.backbone.body(tensor_list.tensors)
        features = self.backbone(tensor_list.tensors)
        # predictions, _ = self.rpn(tensor_list.tensors, features)

        # Generate proposals using the RPN
        feature_maps = list(features.values())
        objectness, pred_bbox_deltas = self.rpn.head(feature_maps)
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = tensor_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]
        self.rpn.anchor_generator.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.rpn.anchor_generator.grid_anchors(grid_sizes, strides)
        anchors: List[List[torch.Tensor]] = []
        for _ in range(len(tensor_list.tensors)):
            anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.rpn.filter_proposals(proposals, objectness, image_sizes, num_anchors_per_level)
        detections, _ = self.roi_heads(features, boxes, image_sizes)

        # Translate the detections to predictions: [bbox, score, label]
        # TODO: optimize code
        predictions = []
        for img_detections in detections:
            img_predictions = []
            for c, cls in enumerate(img_detections['labels']):
                box = img_detections["boxes"][c]
                box = torch.hstack((box, img_detections["scores"][c]))
                if self.obj_id_map is not None:
                    if cls.item() in self.obj_id_map.keys():
                        new_cls = self.obj_id_map[cls.item()]
                        box = torch.hstack((box, torch.tensor(new_cls, dtype=torch.float32, device=device)))
                    else:
                        # Processing object that has a label that is not in the object_id_map --> skip
                        continue
                else:
                    box = torch.hstack((box, cls))
                img_predictions.append(box)
            if len(img_predictions) == 0:
                # Either no objects present or no detected --> Append None
                img_predictions = None
            else:
                img_predictions = torch.stack(img_predictions)
            predictions.append(img_predictions)

        # Prepare the feature map
        out: Dict[str, NestedTensor] = {}
        for name in self.return_layers:
            x = features[name]
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return predictions, out


def build_maskrcnn(args):
    train_backbone = args.lr_backbone > 0
    return_interm_layers = (args.num_feature_levels > 1)
    rcnn_cfg = yaml.load(Path(args.backbone_cfg).read_text(), Loader=yaml.FullLoader)
    n_classes = len(rcnn_cfg["label_to_category_id"])
    backbone = MaskRCNNBackbone(input_resize=(rcnn_cfg["input_resize"][0], rcnn_cfg["input_resize"][1]),
                                dataset=args.dataset,
                                n_classes=n_classes,
                                backbone_str=rcnn_cfg["backbone_str"])
    if args.backbone_weights is not None:
        ckpt = torch.load(args.backbone_weights)
        if args.backbone == "maskrcnn":
            ckpt = ckpt['state_dict']
            backbone.load_state_dict(ckpt)
        elif args.backbone == "fasterrcnn":
            ckpt = ckpt['model']
            missing_keys, unexpected_keys = backbone.load_state_dict(ckpt, strict=False)
            if len(missing_keys) > 0:
                print("Loading Faster R-CNN weights")
                print('Missing Keys: {}'.format(missing_keys))
                print('PoET does not rely on the mask head of Mask R-CNN')
    return backbone

