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
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class PoseMatcher(nn.Module):
    """
    This class computes an assignment between the network's predictions and targets. The matching is
    done based on the predicted bounding boxes. However, the predicted class is used to remove matches if the class is
    off.
    """

    def __init__(self,
                 cost_bbox: float = 1,
                 cost_class: float = 1,
                 bbox_mode: str = "gt",
                 class_mode: str = "specific"):
        """
        cost_bbox: weighting parameter for the bounding box cost
        cost_class: weighting parameter for the class cost
        bbox_mode: mode with which the bounding box information was fed to the transformer part of PoET
        class_mode: determines whether PoET is used in a class specific or agnostic way
        """
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_class = cost_class
        self.bbox_mode = bbox_mode
        self.class_mode = class_mode

    def forward(self, outputs, targets, n_boxes, giou_thresh=0.5):
        """ Performs the matching

                Params:
                    outputs: This is a dict that contains at least these entries:
                         "pred_translation": Tensor of dim [batch_size, num_queries, 3 (*n_classes)] with the predicted translation
                         "pred_rotation": Tensor of dim [batch_size, num_queries, rot_dim (*n_classes)] with the predicted rotations
                         "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                         "pred_classes": Tensor of dim [batch_size, num_queries, 1] with the predicted classes


                    targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                         "relative_pose":
                            "position": Tensor of dim [num_target_boxes, 3 (*n_classes)] containing the target translation
                            "rotation": Tensor of dim [num_target_boxes, rot_dim (*n_classes)] containing the target rotation
                         "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                         "labels": Tensor of dim [num_target_boxes, 1] containing the target labels

                    n_boxes: This is a list of number of boxes (len(n_boxes) = batch_size) predicted per image.

                    giou_thresh: threshold value that the generalized IoU between predicted and target box
                    have to have for the matching

                Returns:
                    A list of size batch_size, containing tuples of (index_i, index_j) where:
                        - index_i is the indices of the selected predictions (in order)
                        - index_j is the indices of the corresponding selected targets (in order)
                    For each batch element, it holds:
                        len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
                """
        with torch.no_grad():
            bs, num_queries = outputs["pred_boxes"].shape[:2]

            # Flatten to compute cost matrices in a batch
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
            out_class = outputs["pred_classes"].flatten(0, 1)

            # Concat target boxes
            tgt_bbox = torch.cat([t["boxes"] for t in targets])
            tgt_class = torch.cat([t["labels"].type(torch.float32) for t in targets])
            
            if self.bbox_mode == "gt":
                # Compute the L1 cost between boxes
                # In 'gt' mode sufficient to match by bounding box center distance as they are the same
                cost_class = 0
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            elif self.bbox_mode == "jitter":
                # Match based purely on the class as there is a perfect one to one matching
                cost_bbox = 0
                cost_class = []
                for cls in out_class:
                    cost_class.append(torch.where(cls == tgt_class, 0., 1.))
                cost_class = torch.stack(cost_class)

            elif self.bbox_mode == "backbone":
                # Compute L1 cost between box centers
                cost_bbox = torch.cdist(out_bbox[:, 0:2], tgt_bbox[:, 0:2], p=1)

                # Compute classification cost
                cost_class = []
                for cls in out_class:
                    cost_class.append(torch.where(cls == tgt_class, 0., 1.))
                cost_class = torch.stack(cost_class)

            # Final cost matrix
            # TODO: Find a better weighting between bounding box and class cost // e.g. normalization
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class
            C = C.view(bs, num_queries, -1).cpu()

            # PoET adds dummy query embeddings to allow for batch processing.
            # The transformer does not change the order of the queries, hence the indices of the dummy embeddings are known
            # Filter them out by taking only the first n_boxes boxes predicted per image in the batch
            sizes = [len(t["boxes"]) for t in targets]
            indices = [linear_sum_assignment(c[i][:n_boxes[i]]) for i, c in enumerate(C.split(sizes, -1))]

            # TODO: Adapt for other modes
            if self.bbox_mode == "backbone":
                # Calculate the generalized IoU and remove matches if the boxes do not overlap at all --> no prediction
                new_indices = []
                for b, (out_box, out_cls, tgt) in enumerate(zip(out_bbox.split(num_queries), out_class.split(num_queries), targets)):
                    tgt_box = tgt["boxes"]
                    tgt_cls = tgt["labels"]
                    gious = generalized_box_iou(box_cxcywh_to_xyxy(out_box[:n_boxes[b]]), box_cxcywh_to_xyxy(tgt_box))
                    new_src_idx = []
                    new_tgt_idx = []
                    for idx, (i, j) in enumerate(zip(indices[b][0], indices[b][1])):
                        if self.class_mode == "specific":
                            if out_cls[i] != tgt_cls[j]:
                                #print("Match removed Class")
                                continue
                        giou = gious[i, j]
                        if giou < giou_thresh:
                            # print("Match removed GIoU: {}".format(giou))
                            continue
                        else:
                            new_src_idx.append(i)
                            new_tgt_idx.append(j)
                    new_indices.append((np.array(new_src_idx), np.array(new_tgt_idx)))
                indices = new_indices

            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    if args.matcher_type == 'hungarian':
        return HungarianMatcher(cost_class=args.set_cost_class,
                                cost_bbox=args.set_cost_bbox,
                                cost_giou=args.set_cost_giou)
    elif args.matcher_type == 'pose':
        return PoseMatcher(cost_bbox=args.set_cost_bbox, cost_class=args.set_cost_class, bbox_mode=args.bbox_mode,
                           class_mode=args.class_mode)
    else:
        print("Matcher type not implemented!")
        raise NotImplementedError
