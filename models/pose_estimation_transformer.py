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

import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list)
from .backbone import build_backbone
from .matcher import build_matcher
from .deformable_transformer import build_deforamble_transformer
from .position_encoding import BoundingBoxEmbeddingSine
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PoET(nn.Module):
    """
    Pose Estimation Transformer module that performs 6D, multi-object relative pose estimation.
    """
    def __init__(self, backbone, transformer, num_queries, num_feature_levels, n_classes, bbox_mode='gt',
                 ref_points_mode='bbox', query_embedding_mode='bbox', rotation_mode='6d', class_mode='agnostic',
                 aux_loss=True, backbone_type="yolo"):
        """
        Initalizing the model.
        Parameters:
            backbone: torch module of the backbone to be used. Includes backbone and positional encoding.
            transformer: torch module of the transformer architecture
            num_queries: number of queries that the transformer receives. Is equal to the number of expected objects
            in the image
            num_feature_levels: number of feature levels that serve as input to the transformer.
            n_classes: number of classes present in the dataset.
            bbox_mode: mode that determines which and how bounding box information is fed into the transformer.
            ref_points_mode: mode that defines how the transformer determines the reference points.
            query_embedding_mode: mode that defines how the query embeddings are determined.
            rotation_mode: determines the rotation representation
            class_mode: determines whether PoET is trained class specific or agnostic
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            backbone_type: object detector backbone type
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.backbone = backbone
        self.backbone_type = backbone_type
        self.aux_loss = aux_loss
        self.n_queries = num_queries
        self.n_classes = n_classes + 1  # +1 for dummy/background class
        self.bbox_mode = bbox_mode
        self.ref_points_mode = ref_points_mode
        self.query_embedding_mode = query_embedding_mode
        self.rotation_mode = rotation_mode
        self.class_mode = class_mode

        # Determine Translation and Rotation head output dimension
        self.t_dim = 3
        if self.rotation_mode == '6d':
            self.rot_dim = 6
        elif self.rotation_mode in ['quat', 'silho_quat']:
            self.rot_dim = 4
        else:
            raise NotImplementedError('Rotational representation is not supported.')

        # Translation & Rotation Estimation Head
        if self.class_mode == 'agnostic':
            self.translation_head = MLP(hidden_dim, hidden_dim, self.t_dim, 3)
            self.rotation_head = MLP(hidden_dim, hidden_dim, self.rot_dim, 3)
        elif self.class_mode == 'specific':
            self.translation_head = MLP(hidden_dim, hidden_dim, self.t_dim * self.n_classes, 3)
            self.rotation_head = MLP(hidden_dim, hidden_dim, self.rot_dim * self.n_classes, 3)
        else:
            raise NotImplementedError('Class mode is not supported.')

        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 1:
            # Use multi-scale features as input to the transformer
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            # If multi-scale then every intermediate backbone feature map is returned
            for n in range(num_backbone_outs):
                in_channels = backbone.num_channels[n]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            # If more feature levels are required than backbone feature maps are available then the last feature map is
            # passed through an additional 3x3 Conv layer to create a new feature map.
            # This new feature map is then used as the baseline for the next feature map to calculate
            # For details refer to the Deformable DETR paper's appendix.
            for n in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            # We only want to use the backbones last feature embedding map.
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            ])

        # Initialize the projection layers
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # Pose is predicted for each intermediate decoder layer for training with auxiliary losses
        # Only the prediction from the final layer will be used for the final pose estimation
        num_pred = transformer.decoder.num_layers
        self.translation_head = nn.ModuleList([copy.deepcopy(self.translation_head) for _ in range(num_pred)])
        self.rotation_head = nn.ModuleList([copy.deepcopy(self.rotation_head) for _ in range(num_pred)])

        # Positional Embedding for bounding boxes to generate query embeddings
        if self.query_embedding_mode == 'bbox':
            self.bbox_embedding = BoundingBoxEmbeddingSine(num_pos_feats=hidden_dim / 8)
        elif self.query_embedding_mode == 'learned':
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
            # TODO: Optimize Code to not generate bounding box query embeddings, when query embed is in learning mode.
            self.bbox_embedding = BoundingBoxEmbeddingSine(num_pos_feats=hidden_dim / 8)
        else:
            raise NotImplementedError('This query embedding mode is not implemented.')

    def forward(self, samples: NestedTensor, targets=None):
        """
        Function expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size X H x W], containing 1 on padded pixels

        Functions expects a list of length batch_size, where each element is a dict with the following entries:
            - boxes: tensor of size [n_obj, 4], contains the bounding box (x_c, y_c, w, h) of each object in each image
            normalized to image size
            - labels: tensor of size [n_obj, ], contains the label of each object in the image
            - image_id: tensor of size [1],  contains the image id to which this annotation belongs to
            - relative_position; tensor of size [n_obj, 3], contains the relative translation for each object present
            in the image w.r.t the camera.
            - relative_rotation: tensor of size [n_obj, 3, 3], contains the relative rotation for each object present
            in the image w.r.t. the camera.


        It returns a dict with the following elements:
            - pred_translation: tensor of size [batch_size, n_queries, 3], predicted relative translation for each
            object query w.r.t. camera
            - pred_rotation: tensor of size [batch_size, n_queries, 3, 3], predicted relative rotation for each
            object query w.r.t. camera
            - pred_boxes: tensor of size [batch_size, n_queries, 4], predicted bounding boxes (x_c, y_c, w, h) for each
            object query normalized to the image size
            - pred_classes: tensor of size [batch_size, n_queries], predicted class for each
            object query
            - aux_outputs: Optional, only returned when auxiliary losses are activated. It is a list of dictionaries
            containing the output values for each decoder layer.

        It returns a list "n_boxes_per_sample" of length [batch_size, 1], which contains the number of
        """

        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        # Store the image size in HxW
        image_sizes = [[sample.shape[-2], sample.shape[-1]] for sample in samples.tensors]
        features, pos, pred_objects = self.backbone(samples)

        # Extract the bounding boxes for each batch element
        pred_boxes = []
        pred_classes = []
        query_embeds = []
        n_boxes_per_sample = []

        # Depending on the bbox mode, we either use ground truth bounding boxes or backbone predicted bounding boxes for
        # transformer query input embedding calculation.
        if self.bbox_mode in ['gt', 'jitter'] and targets is not None:
            for t, target in enumerate(targets):
                # GT from COCO loaded as x1,y1,x2,y2, but by data loader transformed to cx, cy, w, h and normalized
                if self.bbox_mode == 'gt':
                    t_boxes = target["boxes"]
                elif self.bbox_mode == 'jitter':
                    t_boxes = target["jitter_boxes"]
                n_boxes = len(t_boxes)
                n_boxes_per_sample.append(n_boxes)

                # Add classes
                t_classes = target["labels"]

                # For the current number of boxes determine the query embedding
                query_embed = self.bbox_embedding(t_boxes)
                # As the embedding will serve as the query and key for attention, duplicate it to be later splitted
                query_embed = query_embed.repeat(1, 2)

                # We always predict a fixed number of object poses per image set to the maximum number of objects
                # present in a single image throughout the whole dataset. Check whether this upper limit is reached,
                # otherwise fill up with dummy embeddings that are defined as cx,cy,w,h = [-1, -1, -1, -1]
                # Dummy boxes will later be filtered out by the matcher and not used for cost calculation
                if n_boxes < self.n_queries:
                    dummy_boxes = torch.tensor([[-1, -1, -1, -1] for i in range(self.n_queries-n_boxes)],
                                               dtype=torch.float32, device=t_boxes.device)

                    dummy_embed = torch.tensor([[-10] for i in range(self.n_queries-n_boxes)],
                                               dtype=torch.float32, device=t_boxes.device)
                    dummy_embed = dummy_embed.repeat(1, self.hidden_dim*2)
                    t_boxes = torch.vstack((t_boxes, dummy_boxes))
                    query_embed = torch.cat([query_embed, dummy_embed], dim=0)
                    dummy_classes = torch.tensor([-1 for i in range(self.n_queries-n_boxes)],
                                               dtype=torch.int, device=t_boxes.device)
                    t_classes = torch.cat((t_classes, dummy_classes))
                pred_boxes.append(t_boxes)
                query_embeds.append(query_embed)
                pred_classes.append(t_classes)
        elif self.bbox_mode == 'backbone':
            # Prepare the output predicted by the backbone
            # Iterate over batch and prepare each image in batch
            for bs, predictions in enumerate(pred_objects):
                if predictions is None:
                    # Case: Backbone has not predicted anything for image
                    # Add only dummy boxes, but mark that nothing has been predicted
                    n_boxes = 0
                    n_boxes_per_sample.append(n_boxes)
                    backbone_boxes = torch.tensor([[-1, -1, -1, -1] for i in range(self.n_queries - n_boxes)],
                                                  dtype=torch.float32, device=features[0].decompose()[0].device)
                    query_embed = torch.tensor([[-10] for i in range(self.n_queries - n_boxes)],
                                               dtype=torch.float32, device=features[0].decompose()[0].device)
                    query_embed = query_embed.repeat(1, self.hidden_dim * 2)
                    backbone_classes = torch.tensor([-1 for i in range(self.n_queries - n_boxes)], dtype=torch.int64,
                                                    device=features[0].decompose()[0].device)
                else:
                    # Case: Backbone predicted something
                    backbone_boxes = predictions[:, :4]
                    backbone_boxes = box_ops.box_xyxy_to_cxcywh(backbone_boxes)
                    # TODO: Adapt to different image sizes as we assume constant image size across the batch
                    backbone_boxes = box_ops.box_normalize_cxcywh(backbone_boxes, image_sizes[0])
                    n_boxes = len(backbone_boxes)

                    # Predicted classes by backbone // class 0 is "background"
                    # Scores predicted by the backbone are needed for top-k selection
                    backbone_scores = predictions[:, 4]
                    backbone_classes = predictions[:, 5]
                    backbone_classes = backbone_classes.type(torch.int64)

                    # For the current number of boxes determine the query embedding
                    query_embed = self.bbox_embedding(backbone_boxes)
                    # As the embedding will serve as the query and key for attention, duplicate it to be later splitted
                    query_embed = query_embed.repeat(1, 2)

                    if n_boxes < self.n_queries:
                        # Fill up with dummy boxes to match the query size and add dummy embeddings
                        dummy_boxes = torch.tensor([[-1, -1, -1, -1] for i in range(self.n_queries - n_boxes)],
                                                   dtype=torch.float32, device=backbone_boxes.device)
                        dummy_embed = torch.tensor([[-10] for i in range(self.n_queries - n_boxes)],
                                                   dtype=torch.float32, device=backbone_boxes.device)
                        dummy_embed = dummy_embed.repeat(1, self.hidden_dim * 2)
                        backbone_boxes = torch.cat([backbone_boxes, dummy_boxes], dim=0)
                        query_embed = torch.cat([query_embed, dummy_embed], dim=0)
                        dummy_classes = torch.tensor([-1 for i in range(self.n_queries - n_boxes)],
                                                     dtype=torch.int64, device=backbone_boxes.device)
                        backbone_classes = torch.cat([backbone_classes, dummy_classes], dim=0)
                    elif n_boxes > self.n_queries:
                        # Number of boxes will be limited to the number of queries
                        n_boxes = self.n_queries
                        # Case: backbone predicts more output objects than queries available --> take top n_queries
                        # Sort scores to get the post top performing ones
                        backbone_scores, indices = torch.sort(backbone_scores, dim=0, descending=True)
                        backbone_classes = backbone_classes[indices]
                        backbone_boxes = backbone_boxes[indices, :]
                        query_embed = query_embed[indices, :]

                        # Take the top n predictions
                        backbone_scores = backbone_scores[:self.n_queries]
                        backbone_classes = backbone_classes[:self.n_queries]
                        backbone_boxes = backbone_boxes[:self.n_queries]
                        query_embed = query_embed[:self.n_queries]
                    n_boxes_per_sample.append(n_boxes)
                pred_boxes.append(backbone_boxes)
                pred_classes.append(backbone_classes)
                query_embeds.append(query_embed)
        else:
            raise NotImplementedError("PoET Bounding Box Mode not implemented!")

        query_embeds = torch.stack(query_embeds)
        pred_boxes = torch.stack(pred_boxes)
        pred_classes = torch.stack(pred_classes)

        srcs = []
        masks = []
        for lvl, feat in enumerate(features):
            # Iterate over each feature map of the backbone returned.
            # If num_feature_levels == 1 then the backbone will only return the last one. Otherwise each is returned.
            src, mask = feat.decompose()
            srcs.append(self.input_proj[lvl](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            # If more feature levels are required than the backbone provides then additional feature maps are created
            _len_srcs = len(srcs)
            for lvl in range(_len_srcs, self.num_feature_levels):
                if lvl == _len_srcs:
                    src = self.input_proj[lvl](features[-1].tensors)
                else:
                    src = self.input_proj[lvl](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if self.ref_points_mode == 'bbox':
            reference_points = pred_boxes[:, :, :2]
        else:
            reference_points = None

        if self.query_embedding_mode == 'learned':
            query_embeds = self.query_embed.weight

        # Pass everything to the transformer
        hs, init_reference, _, _, _ = self.transformer(srcs, masks, pos, query_embeds, reference_points)

        outputs_translation = []
        outputs_rotation = []
        bs, _ = pred_classes.shape
        output_idx = torch.where(pred_classes > 0, pred_classes, 0).view(-1)

        # Iterate over the decoder outputs to calculate the intermediate and final outputs (translation and rotation)
        for lvl in range(hs.shape[0]):
            output_rotation = self.rotation_head[lvl](hs[lvl])
            output_translation = self.translation_head[lvl](hs[lvl])
            if self.class_mode == 'specific':
                # Select the correct output according to the predicted class in the class-specific mode
                output_rotation = output_rotation.view(bs * self.n_queries, self.n_classes, -1)
                output_rotation = torch.cat([query[output_idx[i], :] for i, query in enumerate(output_rotation)]).view(
                    bs, self.n_queries, -1)

                output_translation = output_translation.view(bs * self.n_queries, self.n_classes, -1)
                output_translation = torch.cat(
                    [query[output_idx[i], :] for i, query in enumerate(output_translation)]).view(bs, self.n_queries,
                                                                                                  -1)

            output_rotation = self.process_rotation(output_rotation)

            outputs_rotation.append(output_rotation)
            outputs_translation.append(output_translation)

        outputs_rotation = torch.stack(outputs_rotation)
        outputs_translation = torch.stack(outputs_translation)

        out = {'pred_translation': outputs_translation[-1], 'pred_rotation': outputs_rotation[-1],
               'pred_boxes': pred_boxes, 'pred_classes': pred_classes}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_translation, outputs_rotation, pred_boxes, pred_classes)

        return out, n_boxes_per_sample

    def _set_aux_loss(self, outputs_translation, outputs_quaternion, pred_boxes, pred_classes):
        return [{'pred_translation': t, 'pred_rotation': r, 'pred_boxes': pred_boxes, 'pred_classes': pred_classes}
                for t, r in zip(outputs_translation[:-1], outputs_quaternion[:-1])]

    def process_rotation(self, pred_rotation):
        """
        Processes the predicted output rotation given the rotation mode.
        '6d' --> Gram Schmidt
        'quat' or 'silho_quat' --> L2 normalization
        else: Raise error
        """
        if self.rotation_mode == '6d':
            return self.rotation_6d_to_matrix(pred_rotation)
        elif self.rotation_mode in ['quat', 'silho_quat']:
            return F.normalize(pred_rotation, p=2, dim=2)
        else:
            raise NotImplementedError('Rotation mode is not supported')

    def rotation_6d_to_matrix(self, rot_6d):
        """
        Given a 6D rotation output, calculate the 3D rotation matrix in SO(3) using the Gramm Schmit process

        For details: https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf
        """
        bs, n_q, _ = rot_6d.shape
        rot_6d = rot_6d.view(-1, 6)
        m1 = rot_6d[:, 0:3]
        m2 = rot_6d[:, 3:6]

        x = F.normalize(m1, p=2, dim=1)
        z = torch.cross(x, m2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        y = torch.cross(z, x, dim=1)
        rot_matrix = torch.cat((x.view(-1, 3, 1), y.view(-1, 3, 1), z.view(-1, 3, 1)), 2)  # Rotation Matrix lying in the SO(3)
        rot_matrix = rot_matrix.view(bs, n_q, 3, 3)  #.transpose(2, 3)
        return rot_matrix


class SetCriterion(nn.Module):
    """ This class computes the loss for PoET, which consists of translation and rotation for now.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise translation and rotation)
    """
    def __init__(self, matcher, weight_dict, losses, ):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

    def loss_translation(self, outputs, targets, indices):
        """
        Compute the loss related to the translation of pose estimation, namely the mean square error (MSE).
        outputs must contain the key 'pred_translation', while targets must contain the key 'relative_position'
        Position / Translation are expected in [x, y, z] meters
        """
        idx = self._get_src_permutation_idx(indices)
        src_translation = outputs["pred_translation"][idx]
        tgt_translation = torch.cat([t['relative_position'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        n_obj = len(tgt_translation)

        loss_translation = F.mse_loss(src_translation, tgt_translation, reduction='none')
        loss_translation = torch.sum(loss_translation, dim=1)
        loss_translation = torch.sqrt(loss_translation)
        losses = {}
        losses["loss_trans"] = loss_translation.sum() / n_obj
        return losses

    def loss_rotation(self, outputs, targets, indices):
        """
        Compute the loss related to the rotation of pose estimation represented by a 3x3 rotation matrix.
        The function calculates the geodesic distance between the predicted and target rotation.
        L = arccos( 0.5 * (Trace(R\tilde(R)^T) -1)
        Calculates the loss in radiant.
        """
        eps = 1e-6
        idx = self._get_src_permutation_idx(indices)
        src_rot = outputs["pred_rotation"][idx]
        tgt_rot = torch.cat([t['relative_rotation'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        n_obj = len(tgt_rot)

        product = torch.bmm(src_rot, tgt_rot.transpose(1, 2))
        trace = torch.sum(product[:, torch.eye(3).bool()], 1)
        theta = torch.clamp(0.5 * (trace - 1), -1 + eps, 1 - eps)
        rad = torch.acos(theta)
        losses = {}
        losses["loss_rot"] = rad.sum() / n_obj
        return losses

    def loss_quaternion(self, outputs, targets, indices):
        """
        Compute the loss related to the rotation of pose estimation represented in quaternions, namely the quaternion loss
        Q_loss = - log(<q_pred,pred_gt>² + eps), where eps is a small values for stability reasons

        outputs must contain the key 'pred_quaternion', while targets must contain the key 'relative_quaternions'
        Quaternions expected in representation [w, x, y, z]
        """
        eps = 1e-4
        idx = self._get_src_permutation_idx(indices)
        src_quaternion = outputs["pred_rotation"][idx]
        tgt_quaternion = torch.cat([t['relative_quaternions'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        n_obj = len(tgt_quaternion)
        bs, q_dim = tgt_quaternion.shape

        dot_product = torch.mul(src_quaternion, tgt_quaternion)
        dp_sum = torch.sum(dot_product, 1)
        dp_square = torch.square(dp_sum)
        loss_quat = - torch.log(dp_square + eps)

        losses = {}
        losses["loss_rot"] = loss_quat.sum() / n_obj
        return losses

    def loss_silho_quaternion(self, outputs, targets, indices):
        """
        Compute the loss related to the rotation of pose estimation represented in quaternions, namely the quaternion loss
        Q_loss = log(1 - |<q_pred,pred_gt>| + eps), where eps is a small values for stability reasons

        outputs must contain the key 'pred_quaternion', while targets must contain the key 'relative_quaternions'
        Quaternions expected in representation [w, x, y, z]
        """
        eps = 1e-4
        idx = self._get_src_permutation_idx(indices)
        src_quaternion = outputs["pred_rotation"][idx]
        tgt_quaternion = torch.cat([t['relative_quaternions'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        n_obj = len(tgt_quaternion)
        bs, q_dim = tgt_quaternion.shape

        dot_product = torch.mul(src_quaternion, tgt_quaternion)
        dp_sum = torch.sum(dot_product, 1)
        loss_quat = torch.log(1 - torch.abs(dp_sum) + eps)

        losses = {}
        losses["loss_rot"] = loss_quat.sum() / n_obj
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'translation': self.loss_translation,
            'rotation': self.loss_rotation,
            'quaternion': self.loss_quaternion,
            'silho_quaternion': self.loss_silho_quaternion
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets, n_boxes):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            n_boxes: Number of predicted objects per image
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, n_boxes)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, n_boxes)
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            indices = self.matcher(enc_outputs, bin_targets, n_boxes)
            for loss in self.losses:
                kwargs = {}
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = PoET(
        backbone,
        transformer,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        n_classes=args.n_classes,
        bbox_mode=args.bbox_mode,
        ref_points_mode=args.reference_points,
        query_embedding_mode=args.query_embedding,
        rotation_mode=args.rotation_representation,
        class_mode=args.class_mode,
        aux_loss=args.aux_loss,
        backbone_type=args.backbone
    )

    matcher = build_matcher(args)
    weight_dict = {'loss_trans': args.translation_loss_coef, 'loss_rot': args.rotation_loss_coef}

    if args.rotation_representation == '6d':
        losses = ['translation', 'rotation']
    elif args.rotation_representation == 'quat':
        losses = ['translation', 'quaternion']
    elif args.rotation_representation == 'silho_quat':
        losses = ['translation', 'silho_quaternion']
    else:
        raise NotImplementedError('Rotation representation not implemented')

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(matcher, weight_dict, losses)
    criterion.to(device)

    return model, criterion, matcher
