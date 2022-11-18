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
Train and eval functions used in main.py
"""
import math
import os
import sys
import time
import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import util.misc as utils
from util.quaternion_ops import quat2rot
from data_utils.data_prefetcher import data_prefetcher

from evaluation_tools.metrics import get_src_permutation_idx, calc_rotation_error, calc_translation_error


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('position_loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('rotation_loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):

        outputs, n_boxes_per_sample = model(samples, targets)
        loss_dict = criterion(outputs, targets, n_boxes_per_sample)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(position_loss=loss_dict_reduced['loss_trans'])
        metric_logger.update(rotation_loss=loss_dict_reduced['loss_rot'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def pose_evaluate(model, matcher, pose_evaluator, data_loader, image_set, bbox_mode, rotation_mode, device, output_dir, epoch=None):
    """
    Evaluate PoET on the whole dataset, calculate the evaluation metrics and store the final performance.
    """
    model.eval()
    matcher.eval()

    # Reset pose evaluator to be empty
    pose_evaluator.reset()

    # Check whether the evaluation folder exists, otherwise create it
    if epoch is not None:
        output_eval_dir = output_dir + "/eval_" + image_set + "_" + bbox_mode + "_" + str(epoch) + "/"
    else:
        output_eval_dir = output_dir + "/eval_" + image_set + "_" + bbox_mode + "/"
    Path(output_eval_dir).mkdir(parents=True, exist_ok=True)

    print("Process validation dataset:")
    n_images = len(data_loader.dataset.ids)
    bs = data_loader.batch_size
    start_time = time.time()
    processed_images = 0
    for samples, targets in data_loader:
        batch_start_time = time.time()
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs, n_boxes_per_sample = model(samples, targets)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Extract final predictions and store them
        indices = matcher(outputs_without_aux, targets, n_boxes_per_sample)
        idx = get_src_permutation_idx(indices)

        pred_translations = outputs_without_aux["pred_translation"][idx].detach().cpu().numpy()
        pred_rotations = outputs_without_aux["pred_rotation"][idx].detach().cpu().numpy()

        if rotation_mode in ['quat', 'silho_quat']:
            pred_rotations = quat2rot(pred_rotations)

        tgt_translations = torch.cat([t['relative_position'][i] for t, (_, i) in zip(targets, indices)], dim=0).detach().cpu().numpy()
        tgt_rotations = torch.cat([t['relative_rotation'][i] for t, (_, i) in zip(targets, indices)], dim=0).detach().cpu().numpy()

        obj_classes_idx = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0).detach().cpu().numpy()
        intrinsics = torch.cat([t['intrinsics'][i] for t, (_, i) in zip(targets, indices)], dim=0).detach().cpu().numpy()
        img_files = [data_loader.dataset.coco.loadImgs(t["image_id"].item())[0]['file_name'] for t, (_, i) in zip(targets, indices) for _ in range(0, len(i))]

        # Iterate over all predicted objects and save them in the pose evaluator
        for cls_idx, img_file, intrinsic, pred_translation, pred_rotation, tgt_translation, tgt_rotation in \
                zip(obj_classes_idx, img_files, intrinsics, pred_translations, pred_rotations, tgt_translations, tgt_rotations):
            cls = pose_evaluator.classes[cls_idx - 1]
            pose_evaluator.poses_pred[cls].append(
                np.concatenate((pred_rotation, pred_translation.reshape(3, 1)), axis=1))
            pose_evaluator.poses_gt[cls].append(
                np.concatenate((tgt_rotation, tgt_translation.reshape(3, 1)), axis=1))
            pose_evaluator.poses_img[cls].append(img_file)
            pose_evaluator.num[cls] += 1
            pose_evaluator.camera_intrinsics[cls].append(intrinsic)

        batch_total_time = time.time() - batch_start_time
        batch_total_time_str = str(datetime.timedelta(seconds=int(batch_total_time)))
        processed_images = processed_images + len(targets)
        remaining_images = n_images - processed_images
        remaining_batches = remaining_images / bs
        eta = batch_total_time * remaining_batches
        eta_str = str(datetime.timedelta(seconds=int(eta)))
        print("Processed {}/{} \t Batch Time: {} \t ETA: {}".format(processed_images, n_images, batch_total_time_str, eta_str))
    # At this point iterated over all validation images and for each object the result is fed into the pose evaluator
    total_time = time.time() - start_time
    time_per_img = total_time / n_images
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    time_per_img_str = str(datetime.timedelta(seconds=int(time_per_img)))
    print("Network Processing Time\nTotal Time: {}\t\tImages: {}\t\ts/img: {}".format(total_time_str, n_images, time_per_img_str))
    print("Start results evaluation")
    start_time = time.time()
    print("Start Calculating ADD")
    pose_evaluator.evaluate_pose_add(output_eval_dir)
    print("Start Calculating ADD-S")
    pose_evaluator.evaluate_pose_adi(output_eval_dir)
    print("Start Calculating ADD(-S)")
    pose_evaluator.evaluate_pose_adds(output_eval_dir)
    print("Start Calculating Average Translation Error")
    pose_evaluator.calculate_class_avg_translation_error(output_eval_dir)
    print("Start Calculating Average Rotation Error")
    pose_evaluator.calculate_class_avg_rotation_error(output_eval_dir)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time: {}".format(total_time_str))
    return


@torch.no_grad()
def bop_evaluate(model, matcher, data_loader, image_set, bbox_mode, rotation_mode, device, output_dir):
    """
    Evaluate PoET on the dataset and store the results in the BOP format
    """
    model.eval()
    matcher.eval()

    output_eval_dir = output_dir + "/bop_" + bbox_mode + "/"
    Path(output_eval_dir).mkdir(parents=True, exist_ok=True)

    out_csv_file = open(output_eval_dir + 'ycbv.csv', 'w')
    out_csv_file.write("scene_id,im_id,obj_id,score,R,t,time")
    n_images = len(data_loader.dataset.ids)

    # CSV format: scene_id, im_id, obj_id, score, R, t, time
    counter = 1
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        pred_start_time = time.time()
        outputs, n_boxes_per_sample = model(samples, targets)
        pred_end_time = time.time() - pred_start_time
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        indices = matcher(outputs_without_aux, targets, n_boxes_per_sample)
        idx = get_src_permutation_idx(indices)

        pred_translations = outputs_without_aux["pred_translation"][idx].detach().cpu().numpy()
        pred_rotations = outputs_without_aux["pred_rotation"][idx].detach().cpu().numpy()

        if rotation_mode in ['quat', 'silho_quat']:
            pred_rotations = quat2rot(pred_rotations)

        obj_classes_idx = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)],
                                    dim=0).detach().cpu().numpy()

        img_files = [data_loader.dataset.coco.loadImgs(t["image_id"].item())[0]['file_name'] for t, (_, i) in
                     zip(targets, indices) for _ in range(0, len(i))]

        for cls_idx, img_file, pred_translation, pred_rotation in zip(obj_classes_idx, img_files, pred_translations, pred_rotations):
            file_info = img_file.split("/")
            scene_id = int(file_info[1])
            img_id = int(file_info[3][:file_info[3].rfind(".")])
            obj_id = cls_idx
            score = 1.0
            csv_str = "{},{},{},{},{} {} {} {} {} {} {} {} {}, {} {} {}, {}\n".format(scene_id, img_id, obj_id, score,
                                                                                    pred_rotation[0, 0], pred_rotation[0, 1], pred_rotation[0, 2],
                                                                                    pred_rotation[1, 0], pred_rotation[1, 1], pred_rotation[1, 2],
                                                                                    pred_rotation[2, 0], pred_rotation[2, 1], pred_rotation[2, 2],
                                                                                    pred_translation[0] * 1000, pred_translation[1] * 1000, pred_translation[2] * 1000,
                                                                                    pred_end_time)
            out_csv_file.write(csv_str)
        print("Processed {}/{}".format(counter, n_images))
        counter += 1

    out_csv_file.close()