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

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
import data_utils.samplers as samplers
from data_utils import build_dataset
from engine import train_one_epoch, pose_evaluate, bop_evaluate
from models import build_model
from evaluation_tools.pose_evaluator_init import build_pose_evaluator


def get_args_parser():
    parser = argparse.ArgumentParser('Pose Estimation Transformer', add_help=False)

    # Learning
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--eval_batch_size', default=16, type=int, help='Batch size for evaluation')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # * Backbone
    parser.add_argument('--backbone', default='yolov4', type=str, choices=['yolov4', 'maskrcnn'],
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--backbone_cfg', default='configs/ycbv_yolov4-csp.cfg', type=str,
                        help="Path to the backbone config file to use")
    parser.add_argument('--backbone_weights', default=None, type=str,
                        help="Path to the pretrained weights for the backbone."
                             "None if no weights should be loaded.")
    parser.add_argument('--backbone_conf_thresh', default=0.4, type=float,
                        help="Backbone confidence threshold which objects to keep.")
    parser.add_argument('--backbone_iou_thresh', default=0.5, type=float, help="Backbone IOU threshold for NMS")
    parser.add_argument('--backbone_agnostic_nms', action='store_true',
                        help="Whether backbone NMS should be performed class-agnostic")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # ** PoET configs
    parser.add_argument('--bbox_mode', default='gt', type=str, choices=('gt', 'backbone', 'jitter'),
                        help='Defines which bounding boxes should be used for PoET to determine query embeddings.')
    parser.add_argument('--reference_points', default='bbox', type=str, choices=('bbox', 'learned'),
                        help='Defines whether the transformer reference points are learned or extracted from the bounding boxes')
    parser.add_argument('--query_embedding', default='bbox', type=str, choices=('bbox', 'learned'),
                        help='Defines whether the transformer query embeddings are learned or determined by the bounding boxes')
    parser.add_argument('--rotation_representation', default='6d', type=str, choices=('6d', 'quat', 'silho_quat'),
                        help="Determine the rotation representation with which PoET is trained.")
    parser.add_argument('--class_mode', default='agnostic', type=str, choices=('agnostic', 'specific'),
                        help="Determine whether PoET ist trained class-specific or class-agnostic")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Matcher
    parser.add_argument('--matcher_type', default='pose', choices=['pose'], type=str)
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=1, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Loss coefficients
    # Pose Estimation losses
    parser.add_argument('--translation_loss_coef', default=1, type=float, help='Loss weighing parameter for the translation')
    parser.add_argument('--rotation_loss_coef', default=1, type=float, help='Loss weighing parameter for the rotation')

    # dataset parameters
    parser.add_argument('--dataset', default='ycbv', type=str, choices=('ycbv', 'lmo'),
                        help="Choose the dataset to train/evaluate PoET on.")
    parser.add_argument('--dataset_path', default='/data', type=str,
                        help='Path to the dataset ')
    parser.add_argument('--train_set', default="train", type=str, help="Determine on which dataset split to train")
    parser.add_argument('--eval_set', default="test", type=str, help="Determine on which dataset split to evaluate")
    parser.add_argument('--synt_background', default='/background/', type=str,
                        help="Directory containing the background images from which to sample")
    parser.add_argument('--n_classes', default=21, type=int, help="Number of classes present in the dataset")
    parser.add_argument('--jitter_probability', default=0.5, type=float,
                        help='If bbox_mode is set to jitter, this value indicates the probability '
                             'that jitter is applied to a bounding box.')
    parser.add_argument('--rgb_augmentation', action='store_true',
                        help='Activate image augmentation for training pose estimation.')
    parser.add_argument('--grayscale', action='store_true', help='Activate grayscale augmentation.')

    # * Evaluator
    parser.add_argument('--eval_interval', type=int, default=10,
                        help="Epoch interval after which the current model is evaluated")
    parser.add_argument('--class_info', type=str, default='/annotations/classes.json',
                        help='path to .txt-file containing the class names')
    parser.add_argument('--models', type=str, default='/models_eval/',
                        help='path to a directory containing the classes models')
    parser.add_argument('--model_symmetry', type=str, default='/annotations/symmetries.json',
                        help='path to .json-file containing the class symmetries')

    # * Misc
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--save_interval', default=5, type=int,
                        help="Epoch interval after which the current checkpoint will be stored")
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Run model in evaluation mode')
    parser.add_argument('--eval_bop', action='store_true', help="Run model in BOP challenge evaluation mode")
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser


def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build the model and evaluator
    model, criterion, matcher = build_model(args)
    model.to(device)

    pose_evaluator = build_pose_evaluator(args)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # Build the dataset for training and validation
    dataset_train = build_dataset(image_set=args.train_set, args=args)
    dataset_val = build_dataset(image_set=args.eval_set, args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.eval_batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n,
                                                                                                   args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    output_dir = Path(args.output_dir)
    # Load checkpoint
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler
            #  (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print(
                    'Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

    # Evaluate the models performance
    if args.eval:
        if args.resume:
            eval_epoch = checkpoint['epoch']
        else:
            eval_epoch = None

        pose_evaluate(model, matcher, pose_evaluator, data_loader_val, args.eval_set, args.bbox_mode,
                      args.rotation_representation, device, args.output_dir, eval_epoch)
        return

    # Evaluate the model for the BOP challenge
    if args.eval_bop:
        bop_evaluate(model, matcher, data_loader_val, args.eval_set, args.bbox_mode,
                     args.rotation_representation, device, args.output_dir)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # Do evaluation on the validation set every n epochs
        if epoch % args.eval_interval == 0:
            pose_evaluate(model, matcher, pose_evaluator, data_loader_val, args.eval_set, args.bbox_mode,
                          args.rotation_representation, device, args.output_dir, epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Evaluate final trained model')
    eval_start_time = time.time()
    pose_evaluate(model, matcher, pose_evaluator, data_loader_val, args.eval_set, args.bbox_mode,
                  args.rotation_representation, device, args.output_dir)
    eval_total_time = time.time() - eval_start_time
    eval_total_time_str = str(datetime.timedelta(seconds=int(eval_total_time)))
    print('Evaluation time {}'.format(eval_total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PoET training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
