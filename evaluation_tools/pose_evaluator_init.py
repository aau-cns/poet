# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np
import json
import evaluation_tools.model_tools as model_tools
from evaluation_tools.pose_evaluator import PoseEvaluator
from evaluation_tools.pose_evaluator_lmo import PoseEvaluatorLMO


# Functions to initialize the PoseEvaluator module
def load_classes(path):
    """
    Load the class information from a json file. This file contains a mapping between class ID and class name.
    """
    with open(path, 'r') as f:
        classes = json.load(f)
    return classes


def load_model_info(points):
    """
    Load information about the 3D model from the BOP files
    """
    infos = {}
    extents = 2 * np.max(np.absolute(points), axis=0)
    infos['diameter'] = np.sqrt(np.sum(extents * extents))
    infos['min_x'], infos['min_y'], infos['min_z'] = np.min(points, axis=0)
    infos['max_x'], infos['max_y'], infos['max_z'] = np.min(points, axis=0)
    return infos


def load_models(path, classes):
    """
    Load the 3D model point cloud and store it in a dict.
    """

    with open(path + 'models_info.json', 'r') as f:
        models_info_data = json.load(f)

    models = {}
    models_info = {}

    for cls in classes:
        model_class = classes[cls]
        model_file = "obj_" + f'{int(cls):06d}' + ".ply"
        model = model_tools.load_ply(path + model_file)
        models[model_class] = model
        models[model_class]['pts'] = models[model_class]['pts'] / 1000  # Scale the model to meters.
        models_info[model_class] = models_info_data[cls]
    return models, models_info


def load_model_symmetry(path, classes):
    """
    Load information whether objects are symmetric or not.
    """
    model_symmetry = {}

    with open(path, 'r') as f:
        symmetry_dict = json.load(f)

    for cls in classes:
        model_cls = classes[cls]
        model_symmetry[model_cls] = symmetry_dict[model_cls]

    return model_symmetry


def build_pose_evaluator(args):
    """
    Function to build the Pose Evaluator by loading the 3D point clouds and additional information.
    """
    classes_path = args.dataset_path + args.class_info
    classes = load_classes(classes_path)

    models_path = args.dataset_path + args.models
    models, models_info = load_models(models_path, classes)

    symmetries_path = args.dataset_path + args.model_symmetry
    model_symmetry = load_model_symmetry(symmetries_path, classes)
    classes = [classes[k] for k in classes]
    if args.dataset == 'ycbv':
        evaluator = PoseEvaluator(models, classes, models_info, model_symmetry)
    elif args.dataset == 'lmo':
        evaluator = PoseEvaluatorLMO(models, classes, models_info, model_symmetry)
    else:
        raise ValueError("Unknown dataset.")
    return evaluator