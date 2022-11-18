# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------

import torch

@torch.no_grad()
def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


# TODO: Move these functions to util/quaternion_ops
def quaternion_multiplication(q, p):
    """
    Calculate the multiplication of two quaternions given in [w, x, y, z]
    """
    res = torch.zeros_like(q)
    if q.ndim > 1:
        # Batches of quaternion
        res[:, 0] = q[:, 0] * p[:, 0] - q[:, 1] * p[:, 1] - q[:, 2] * p[:, 2] - q[:, 3] * p[:, 3]
        res[:, 1] = q[:, 0] * p[:, 1] + q[:, 1] * p[:, 0] - q[:, 2] * p[:, 3] - q[:, 3] * p[:, 2]
        res[:, 2] = q[:, 0] * p[:, 2] - q[:, 1] * p[:, 3] + q[:, 2] * p[:, 0] + q[:, 3] * p[:, 1]
        res[:, 3] = q[:, 0] * p[:, 3] + q[:, 1] * p[:, 2] - q[:, 2] * p[:, 1] + q[:, 3] * p[:, 0]
    else:
        # Single quaternion
        res[0] = q[0] * p[0] - q[1] * p[1] - q[2] * p[2] - q[3] * p[3]
        res[1] = q[0] * p[1] + q[1] * p[0] - q[2] * p[3] - q[3] * p[2]
        res[2] = q[0] * p[2] - q[1] * p[3] + q[2] * p[0] + q[3] * p[1]
        res[3] = q[0] * p[3] + q[1] * p[2] - q[2] * p[1] + q[3] * p[0]
    return res


def inverse_quaternion(q):
    """
    Calculate the inverse of a quaternion in [w, x, y, z] notation.
    Assuming normalized quaternions, then inv(q) = conj(q) = [w, -x, -y, -z]
    """
    inv_q = q.detach().clone()
    if q.ndim > 1:
        # Batches of quaternion
        inv_q[:, 1:] *= -1
    else:
        # Single quaternion
        inv_q[1:] *= -1
    return inv_q


@torch.no_grad()
def calc_q_rotation_error(q_gt, q_pred):
    """
    Calculate the angle error between two quaternions
    This function only processes a single quaternion pair // not suited for batches
    Differentiate between tensors and regular arrays
    """
    bs, dim = q_gt.shape
    dot_product = torch.bmm(q_gt.view(bs, 1, dim), q_pred.view(bs, dim, 1)).squeeze()
    angle_diff = torch.rad2deg(torch.acos(2 * torch.square(dot_product) - 1))

    if bs == 1:
        # TODO: Make this nicer
        angle_diff = angle_diff.reshape((1,))

    # # Second approach --> Same result
    # inv_q_pred = inverse_quaternion(q_pred)
    # z = quaternion_multiplication(q_gt, inv_q_pred)
    # angle_diff_2 = torch.rad2deg(2 * torch.acos(torch.abs(z[:, 0])))

    return angle_diff


@torch.no_grad()
def calc_rotation_error(rot_gt, rot_pred):
    """
    Calculaten the geodesic distance between two rotations
    """
    bs, _, _ = rot_gt.shape
    product = torch.bmm(rot_pred, rot_gt.transpose(1, 2))
    trace = torch.sum(product[:, torch.eye(3).bool()], 1)
    angle_diff = torch.rad2deg(torch.acos(0.5 * (trace-1)))

    if bs == 1:
        angle_diff = angle_diff.reshape((1,))

    return angle_diff


@torch.no_grad()
def calc_translation_error(t_gt, t_pred):
    """
    Calculate the L1 error between two translations.
    This function only processes a single quaternion pair // not suited for batches
    Differentiate between tensors and regular arrays
    """

    error = torch.sqrt(torch.square((t_gt - t_pred)).sum(dim=1))
    return error

