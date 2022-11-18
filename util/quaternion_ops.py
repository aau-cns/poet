# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np


def quat2rot(q):
    """
        Covert a quaternion into a full three-dimensional rotation matrix.

        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) = (w, x, y, z)

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix.
                 This rotation matrix converts a point in the local reference
                 frame to a point in the global reference frame.
        """
    bs = q.shape[0]
    # Extract the values from Q
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.zeros((bs, 3, 3))
    rot_matrix[:, 0, 0] = r00
    rot_matrix[:, 0, 1] = r01
    rot_matrix[:, 0, 2] = r02
    rot_matrix[:, 1, 0] = r10
    rot_matrix[:, 1, 1] = r11
    rot_matrix[:, 1, 2] = r12
    rot_matrix[:, 2, 0] = r20
    rot_matrix[:, 2, 1] = r21
    rot_matrix[:, 2, 2] = r22

    # rot_matrix = np.array([[r00, r01, r02],
    #                        [r10, r11, r12],
    #                        [r20, r21, r22]])

    return rot_matrix


def rot2quat(rots):
    """
    Convert a rotation matrix into a quaternion
    """
    qs = []
    for rot in rots:
        m00 = rot[0, 0]
        m01 = rot[0, 1]
        m02 = rot[0, 2]
        m10 = rot[1, 0]
        m11 = rot[1, 1]
        m12 = rot[1, 2]
        m20 = rot[2, 0]
        m21 = rot[2, 1]
        m22 = rot[2, 2]
        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                         [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                         [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                         [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

        if q[0] < 0.0:
            np.negative(q, q)
        qs.append(q)
    return np.array(qs)
