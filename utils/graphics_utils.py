#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple): # Class definition for PointCloid
    points : np.array
    colors : np.array
    normals : np.array

# unused function
def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

# Unsued function for World-3D system to Camera-3D system
def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

# used in scene/cameras.py and dataset_readers.py
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0): # transforms world coord systems points to 3D points in camera's own coord system
    Rt = np.zeros((4, 4)) # initialize the [R|t]] pose
    Rt[:3, :3] = R.transpose() # make the world to camera space extrinsic, R is stored transposed due to 'glm' in CUDA code
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0 # Rt transforms world points to camera's own coord system

    C2W = np.linalg.inv(Rt) # its inverse to get the camera pose again (C2W)
    cam_center = C2W[:3, 3] # translation vector t is the cam center here
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W) # reverse again to Rt (W2C) for cam-coord system
    return np.float32(Rt)

# Perspective projection
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    # defining the view frustum
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
