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
import sys
from datetime import datetime
import numpy as np
import random

# https://en.wikipedia.org/wiki/Logit
# Applying inverse_sigmoid to derive a logit (similar to wx+b), and then apply logistic regression (which is sigmoid activation)
def inverse_sigmoid(x):
    return torch.log(x/(1-x)) # Obtain logits from raw values: akin to probit function that coverts probability(between 0 and 1) to a score

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution) # resize to a specific resolution
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0 # normalize rgb values
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1) # permute: (C, 
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1) # here step is current iter, and max step is total iters we set
        # what this eq means that lr rate decays exponentially from lr_init to lr_final as step goes from 0 to max_step
        # so, decay is quick at first and then slows down as step approaches max_step
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L): # this function not used: but is the same CUDA implementation, selecting only right upper triangular entries
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda") # makes it positive semi-definite 

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):  # Make symmetric
    return strip_lowerdiag(sym)

def build_rotation(r):
    # r is a N,4 dimensional vector: quaternion (r,x,y,z)
    # normalize the quaternion: Determines that it is 3D Rotation Matrix  https://kaistackr-my.sharepoint.com/personal/mhsung_kaist_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmhsung%5Fkaist%5Fac%5Fkr%2FDocuments%2FCourses%2F2022%5Ffall%5Fcs492j%2Fslides%5Fpdf%2Fkaist%5Fcs492j%5Ffall%5F2022%5Flecture%5F4%2Epdf&parent=%2Fpersonal%2Fmhsung%5Fkaist%5Fac%5Fkr%2FDocuments%2FCourses%2F2022%5Ffall%5Fcs492j%2Fslides%5Fpdf&ga=1
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3]) # calculate l2 norm
    q = r / norm[:, None] # normalize r/|r|

    R = torch.zeros((q.size(0), 3, 3), device='cuda') # create 3X3 rotation for N gaussians

    r = q[:, 0] # separate the normalized quaternion components
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    # from quaternion recover the (3,3) rotation matrix 
    
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r): # Why function not used? TODO: Check
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent): # silent = False
    old_f = sys.stdout
    class F: # None
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
