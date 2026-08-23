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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation): # why is it not used? # TODO: Check
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            print(symm)
            return symm
        
        # exponential activation for scaling, to ensure it is positive and smooth
        self.scaling_activation = torch.exp
        # log scale applied to scale vector at new initialization during splitting
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        # S5.1: sigmoid activation for opacity, to ensure it is in [0,1)
        # Obtain smooth gradients (S-shaped curve and continuous)
        self.opacity_activation = torch.sigmoid
        
        # Inverse opacity activation to initialise opacity values at initialization
        self.inverse_opacity_activation = inverse_sigmoid

        # what is rotational_activation? Normalizing the quaternions
        # Unit quaternion represent pure rotation in 3D space; also compact and numerically stable representation of rotation
        # avoid gimbal lock problem
        # Wiki: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation ()
        self.rotation_activation = torch.nn.functional.normalize


    # the initialization of G-attributes
    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0) # gaussian positions
        self._features_dc = torch.empty(0) # gaussian base color of sh
        self._features_rest = torch.empty(0) # gaussian other higher degree sh
        self._scaling = torch.empty(0) # scale vector
        self._rotation = torch.empty(0) # quaternion rotation vector
        self._opacity = torch.empty(0) # 1D opacity of gaussian
        self.max_radii2D = torch.empty(0) # 2D pixel radius of 2D gaussian
        self.xyz_gradient_accum = torch.empty(0) # # what is this parameter of gaussians
        self.denom = torch.empty(0) # counter for a 2D/3D gaussian 
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    # for saving to checkpoint, check train.py line:201
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args): # load the checkpoint and the saved tensors
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict) # load the checkpoint

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc # diffuse color (sh base 0)
        features_rest = self._features_rest # higher bases - view dependent effects
        return torch.cat((features_dc, features_rest), dim=1) # both are concatenated and used in gaussian_renderer/__init__.py
        # features_dc: [N,1,3], features_rest: [N, 15, 3] >> concat at dim=1 >> shape [N, 16, 3]
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name): # TODO: Check
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    # Check train.py line 95-98: every 1K iter, increase degree to 1 at 1K, 2 at 2K, 3 at 3K, and then keep it at 3
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale # spatial_lr_scale = 5.779 for T&T truck
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda() # consttuct tensor of 3D points from point cloud
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda()) # pcd colors are normalized rgb [0,1]
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # shape features: [N, 3, 16]
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0 # this line is redundant, alrdy initialized to 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0]) # the number of points in the initial point cloud

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001) # mean squared distance using KNN
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3) # Start as isotropic splats (uniform scale in all direction)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1 # Initialize as identity quaternion (1, 0, 0, 0)
        
        # torch.ones((fused_point_cloud.shape[0], 1) give (N, 1) tensor of 1s
        
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        # opacities now all are init to -2.1972246170043945; sigmoid of -2.1972246170043945 is 0.1, which is the initial opacity value for all gaussians
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) # separated from dc to learn at diff rate
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda") # all 0 initialized w/ shape [N]
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense # this is 0.01
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # gradient accumulator used for densification
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # [N, 1] # 

        '''
            Different Parameters have different learning rates
            xyz learning scale includes to the scene scale factors
            _features_dc: diffuse color (SH base = 0)
            _features_rest: higher SH bases (1,2,3): learning is set slower to prevent
            Mentioned in paper: sH coeff sensitive to the lack of angular information
        '''
        
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"}, # xyz dim [N,3]
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},# SH=0 (diffuse color) dim [N,1,3] # 3 is spatial dim xyz
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},# SH>0 dim [N, 15, 3]
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},# opacity dim [N,1]
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"}, # # scaling dim [N,3]
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"} # rotation is quaternion of dim [N,4]
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15) # not enabled for this study
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps) # this is applied for xyz learning (3D gaussian locations are off at beginning)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration) # for gaussian means set iteration based exponential decay-based learning
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path)) # set the path to save the pointcloud

        xyz = self._xyz.detach().cpu().numpy() # detach 3D gaussian position tensor from computation graph (no grad)
        normals = np.zeros_like(xyz) # init normals to 0 for 3D Gaussians
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() # detach the cr channel; a
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() # detach the residual sh channel
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # Resetting Opacity to 0 every 3000 iterations to remove floaters stuck close to cameras
    # Check https://github.com/graphdeco-inria/gaussian-splatting/issues/556 and understanding
    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01)) # every 3000 iters make all Gaussians opacities (0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity") # replace_tensor_to_optimizer zeros out Adam's state for this parameter
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False): # loading a pretrained point cloud from 3DGS
        plydata = PlyData.read(path)
        if use_train_test_exp: # exposure flag: Set False
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1) # stack the set of 3D points
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Loading for GPU
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name): # replace the optimizable param and reset Adam's momentum for that param
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor) # zero-out running mean of gradients
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]] # del old param
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True)) # set new param
                self.optimizer.state[group['params'][0]] = stored_state # with zero running mean of gradients

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask): # Pruning is erasing them from the optimizer, mask to keep non-prunable gaussians
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None) # 
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask] # get running mean of gradients for masked gaussians
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]] # del old param
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True))) # newly init masked gaussians
                self.optimizer.state[group['params'][0]] = stored_state # set their old running mean of gradients

                optimizable_tensors[group["name"]] = group["params"][0] # register
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # prune: if a Gaussian becomes too transparent, remove it
    def prune_points(self, mask): # function for pruning/killing Gs based on boolean mask of shape [N]; True - these gaussians should be deleted
        valid_points_mask = ~mask # invert to keep valid Gaussians (see function above it)
        optimizable_tensors = self._prune_optimizer(valid_points_mask) # 
        # reassign all parameter to the remaining gaussians
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]
    
    def cat_tensors_to_optimizer(self, tensors_dict): # <scene.gaussian_model.GaussianModel object>
        # During clone or split, we make new gaussians tensor_dict
        optimizable_tensors = {} #
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0) # concat them to other params, but w/ 0 running mean of grad
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]] # del old params, make new ones and set stored states 
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d) # make new gaussians optimizable params
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # reset grad accumulation
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # and also reset collection of gradient accumulating gaussians
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    # split: if Gaussians have high gradient and is still too large, replace it with smaller children    
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0] # get the number of gaussians
        
        padded_grad = torch.zeros((n_init_points), device="cuda") # make the gradient tensor for all points
        padded_grad[:grads.shape[0]] = grads.squeeze() # for points with gradients, fill in; rest have 0 gradient
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False) # make a boolean mask to filter points with gradients > threshold (threshold set in paper)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent) # 
        # for them with exceeding gradient threshold, split if largest scale axis exceeds threshold
        
        # For selected_pts_mask to be split
        stds = self.get_scaling[selected_pts_mask].repeat(N,1) # get the scaling vectors of selected gaussains
        means = torch.zeros((stds.size(0), 3),device="cuda") # sample gaussians from center
        samples = torch.normal(mean=means, std=stds) # create new samples with different covariance
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1) # make 3x3 rotation matrices from quaternions
        # from selected points to be split, offset them by rotation with created samples
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N)) # N=2 means two new gaussians, 0.8*(N=2) = 1.6 {Mentioned in paper S5.2; scale vector is divided by 1.6}
        
        
        # Copy the parent's rotation, color, and opacity to each child
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        # And reset the gradient accumulation for densification and cat_tensors_to_optimizer
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        # Prune the old parent gaussians that were splitted
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    # clone: if a Gaussian has high gradient and is already small, duplicate it nearby
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        # mask to select points with gradient above threshold for densify and clone
        # and also maximum axial length less than some size threshold
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # selected_points[0] = number of points = _xyz.shape[0]
        
        # clones initialized with same parameter values 
        new_xyz = self._xyz[selected_pts_mask] 
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        # but adam momentum is set to 0 for clones, similar to splits
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom # Shape: [N, 1]
        grads[grads.isnan()] = 0.0 # 0 gradient across some gaussians

        self.tmp_radii = radii
        # max_grad: 0.0002 from code and paper
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze() # mask to prune low opacity < threshold Gaussians
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size # too large viewspace footprint -> prune (select by mask)
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent # scale too large in worldspace -> prune
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws) # OR: prune if any of the condition is true
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None # clean up temporary radii storage

        torch.cuda.empty_cache() # free gpu memory of deleted gaussians

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # update_filter: boolean tensor of shape [P, 1] w/ index to viewspace_point_tensor:[N, 3]
        #  P < N, and filter selects specific viewspace points' gradients and accumulate over iterations
        # In viewspace, we consider 2D projected gaussians, so depth is not used; gradient at (x,y), hence grad[update_filter,:2]
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1 # counter for a 2D/3D gaussian for its update filter, how many times it appeared
