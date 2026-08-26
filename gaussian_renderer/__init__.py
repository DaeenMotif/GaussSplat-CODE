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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

'''
Below:
Viewpoint camera: contains the camera image, fovx, fovy along with projection matrix for 3D->2D
Also relative pose of camera from colmap
contains uid
contains the z_far and z_near to clip the view frustum in 3D
'''

# Call from render in train.py
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Input:
        viewpoint camera : One of the CamInfos along with the Initialized Gaussian Model
        separate_sh : do we separate sh components: pc.get_features_dc, pc.get_features_rest
        scale_modifier = 1.0, same scale set for training
        bg_color : Default is [0,0,0] (black) for the rendered image
        pc: point cloud gaussian model with gaussian attributes (mean, cov3D, color, opacity)
        pipe: PipelineParams
        class PipelineParams(ParamGroup):
            self.convert_SHs_python
            self.compute_cov3D_python
            self.debug
            self.antialiasing
        use_trained_exp : for exposure compensation
        Description:
        Set the rasterization settings using unique viewpoint (img+camera) information
        
        
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # IN screenspace,  # tanfovx = 0.5W/focal_length 
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # set the gaussian rasterization settings
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx, # horizontal focal term used to build the 2D projection Jacobian
        tanfovy=tanfovy, # vertical focal term used to build the 2D projection Jacobian
        bg=bg_color, #  background color is added if nothing left to blend
        scale_modifier=scaling_modifier, # 1.0
        viewmatrix=viewpoint_camera.world_view_transform, # #  4X4 world to camera viewpoint matrix
        projmatrix=viewpoint_camera.full_proj_transform, # 4X4 world-to-screenspace transform
        sh_degree=pc.active_sh_degree, # current iterations active SH (which starts at 0 and increases by 1 every 1K iter)
        campos=viewpoint_camera.camera_center, # tensor is size [3]
        prefiltered=False, # asserts the input Gaussians were already frustum-culled (in_frustum call in auxiliary.h)
        debug=pipe.debug,
        antialiasing=pipe.antialiasing # enable anti-aliasing which optimizes opacity differently
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz # [N,3]
    means2D = screenspace_points # [N,3] means2D are first set to 0
    opacity = pc.get_opacity # [N,1] at beginning opacity is 0.1 for all

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer. (Comment written by author)
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python: # default is false
        cov3D_precomp = pc.get_covariance(scaling_modifier) # else calls get_covariance in gaussian_model.py
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer. (Comment written by author)
    shs = None
    colors_precomp = None
    if override_color is None: # it is none
        if pipe.convert_SHs_python: ## pc.get_features: [N,16,3] and transpose(1,2).view(-1,3,(3+1)**2) >> [N,3,16]
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2) # pc.get_features is concatenated SH base 0  and higher SH base features
            # same code below in CUDA: forward.cu:line21 : __device__ glm::vec3 computeColorFromSH
            # viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1) : [N, 3], get_xyz : [N, 3]
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)) # center of 3D Gaussian - center of cam-center
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) # same formula in CUDA
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh: # we dont use this
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else: # we combine the bases
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else: # we use this settings
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        # radii is the set of gaussians radius in screenspace
        # rendered_image: RGB image of same size as GT image
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name) # return learnt exposure
        # multiply each rendered image with its exposure and exposure bias
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None] 

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points, # # In viewspace, we consider 2D projected gaussians
        "visibility_filter" : (radii > 0).nonzero(), # filter selects specific viewspace points' gradients and accumulate over iterations
        "radii": radii, # radii of gaussians
        "depth" : depth_image
        }
    
    return out
