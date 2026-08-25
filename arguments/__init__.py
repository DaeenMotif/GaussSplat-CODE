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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

# In train.py, 4 parser groups are aggragated
class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name) # create a custom argument group with a name
        for key, value in vars(self).items(): # key:value pairs of arguments and their values
            shorthand = False
            if key.startswith("_"): # for ModelParams Class below, check parameter names: startwith("_") [e.g. _source_path]
                shorthand = True # we set shorthand true as a Flag and (next line)
                key = key[1:] # remove the '_'
            t = type(value) # value of the argument, e.g. sh_degree value is 3
            value = value if not fill_none else None # just affirmation its not None
            if shorthand: # only is shorthand true
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else: # if type t not boolean
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t) # _source_path -> --source_path
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t) # key = 'sh_degree', value = 3, t = 'int'

    def extract(self, args):
        group = GroupParams() # collect extracted arguments
        '''Example of parsed args (truncated):
        dict_items([
            ('sh_degree', 3),
            ('source_path', '/path/to/dataset'),
            ('model_path', '/path/to/output'),
            ('images', 'images'),
            ('depths', ''),
            ('resolution', -1),
            ('white_background', False),
            ('data_device', 'cuda'),
            ('iterations', 4000),
            ('position_lr_init', 0.00016),
            ('feature_lr', 0.0025),
            ('opacity_lr', 0.025),
            ('percent_dense', 0.01),
            ('lambda_dssim', 0.2),
            ('convert_SHs_python', False),
            ('compute_cov3D_python', False),
            ('debug', False),
            ('antialiasing', False),
            ('ip', '127.0.0.1'),
            ('port', 6009),
        ])
        '''
        for arg in vars(args).items(): 
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1]) # set the named attribute (arg[0]) on a given object (group) to specific value (arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False): # data loading and scene settings: sh, background, resolution
        self.sh_degree = 3 # degree of sh: 0: rgb, 1,2,3 are viewing direction dependent
        self._source_path = "" # Path to the source directory containing dataset
        self.model_path = "" # model saving path
        self._images = "images" # alternative subdirectory for COLMAP images
        self._depths = "" # path to depth; not used for supervision in original 3DGS
        self._resolution = -1 # this specifies the resolution of loaded images (1, 1/2, 1/4, etc.)
        self._white_background = False # default BG set to black; except NeRF-Synthetic
        self.train_test_exp = False 
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path) # source path is absolute
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False # set True for pass with pytorch (not cuda)
        self.compute_cov3D_python = False # set True for pass with pytorch (not cuda)
        self.debug = False # debugging mode
        self.antialiasing = False # low-pass filter (adds 0.3 variance) to prevent aliasing
        # sub-pixel Gaussians from causing aliasing effects 
        super().__init__(parser, "Pipeline Parameters")

# Refer: Section 5 of the paper for details on the optimization parameters
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000 # iterations set by paper
        self.position_lr_init = 0.00016 # lr for 3D gaussian means at start (high) -> exponentially decays
        # # High initial LR allows Gaussians to move quickly to cover empty space
        self.position_lr_final = 0.0000016 # lr for 3D gaussian means at end (low at end)
        self.position_lr_delay_mult = 0.01 # this is the decay multiplier
        # for decay function, look at self.xyz_scheduler_args in gaussian_model.py and utils/general_utils.py
        self.position_lr_max_steps = 30_000 # heuristic setting across all experiments in paper, allow more steps to better optimize
        self.feature_lr = 0.0025 # learning rate for the SH parameters, separate rates for different degrees
        self.opacity_lr = 0.025 # heuristic setting of opacity
        self.scaling_lr = 0.005 # heuristic setting of scaling vector of covariance matrix
        self.rotation_lr = 0.001 # heuristic setting of quaternion vector of covariance matrix
        
        # Exposure compensation learning rate parameters (additional implementation)
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        
        
        self.percent_dense = 0.01 # this is scaling factor for the scene-scale; important for xyz scaling
        self.lambda_dssim = 0.2 # heuristic scale for loss function, this value (0.2) weights the ssim loss for structural learning from viewspace
        self.densification_interval = 100 # densify every 100 iterations
        self.opacity_reset_interval = 3000 # reset opacity every 3K iterations
        self.densify_from_iter = 500 # start densifying after 500 iterations, every 100 iterations, until 15K iterations
        self.densify_until_iter = 15_000 # densify until 15K iterations, then stop densifying
        self.densify_grad_threshold = 0.0002 # max_grad: 0.0002 from code and paper
        
        # Depth Params
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        
        self.random_background = False # set to False 
        self.optimizer_type = "default" # ADAM
        super().__init__(parser, "Optimization Parameters")



def get_combined_args(parser : ArgumentParser):
    #  this is for resuming a run again, given the model_path (e.g. run killed earlier needs to resume)
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args") # load saved config args
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy() # merge the cmd line arguments for config written to output/{run}/cfg_args
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
