"""This script contains the test options for Deep3DFaceRecon_pytorch
"""

# from .base_options import BaseOptions

import argparse

class Face_Render_Options:
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
    def init_options(self):
        # parser = BaseOptions.initialize(self, parser)  # define shared options
        self.parser.add_argument('--name', type=str, default='face_recon', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--dataset_mode', type=str, default=None,
                            help='chooses how datasets are loaded. [None | flist]')
        self.parser.add_argument('--bfm_folder', type=str, default='./deep3d/BFM')
        self.parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')


        # renderer parameters
        self.parser.add_argument('--focal', type=float, default=1015.)
        self.parser.add_argument('--center', type=float, default=112.)
        self.parser.add_argument('--camera_d', type=float, default=10.)
        self.parser.add_argument('--z_near', type=float, default=5.)
        self.parser.add_argument('--z_far', type=float, default=15.)
        self.parser.add_argument('--use_opengl', type=bool, nargs='?', const=True, default=False, help='use opengl context or not')


        # loss weights
        self.parser.add_argument('--w_feat', type=float, default=0.2, help='weight for feat loss')
        self.parser.add_argument('--w_color', type=float, default=1.92, help='weight for loss loss')
        self.parser.add_argument('--w_reg', type=float, default=3.0e-4, help='weight for reg loss')
        self.parser.add_argument('--w_id', type=float, default=1.0, help='weight for id_reg loss')
        self.parser.add_argument('--w_exp', type=float, default=0.8, help='weight for exp_reg loss')
        self.parser.add_argument('--w_tex', type=float, default=1.7e-2, help='weight for tex_reg loss')
        self.parser.add_argument('--w_gamma', type=float, default=10.0, help='weight for gamma loss')
        self.parser.add_argument('--w_lm', type=float, default=1.6e-3, help='weight for lm loss')
        self.parser.add_argument('--w_reflc', type=float, default=5.0, help='weight for reflc loss')
        self.parser.add_argument('--isTrain', type=bool, default=True)
        self.parser.add_argument('--device', type=int, default=0)
        # opt, _ = self.parser.parse_known_args()
        self.parser.set_defaults(
            focal=1015., center=112., camera_d=10., use_last_fc=False, z_near=5., z_far=15.
        )
        # self.isTrain = False
        return self.parser.parse_args()
    # net structure and parameters
