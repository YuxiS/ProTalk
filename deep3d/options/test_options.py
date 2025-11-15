"""This script contains the test options for Deep3DFaceRecon_pytorch
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--dataset_mode', type=str, default=None, help='chooses how datasets are loaded. [None | flist]')
        parser.add_argument('--img_folder', type=str, default='datasets/examples', help='folder for test images.')
        parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'],
                            help='network structure')
        parser.add_argument('--init_path', type=str, default='checkpoints/init_model/resnet50-0676ba61.pth')
        # parser.add_argument('--use_last_fc', type=util.str2bool, nargs='?', const=True, default=False,
        #                     help='zero initialize the last fc')
        parser.add_argument('--bfm_folder', type=str, default='BFM')
        parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

        # renderer parameters
        parser.add_argument('--focal', type=float, default=1015.)
        parser.add_argument('--center', type=float, default=112.)
        parser.add_argument('--camera_d', type=float, default=10.)
        parser.add_argument('--z_near', type=float, default=5.)
        parser.add_argument('--z_far', type=float, default=15.)
        # parser.add_argument('--use_opengl', type=util.str2bool, nargs='?', const=True, default=False,
        #                     help='use opengl context or not')

        # if is_train:
        # training parameters
        # parser.add_argument('--net_recog', type=str, default='r50', choices=['r18', 'r43', 'r50'],
        #                     help='face recog network structure')
        # parser.add_argument('--net_recog_path', type=str,
        #                     default='checkpoints/recog_model/ms1mv3_arcface_r50_fp16/backbone.pth')
        # parser.add_argument('--use_crop_face', type=util.str2bool, nargs='?', const=True, default=False,
        #                     help='use crop mask for photo loss')
        # parser.add_argument('--use_predef_M', type=util.str2bool, nargs='?', const=True, default=False,
        #                     help='use predefined M for predicted face')

        # augmentation parameters
        # parser.add_argument('--shift_pixs', type=float, default=10., help='shift pixels')
        # parser.add_argument('--scale_delta', type=float, default=0.1, help='delta scale factor')
        # parser.add_argument('--rot_angle', type=float, default=10., help='rot angles, degree')

        # loss weights
        parser.add_argument('--w_feat', type=float, default=0.2, help='weight for feat loss')
        parser.add_argument('--w_color', type=float, default=1.92, help='weight for loss loss')
        parser.add_argument('--w_reg', type=float, default=3.0e-4, help='weight for reg loss')
        parser.add_argument('--w_id', type=float, default=1.0, help='weight for id_reg loss')
        parser.add_argument('--w_exp', type=float, default=0.8, help='weight for exp_reg loss')
        parser.add_argument('--w_tex', type=float, default=1.7e-2, help='weight for tex_reg loss')
        parser.add_argument('--w_gamma', type=float, default=10.0, help='weight for gamma loss')
        parser.add_argument('--w_lm', type=float, default=1.6e-3, help='weight for lm loss')
        parser.add_argument('--w_reflc', type=float, default=5.0, help='weight for reflc loss')

        opt, _ = parser.parse_known_args()
        parser.set_defaults(
            focal=1015., center=112., camera_d=10., use_last_fc=False, z_near=5., z_far=15.
        )
        # if is_train:
        #     parser.set_defaults(
        #         use_crop_face=True, use_predef_M=False
        #     )
        return parser
        # parser.add_argument('--use_opengl', type=bool, default=False)

        # Dropout and Batchnorm has different behavior during training and test.
        self.isTrain = False
        return parser
    # net structure and parameters
