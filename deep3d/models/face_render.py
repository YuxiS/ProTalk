import numpy as np
import torch
import torch.nn as nn
# from mellotron.deep3d import networks
from deep3d.models.bfm import ParametricFaceModel
from deep3d.models.losses import  reflectance_loss, landmark_loss
from deep3d.util.nvdiffrast import MeshRenderer
import  deep3d.models.networks as networks 
import os
import pdb

class Face_render:
    def __init__(self, opt):
        # super().__init__()
        self.visual_names = ['output_vis']
        self.model_names = ['net_recon']
        self.parallel_names = self.model_names + ['renderer']
        self.opt = opt
        self.isTrain = opt.isTrain
        # self.device = torch.device('0')
        # self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        if opt.isTrain==False:
            self.net_recon = networks.define_net_recon(
                net_recon=opt.net_recon, use_last_fc=opt.use_last_fc, init_path=opt.init_path
            )
            # self.device = torch.device(opt.local_rank)
            self.net_recon.to(torch.device(opt.local_rank))

        self.facemodel = ParametricFaceModel(
            bfm_folder=opt.bfm_folder, camera_distance=opt.camera_d, focal=opt.focal, center=opt.center,
            is_train=self.isTrain, default_name=opt.bfm_model
        )
        # self.facemodel.to(self.device)
#         self.facemodel.to('cuda:0')
        self.device = torch.device(opt.local_rank)
        self.facemodel.to(self.device)


        fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
        self.renderer = MeshRenderer(
            rasterize_fov=fov, znear=opt.z_near, zfar=opt.z_far, rasterize_size=int(2 * opt.center),
            use_opengl=opt.use_opengl
        )
        self.renderer.eval()
        # self.loss_names = ['all', 'feat', 'color', 'lm', 'reg', 'gamma', 'reflc']

        # self.net_recog = networks.define_net_recog(
        #     net_recog=opt.net_recog, pretrained_path=opt.net_recog_path
        # )
        # # loss func name: (compute_%s_loss) % loss_name
        # self.compute_feat_loss = perceptual_loss
        # self.comupte_color_loss = photo_loss
        self.compute_lm_loss = landmark_loss
        # self.compute_reg_loss = reg_loss
        self.compute_reflc_loss = reflectance_loss
        # self.parallel_names += ['net_recog']

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_img = input['imgs'].to(self.device) 
        self.atten_mask = input['msks'].to(self.device) if 'msks' in input else None
        self.gt_lm = input['lms'].to(self.device)  if 'lms' in input else None
        self.trans_m = input['M'].to(self.device) if 'M' in input else None
        self.image_paths = input['im_paths'] if 'im_paths' in input else None


    def forward(self, coff):
        pred_vertex, pred_tex, pred_color, pred_lm = \
            self.facemodel.compute_for_render(coff)
        pred_mask, _, pred_face = self.renderer(
            pred_vertex, self.facemodel.face_buf, feat=pred_color)
        pred_lm[:, :, 1] = 224-pred_lm[:, :, 1]
        return pred_lm, pred_tex, pred_mask, pred_face

    def render_img(self, original_img, pred_mask, pred_face):
        """

        Args:
                original_img: [B, C, W, H]
                # img_index:[B, frames_num]
        Returns:
        """
        # pdb.set_trace()
        output_vis = pred_face * pred_mask + (1 - pred_mask) * original_img
        return output_vis

    def compute_losses(self, pred_tex, pred_lm, gt_lm):
        """
        Args:
            input_img:
        Returns:
        Calculate losses, gradients, and update network weights; called in every training iteration
        """
        # assert self.net_recog.training == False
        # trans_m = self.trans_m
        # if not self.opt.use_predef_M:
        # trans_m = estimate_norm_torch(self.pred_lm, input_img.shape[-2])
        # pred_feat = self.net_recog(self.pred_face, trans_m)
        # gt_feat = self.net_recog(input_img, self.trans_m)
        # self.loss_feat = self.opt.w_feat * self.compute_feat_loss(pred_feat, gt_feat)
        # face_mask = self.pred_mask
        # if self.opt.use_crop_face:
        #     face_mask, _, _ = self.renderer(self.pred_vertex, self.facemodel.front_face_buf)
        # face_mask = face_mask.detach()
        # self.loss_color = self.opt.w_color * self.comupte_color_loss(
        #     self.pred_face, input_img, self.atten_mask * face_mask)
        # loss_reg, loss_gamma = self.compute_reg_loss(self.pred_coeffs_dict, self.opt)
        # self.loss_reg = self.opt.w_reg * loss_reg
        # self.loss_gamma = self.opt.w_gamma * loss_gamma

        loss_lm = self.opt.w_lm * self.compute_lm_loss(pred_lm, gt_lm)
        # return loss_lm
        loss_reflc = self.opt.w_reflc * self.compute_reflc_loss(pred_tex, self.facemodel.skin_mask)
        # loss_all = loss_lm + loss_reflc
        loss_all = loss_lm
        return loss_all, loss_lm, loss_reflc

    def inference(self, exp_coff_list):
        """
    
        Args:
            coff_list:[T, 70]
    
        Returns:
    
        """
        from PIL import Image
        import os
        import scipy.io as scio
        self.facemodel.to(self.device)
        res = []
        with torch.no_grad():
            # output_coeff = self.net_recon(self.input_img)
            coff = scio.loadmat('/home/songyifei9/code/prosody/StyleProsody/mellotron/test/000001.mat')
            # pdb.set_trace()
            output_coeff = np.concatenate([np.array(coff['id']),np.array(coff['exp']),
                                        np.array(coff['tex']),np.array(coff['angle']),
                                        np.array(coff['gamma']), np.array(coff['trans'])], axis=1)
            output_coeff = torch.from_numpy(output_coeff).cuda()
            # pdb.set_trace()
            exp_coff_list = exp_coff_list.squeeze(0)    # (T, 70)
            output_coeff = output_coeff.repeat(exp_coff_list.size(0), 1)
            # pdb.set_trace()
            output_coeff[:, 80:144] = exp_coff_list[:, :64]
            output_coeff[:, 224:227] = exp_coff_list[:, 64:67]
            output_coeff[:, 254:] = exp_coff_list[:, 67:]

            for t in range(output_coeff.size(0)):
                coff = output_coeff[t, :].unsqueeze(0)
                pred_vertex, pred_tex, pred_color, pred_lm = \
                    self.facemodel.compute_for_render(coff)
                pred_mask, _, pred_face = self.renderer(
                    pred_vertex, self.facemodel.face_buf, feat=pred_color)

                # recon_shape = pred_vertex  # get reconstructed shape
                # recon_shape[..., -1] = 10 - recon_shape[..., -1]  # from camera space to world space
                # recon_shape = recon_shape.cpu().numpy()[0]
                # recon_color = pred_color.cpu().numpy()[0]
                # tri = self.facemodel.face_buf.cpu().numpy()
                # import trimesh
                # mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri,
                #                        vertex_colors=np.clip(255. * recon_color, 0, 255).astype(np.uint8), process=False)
                # mesh.export('./test/{}.obj'.format(t))
    
                # input_img_numpy = 255. * self.input_img.detach().cpu().permute(0, 2, 3, 1).numpy()
                # output_vis = pred_face * pred_mask + (1 - self.pred_mask) * self.input_img
                # output_vis_numpy_raw = 255. * pred_face.detach().cpu().permute(0, 2, 3, 1).numpy()
                # output_vis_numpy_raw=output_vis_numpy_raw.astype(np.uint8)
                # img = Image.fromarray(output_vis_numpy_raw)
                # img.save('jjjjjjjjj.jpg')
                # raise SystemExit

                res.append(pred_face)
        return res
                # img.save(os.path.join(target_dir, '{}.jpg'.format(t)))

