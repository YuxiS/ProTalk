import random
import torch
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import torchvision
import numpy as np
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy

class GANLogger:
    def __init__(self, logdir):
        super(GANLogger, self).__init__()
        self.writer = SummaryWriter(logdir)

    def log_training(self, coeff_exp, coeff_exp_delta,
                            coeff_pose, coeff_pose_delta, # iteration):
                            loss_gan_D, loss_GAN_G,
                            iteration):
                            
        """

        Args:
            image_real: 8 frames [8, C, H, W] tensor
            image_render: [8, C, H, W]
            image_retail:  8 frames [8, C, H, W] tensor
        Returns:
        """
        # coeff
        self.writer.add_scalar('Training coeff exp', coeff_exp, iteration)
        self.writer.add_scalar('Training coeff exp delta', coeff_exp_delta, iteration)
        # 3dmm
        self.writer.add_scalar('Training coeff pose', coeff_pose, iteration)
        self.writer.add_scalar('Training coeff pose delta', coeff_pose_delta, iteration)
        # img
        self.writer.add_scalar('Training GAN D ', loss_gan_D, iteration)
        self.writer.add_scalar('Training GAN G', loss_GAN_G, iteration)


    def log_val(self, coeff_exp, coeff_exp_delta,
                    coeff_pose, coeff_pose_delta,
                    face_show, real_face,
                    iteration):

        self.writer.add_scalar('Val exp', coeff_exp, iteration)
        self.writer.add_scalar('Val exp delta', coeff_exp_delta, iteration)
        # 3dmm
        self.writer.add_scalar('Val pose', coeff_pose, iteration)
        self.writer.add_scalar('Val pose delta', coeff_pose_delta, iteration)
        # img
        # self.writer.add_scalar('Val img lpips ', img_lpips, iteration)
        # self.writer.add_scalar('Val img l1', img_l1, iteration)
        # # ssim
        # self.writer.add_scalar('Val render ssim', ssim_render, iteration)
        # self.writer.add_scalar('Val retail ssim', ssim_retail, iteration)
        # images = torch.cat([image_real, image_render, image_retail], dim=0)
        # images = images*0.5+0.5
        # images = torchvision.utils.make_grid(images)
        
        
        # self.writer.add_image('Real_image, render_image, retail_image', images, iteration)
        pred_face = face_show*0.5+0.5
        pred_face = torchvision.utils.make_grid(pred_face)
        self.writer.add_image("Pred Face", pred_face, iteration)

        real_face = real_face*0.5+0.5
        real_face = torchvision.utils.make_grid(real_face)
        self.writer.add_image("Real Face", real_face, iteration)
        # faces = torch.cat([real_face, face_show], dim=0)
        # faces = faces*0.5+0.5
        # faces = torchvision.utils.make_grid(faces)
        # self.writer.add_image("real face and render face", faces, iteration)

        # masks = torch.cat([real_mask, mask_show], dim=0)
        # masks = torchvision.utils.make_grid(masks)
        # self.writer.add_image('Real and pred Mask', masks, iteration)

class Logger:
    def __init__(self, logdir):
        super(Logger, self).__init__()
        self.writer = SummaryWriter(logdir)

    def log_training(self, coeff_exp, coeff_exp_delta,
                            coeff_pose, coeff_pose_delta, 
                            loss_gan, loss_exp, imgs,
                            iteration):
                            # loss_gan_D, loss_GAN_G,
                            # iteration):
                            
        """

        Args:
            image_real: 8 frames [8, C, H, W] tensor
            image_render: [8, C, H, W]
            image_retail:  8 frames [8, C, H, W] tensor
        Returns:
        """
        # coeff
        self.writer.add_scalar('Training coeff exp', coeff_exp, iteration)
        self.writer.add_scalar('Training coeff exp delta', coeff_exp_delta, iteration)
        # 3dmm
        self.writer.add_scalar('Training coeff angle', coeff_pose, iteration)
        self.writer.add_scalar('Training coeff trans', coeff_pose_delta, iteration)

        for key, value in loss_gan.items():
            self.writer.add_scalar('Training {}'.format(key), value, iteration)
        for key, value in loss_exp.items():
            self.writer.add_scalar('Training {}'.format(key), value, iteration)

        imgs = imgs*0.5+0.5
        imgs = torchvision.utils.make_grid(imgs)
        self.writer.add_image("Real Face", imgs, iteration)

    def log_val(self, coeff_exp, coeff_exp_delta,
                    coeff_pose, coeff_pose_delta,
                    face_show, real_face,
                    iteration):

        self.writer.add_scalar('Val exp', coeff_exp, iteration)
        self.writer.add_scalar('Val exp delta', coeff_exp_delta, iteration)
        # 3dmm
        self.writer.add_scalar('Val pose', coeff_pose, iteration)
        self.writer.add_scalar('Val pose delta', coeff_pose_delta, iteration)
        # img
        # self.writer.add_scalar('Val img lpips ', img_lpips, iteration)
        # self.writer.add_scalar('Val img l1', img_l1, iteration)
        # # ssim
        # self.writer.add_scalar('Val render ssim', ssim_render, iteration)
        # self.writer.add_scalar('Val retail ssim', ssim_retail, iteration)
        # images = torch.cat([image_real, image_render, image_retail], dim=0)
        # images = images*0.5+0.5
        # images = torchvision.utils.make_grid(images)
        
        
        # self.writer.add_image('Real_image, render_image, retail_image', images, iteration)
        pred_face = face_show*0.5+0.5
        pred_face = torchvision.utils.make_grid(pred_face)
        self.writer.add_image("Pred Face", pred_face, iteration)

        real_face = real_face*0.5+0.5
        real_face = torchvision.utils.make_grid(real_face)
        self.writer.add_image("Real Face", real_face, iteration)
        # faces = torch.cat([real_face, face_show], dim=0)
        # faces = faces*0.5+0.5
        # faces = torchvision.utils.make_grid(faces)
        # self.writer.add_image("real face and render face", faces, iteration)

        # masks = torch.cat([real_mask, mask_show], dim=0)
        # masks = torchvision.utils.make_grid(masks)
        # self.writer.add_image('Real and pred Mask', masks, iteration)


class ImgLogger:
    def __init__(self, logdir):
        super().__init__()
        self.writer = SummaryWriter(logdir)

    def log_training(self, coeff_exp, coeff_exp_delta,
                        coeff_pose, coeff_pose_delta,
                        exp_dict, img,
                        iteration):
                            
        """

        Args:
            image_real: 8 frames [8, C, H, W] tensor
            image_render: [8, C, H, W]
            image_retail:  8 frames [8, C, H, W] tensor
        Returns:
        """
        # coeff
        self.writer.add_scalar("Train exp coeff loss", coeff_exp, iteration)
        self.writer.add_scalar('Train exp coeff delta loss', coeff_exp_delta, iteration)
        # img
        self.writer.add_scalar('Train angle coeff loss', coeff_pose, iteration)
        self.writer.add_scalar('Train trans coeff loss', coeff_pose_delta, iteration)
        # 
        for key, value in exp_dict.items():
            self.writer.add_scalar(key, np.mean(value[-20:]), iteration)
        
        pred_face = img*0.5+0.5
        pred_face = torchvision.utils.make_grid(pred_face)
        self.writer.add_image("Gen Img", pred_face, iteration)

        # real_face = real_face*0.5+0.5
        # real_face = torchvision.utils.make_grid(real_face)
        # self.writer.add_image("Real Face", real_face, iteration)


    def log_val(self, img_l1, img_lpips,
                    face_show, real_face,
                    iteration):

        # self.writer.add_scalar('Val exp', coeff_exp, iteration)
        # self.writer.add_scalar('Val exp delta', coeff_exp_delta, iteration)
        # # 3dmm
        # self.writer.add_scalar('Val pose', coeff_pose, iteration)
        # self.writer.add_scalar('Val pose delta', coeff_pose_delta, iteration)
        # img
        self.writer.add_scalar('Val img lpips ', img_lpips, iteration)
        self.writer.add_scalar('Val img l1', img_l1, iteration)
        # # ssim
        # self.writer.add_scalar('Val render ssim', ssim_render, iteration)
        # self.writer.add_scalar('Val retail ssim', ssim_retail, iteration)
        # images = torch.cat([image_real, image_render, image_retail], dim=0)
        # images = images*0.5+0.5
        # images = torchvision.utils.make_grid(images)
        
        
        # self.writer.add_image('Real_image, render_image, retail_image', images, iteration)
        pred_face = face_show*0.5+0.5
        pred_face = torchvision.utils.make_grid(pred_face)
        self.writer.add_image("Pred Face", pred_face, iteration)

        real_face = real_face*0.5+0.5
        real_face = torchvision.utils.make_grid(real_face)
        self.writer.add_image("Real Face", real_face, iteration)
        # faces = torch.cat([real_face, face_show], dim=0)
        # faces = faces*0.5+0.5
        # faces = torchvision.utils.make_grid(faces)
        # self.writer.add_image("real face and render face", faces, iteration)

        # masks = torch.cat([real_mask, mask_show], dim=0)
        # masks = torchvision.utils.make_grid(masks)
        # self.writer.add_image('Real and pred Mask', masks, iteration)