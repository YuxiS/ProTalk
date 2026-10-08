import argparse
from collections import OrderedDict
from  face_utils.renders import PIRenderFaceGenerator, PIRenderPreprocessor, PIRenderPostProcessor
# from unet.unet_model import UNet
from deep3d.models.face_render import Face_render as DeepFace3D
# from deep3d.options.face_render_options import Face_Render_Options
from deep3d.models.bfm import ParametricFaceModel 
import torch
from torch.utils.data import DataLoader
# from model import Tacotron2
# from model_vae import ProsoNet
# from models_easy import AudioEncoder, Aduio_Encoder_with_Attention
# from model import Audio_Exp_Model
from stylemodel import ProsoResNet, ResNet, ProsoLinear
from CoeffDataset import AudioDataset, collate_fn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn as nn
from torch.optim import lr_scheduler
from loss_function import CoffLoss, ExpLoss, ImgLoss
from hparams import create_hparams
from runtime_utils import str2bool
import os
import numpy as np
from utils import draw_keypoints
import time
import pdb
import warnings
import torch.nn.functional as F
import torchvision.utils as tvu
import face_alignment
from vqvae.generate import VAE as PoseGEN
import wandb
 
warnings.filterwarnings('ignore')
os.environ["CUDA_LAUNCH_BLOCKING"]='1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

# torch.autograd.set_detect_anomaly(True)

def model_to_gpu(model, opt):
    if opt.distributed_run:
        model = model.cuda()
        model = DDP(model, device_ids=[opt.local_rank], find_unused_parameters=True)
    else:
        model = model.cuda()

    return model


def initModel(hparams, opt):
    # audio_model = Audio_Exp_Model(hparams)
    # audio_model = ProsoLinear(hparams, 244, pro_channels=2, out_channels=64)
    # audio_model = ProsoResNet(hparams, 244, pro_channels=2, out_channels=64)

    audio_model = (ProsoResNet(hparams, 244, 2, 64) if opt.prosody
                   else ResNet(244, 64))
    # audio_model.load_state_dict(torch.load(hparams.audio_weight)['audio_model_with_exp'])

    pose_encooder = PoseGEN(hparams.PoseModel.in_dim, hparams.PoseModel.n_embeddings, 
        hparams.PoseModel.embedding_dim, hparams.PoseModel.n_hiddens, hparams.PoseModel.pose_dim,
        hparams.PoseModel.beta, None, None)
    face_render = PIRenderFaceGenerator()
    return audio_model, pose_encooder, face_render

def save_checkpoints(audio_model, epoch, save_dir):
    print("Saving model and optimizer state at iteration {} to {}".format(
        epoch, save_dir))
    torch.save({'iteration': epoch,
                'audio_model': (audio_model.module if hasattr(audio_model, 'module') else audio_model).state_dict()}, save_dir)
def draw_batch_KeyPoints(imgs,  keypoints):
    """
    imgs :[B, T, 3, w, h]
    keypoints:[B, T, 68, 3]
    """
    imgs_with_kp = []
    for b in range(imgs.shape[0]):
        for t in range(imgs.shape[1]):
            img = imgs[b, t, ...]
            img = ((img*0.5+0.5)*255).to(torch.uint8)
            ld = keypoints[b, t, ...]
            ld = ld.unsqueeze(0)
            img = draw_keypoints(img, ld, colors='red')
            imgs_with_kp.append(img/255)
    imgs_with_kp = torch.stack(imgs_with_kp).contiguous().view(imgs.shape[0], imgs.shape[1], imgs.shape[2],imgs.shape[3],imgs.shape[4])
    return imgs_with_kp.cuda()


def load_datasets(hparams):
    trainset = AudioDataset(hparams.training_files, hparams, mode='train', dataset='Wild')
    valset = AudioDataset(hparams.validation_files, hparams, mode='train', dataset='Wild')

    if hparams.distributed_run:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    else:
        train_sampler = None
        val_sampler=None

    train_loader = DataLoader(trainset, num_workers=4, 
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        valset, num_workers=4, sampler=val_sampler, batch_size=hparams.val_batch_size,
        drop_last=True, collate_fn=collate_fn)
    return train_loader, val_loader


def StructureCoeff(coeff_static, coeff_dynamic):
    _id, _tex, _gamma = coeff_static[:, :, :80], coeff_static[:, :, 80:160], coeff_static[:, :, 160:]
    _exp, _angle, _trans = coeff_dynamic[:, :, :64], coeff_dynamic[:, :, 64:67],coeff_dynamic[:, :, 67:] 
    coeff = torch.cat([_id, _exp, _tex,  _angle, _gamma, _trans], dim=2)
    return coeff

def get_coeff_dict(coeff_vect, coeff_crop):
    # Pirender只需要预测的三个参数即可
    coeff ={}
    coeff['exp'] = coeff_vect[:, :, 80:144] 
    coeff['angle'] = coeff_vect[:, :, 224:227]
    coeff['trans'] = coeff_vect[:, :, 254:257]
    coeff_crop[:, :, 1:] = coeff_crop[:, :, 1:]*256
    coeff['crop'] = coeff_crop
    # pdb.set_trace()
    return coeff

def compute_ld(detector, coeffs):

    B, T, C = coeffs.shape
    coeffs = coeffs.contiguous().view(-1, coeffs.shape[-1])
    coef_dict = detector.split_coeff(coeffs)
    face_shape = detector.compute_shape(coef_dict['id'], coef_dict['exp'])
    rotation = detector.compute_rotation(coef_dict['angle'])

    face_shape_transformed = detector.transform(face_shape, rotation, coef_dict['trans'])
    face_vertex = detector.to_camera(face_shape_transformed)
    
    face_proj = detector.to_image(face_vertex)
    landmark = detector.get_landmarks(face_proj)
    landmark = 224 - landmark[:, :, 1]
    landmark = landmark[:, :, 1:]*(224/256)
    landmark = landmark.contiguous().view(B, T, landmark.shape[-2], landmark.shape[-1])
    return landmark



def training_loop(opt):
    hparams = create_hparams(yaml_file=opt.hparams)
    hparams.local_rank = opt.local_rank
    hparams.distributed_run = opt.distributed_run
    if opt.multi_node:
        init_method = 'tcp://'+opt.master_addr+':'+str(opt.master_port)
    else:
        init_method = 'env://'

    print(init_method)
    if hparams.distributed_run:
        print("Distributed RUN!!!")
        dist.init_process_group(
            backend = hparams.dist_backend,
            init_method = init_method
        )
        torch.cuda.set_device(opt.local_rank)
    train_loader, val_loader = load_datasets(hparams)

    #################Init Model##################
    audio_model, _, face_render= initModel(hparams, opt)
    #################################################################################
    # audio_dict = torch.load('/home/songyifei9/code/prosody/StyleProsody/mellotron/checkpoints_Last/All/ALL_MEAD_NO_STYLE2023-04-06-11_14/checkpoint_epoch_15.pth', map_location='cpu')
    # audio_model.load_state_dict(audio_dict)
    # hparams.epoch_offset = 16

    #################################################################################
    face_render.load_state_dict(torch.load(hparams.pirender_weight)['net_G_ema'])
    preprocessor = PIRenderPreprocessor(semantic_radius=13)
    audio_model = model_to_gpu(audio_model, opt)
    # pose_model = model_to_gpu(pose_model, opt)
    face_render = face_render.cuda()
    mesh_render = DeepFace3D(opt)
    face_render.eval()
    #############################################
    
    ##################Loss######################
    hparams.lambda_coff_exp = 0.5
    hparams.lambda_coff_exp_delta = 1.
    hparams.lambda_coff_pose = 0
    hparams.lambda_coff_pose_delta = 0
    ################################
    loss_fun_coff = CoffLoss(w_exp=hparams.lambda_coff_exp, w_delta_exp=hparams.lambda_coff_exp_delta,
                w_pose=hparams.lambda_coff_pose, w_pose_delta=hparams.lambda_coff_pose_delta)
    loss_fun_exp = ExpLoss().cuda()
    loss_fun_img = ImgLoss(w_l1=hparams.lambda_img_l1)
    #############################################
    
    ################Train Params#################
    optimizer = torch.optim.Adam([
        {'params':audio_model.parameters(),'lr': hparams.learning_rate, 'weight_decay':hparams.weight_decay},
        # {'params':pose_model.parameters(), 'lr': hparams.finetune_learning_rate, 'betas':(hparams.PoseModel.beta1, 0.999)}
    ])
    step_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
   ####################################################################
    if opt.local_rank==0:
        log_dir = os.path.join(hparams.log_last_dir, 'All', 'ALL_MEAD_WILD' + time.strftime('%Y-%m-%d-%H_%M'))
        # logger = SummaryWriter(log_dir)
    output_directory = os.path.join(hparams.checkpoints_last_dir, 'ablia/exp_no_prosody', 'MEAD_WILD_NO_PRO'+ time.strftime('%Y-%m-%d-%H_%M'))
    os.makedirs(output_directory, exist_ok=True)

    coeff_mean = {}
    coeff_std = {}
    
    coeff_mean = torch.from_numpy(np.load(os.path.join(hparams.mean_std_root,  'mean_wild.npy'))).view(1, 1, -1).float().cuda()
    coeff_std = torch.from_numpy(np.load(os.path.join(hparams.mean_std_root, 'std_wild.npy'))).view(1, 1, -1).float().cuda()
        
    iteration = 0
    loss_coeff_exp = []
    loss_coeff_exp_delta = []
    loss_coeff_pose = []
    loss_coeff_pose_delta = []
    loss_coeff_crop = []
    loss_img_l1 = []

    loss_ld_mean = {
        'landmark': [],
        'relative_landmark': [],
        'lip_landmarks': [],
        'expression': [],
        # 'lipread': [],
        'all_loss': [],
    }
    ##############################
    loss_fun_exp.cfg={
        'landmark':0.0001,
        'relative_landmark': 0,
        'lip_landmarks': 0.5, 
        'expression':100,
        # 'lipread':50,
    }
    ##############################
    i = 0
    for epoch in range(hparams.epoch_offset, hparams.epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        if opt.local_rank==0:
            print("Epoch:{}".format(epoch))
        for batch in train_loader:
            optimizer.zero_grad()
            mel, f0, energy, coff_dynamic, coff_static, coff_crop, frames, audio_length, frames_index = batch
            mel = mel.cuda()
            f0 = f0.cuda()
            energy = energy.cuda()
            coff_dynamic = coff_dynamic.cuda()
            coff_static = coff_static.cuda()
            coff_crop = coff_crop.cuda()
            frames = frames.cuda()
            ####################################################
            # base_exp = coff_dynamic[:, 0, :64].unsqueeze(1)
            local_prosody = torch.cat((f0, energy), dim=2)
            # mel = mel[:, 1:, :]
            # local_prosody = local_prosody[:, 3:, :]
            # for l in audio_length:
            #     l = l - 1
            # 参数预测
            #############################
            # coff_exp_pred = audio_model(mel, local_prosody, audio_length) # for prosody
            coff_exp_pred = (audio_model(mel, local_prosody, audio_length) if opt.prosody
                             else audio_model(mel, audio_length))
            # coff_pose_pred = pose_model(local_prosody, audio_length)
            coff_pose = coff_dynamic[:, :, -6:]
            # coff_crop_pred = coff_pose_pred[:, :, -3:]
            coff_vect_pred = torch.cat([coff_exp_pred, coff_pose], dim=2)
            loss_coff, loss_exp, loss_delta_exp, loss_pose, loss_pose_delta = loss_fun_coff(
                coff_exp_pred, coff_dynamic[:, :, :64],
                torch.cat([coff_dynamic[:, :, 64:], coff_crop], dim=2),
                torch.cat([coff_dynamic[:, :, 64:], coff_crop], dim=2))
            # loss_crop = F.l1_loss(coff_pose_pred[:, :, -3:], coff_crop)
            
            #############################
            loss_coeff_exp.append(loss_exp.item())
            loss_coeff_exp_delta.append(loss_delta_exp.item())
            loss_coeff_pose.append(loss_pose.item())
            loss_coeff_pose_delta.append(loss_pose_delta.item())
            # loss_coeff_crop.append(loss_crop.item())
            #############################

            # Render
            ##############################################################################
            coeff_3dmm_pred = StructureCoeff(coff_static, coff_vect_pred)*coeff_std+coeff_mean
            coeff_3dmm_real = StructureCoeff(coff_static, coff_dynamic)*coeff_std+coeff_mean 
            pred_frames = []
            coeff_3dmm_dict = get_coeff_dict(coeff_3dmm_pred, coff_crop) 
            semantics = preprocessor.prepare_coeffs_batch(coeff_3dmm_dict, frames_index, audio_length)
            for i in range(semantics.shape[1]):
                img = face_render(frames[:, 0, ...], semantics[:, i, ...])
                pred_frames.append(img)
            pred_frames = torch.stack(pred_frames, dim=1)
            B, T, C, W, H = frames[:, 1:, ...].shape
           ###############################################################################
            coeffs_render_pred = []
            coeffs_render_real = []
            for i, index in enumerate(frames_index):
                coeffs_render_pred.append(coeff_3dmm_pred[i, index, :])
                coeffs_render_real.append(coeff_3dmm_real[i, index, :])
            coeffs_render_pred = torch.stack(coeffs_render_pred, dim=0)
            coeffs_render_real = torch.stack(coeffs_render_real, dim=0)
            landmarks_pred, _, _, pred_mesh = mesh_render.forward(coeffs_render_pred.contiguous().view(-1, coeffs_render_pred.shape[-1]))
            landmarks_real, _, _, real_mesh = mesh_render.forward(coeffs_render_real.contiguous().view(-1, coeffs_render_real.shape[-1]))
            landmarks_pred = landmarks_pred.contiguous().view(B, T, landmarks_pred.shape[-2], landmarks_pred.shape[-1])
            landmarks_real = landmarks_real.contiguous().view(B, T, landmarks_real.shape[-2], landmarks_real.shape[-1])
            pred_mesh = pred_mesh.contiguous().view(B, T, pred_mesh.shape[-3], pred_mesh.shape[-2], pred_mesh.shape[-1])
            real_mesh = real_mesh.contiguous().view(B, T, real_mesh.shape[-3], real_mesh.shape[-2], real_mesh.shape[-1])

            loss_ld = loss_fun_exp(landmarks_real+16, landmarks_pred+16, 
                                    pred_frames, frames[:, 1:, ...], landmarks_real.clone()+16, landmarks_pred.clone()+16)
            
            for key, values in loss_ld.items():
                loss_ld_mean[key].append(values.item())
            loss_img = loss_fun_img(pred_frames, frames[:, 1:, ...])
            loss_img_l1.append(loss_img.item())
            ##############################################################################
            loss = loss_coff + loss_ld['all_loss'] 
            loss.backward()
            optimizer.step()
            # pred_mesh = draw_batch_KeyPoints(pred_mesh, landmarks_pred)
            # real_mesh = draw_batch_KeyPoints(real_mesh, landmarks_real)
            # frames = draw_batch_KeyPoints(frames[:, 1:, ...], landmarks_real+16)
            # pred_frames = draw_batch_KeyPoints(pred_frames, landmarks_pred+16)
            # Log
            ##############################################################################
            iteration += 1
            i += 1
            # print(coff_frame_mean, coff_delta_mean, loss_lm_mean, loss_l1_mean, loss_lpips_mean)
            if opt.local_rank == 0 and iteration % hparams.iters_per_checkpoint == 0:
                
                print("Interation{} Training loss: coeff_exp:{:6f} coeff_delta_exp:{:6f} coeff_pose:{:6f} coeff_pose_delta:{:6f} img_l1:{:6f}".format(
                    iteration, np.mean(loss_coeff_exp[-hparams.iters_per_checkpoint:]),
                    np.mean(loss_coeff_exp_delta[-hparams.iters_per_checkpoint:]),
                    np.mean(loss_coeff_pose[-hparams.iters_per_checkpoint:]),
                    np.mean(loss_coeff_pose_delta[-hparams.iters_per_checkpoint:]),
                    np.mean(loss_img_l1[-hparams.iters_per_checkpoint:])
                    ))
                if not opt.debug:
                    wandb.log({'Coeff_exp':np.mean(loss_coeff_exp[-hparams.iters_per_checkpoint:])}, step=iteration)
                    wandb.log({'Coeff_exp_delta':np.mean(loss_coeff_exp_delta[-hparams.iters_per_checkpoint:])}, step=iteration)
                    wandb.log({'Exp Loss':np.mean(loss_ld_mean['expression'][-hparams.iters_per_checkpoint:])}, step=iteration)
                    wandb.log({'Landmarks Loss':np.mean(loss_ld_mean['landmark'][-hparams.iters_per_checkpoint:])}, step=iteration)
                    wandb.log({'Epoch':epoch})
                for key, values in loss_ld_mean.items():
                    print('{}:{}'.format(key, np.mean(values[-hparams.iters_per_checkpoint:])), end=' ')
                print()
                # for key, value in loss_GAN_mean.items():
                #     print('{}:{}'.format(key, value), end=' ')
                # print()
                # pdb.set_trace()
                # mesh_show = torch.cat([pred_mesh, real_mesh], dim=1)
                face_show = torch.cat([pred_frames, frames[:, 1:, ...]], dim=1)
                # mesh_show = mesh_show.contiguous().view(-1, mesh_show.shape[-3],mesh_show.shape[-2],mesh_show.shape[-1])[:80, ...]
                face_show = face_show.contiguous().view(-1, face_show.shape[-3], face_show.shape[-2], face_show.shape[-1])[:80, ...]
                # real_face_show = real_.contiguous().view(-1, real_face.shape[-3], real_face.shape[-2], real_face.shape[-1])
                # pred_face_show = pred_face.contiguous().view(-1, pred_face.shape[-3], pred_face.shape[-2], pred_face.shape[-1])
                # pred_frames_show = pred_frames.contiguous().view(-1, pred_frames.shape[-3], pred_frames.shape[-2], pred_frames.shape[-1])
                # real_frames_show = frames.contiguous().view(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
                os.makedirs(os.path.join(log_dir, 'train'), exist_ok=True)

                #################################
                
                #################################
                # tvu.save_image(mesh_show, os.path.join(log_dir, 'train' ,'mesh_{}.jpg'.format(iteration)))
                tvu.save_image((face_show*0.5+0.5) , os.path.join(log_dir, 'train' ,'{}.jpg'.format(iteration)))
                # logger.add_scalar('Train Coeff Exp', np.mean(loss_coeff_exp[-hparams.iters_per_checkpoint:]), iteration)
                # logger.add_scalar('Train Coeff Exp Delta', np.mean(loss_coeff_exp_delta[-hparams.iters_per_checkpoint:]), iteration)
                # logger.add_scalar('Train Coeff Pose', np.mean(loss_coeff_pose[-hparams.iters_per_checkpoint:]), iteration)
                # logger.add_scalar('Train Coeff Pose Delta', np.mean(loss_coeff_pose_delta[-hparams.iters_per_checkpoint:]), iteration)
                # # logger.add_scalar('Train Coeff Crop', np.mean(loss_coeff_crop[-hparams.iters_per_checkpoint:]), iteration)
                # for key, values in loss_ld_mean.items():
                #     logger.add_scalar("Train "+key, np.mean(values[-hparams.iters_per_checkpoint:]), iteration)

                    
                    
                    
                # tvu.save_image(real_face_show*0.5+0.5, os.path.join(log_dir, 'train' ,'{}_mask_real.jpg'.format(iteration)))
                # tvu.save_image(pred_face_show*0.5+0.5, os.path.join(log_dir, 'train' ,'{}_mask_fake.jpg'.format(iteration)))
                # tvu.save_image(real_frames_show*0.5+0.5, os.path.join(log_dir, 'train' ,'{}_face_real.jpg'.format(iteration)))
                # tvu.save_image(pred_frames_show*0.5+0.5, os.path.join(log_dir, 'train' ,'{}_face_fake.jpg'.format(iteration)))
                # img_show = torch.cat((real_face_show, pred_face_show, pred_frames_show, real_frames_show), dim=0)
                # img_save = img_show*0.5 +0.5
                
                # tvu.save_image(img_save, os.path.join(log_dir, 'train' ,'{}_.jpg'.format(iteration)))

                # logger.log_training(coff_exp.item(), coff_delta_exp.item(),
                #                     coff_pose.item(), coff_trans.item(), pred_face_show, real_face_show, iteration)
                # logger.log_training(coff_exp.item(), coff_delta_exp.item(),
                #                     coff_pose.item(), coff_trans.item(),
                #                     loss_GAN_mean, loss_exp_mean, img_show, iteration)
        step_scheduler.step()
            # step_D_scheduler.step()
        if opt.local_rank == 0 and (epoch+1) % hparams.iters_save_weight == 0: 
            checkpoints = os.path.join(output_directory, "checkpoint_epoch_{}.pth".format(epoch))
            save_checkpoints(audio_model, iteration, checkpoints)
     



if __name__ == '__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument('--prosody', action='store_true', help='Use ProsoResNet; default is the historical no-prosody ablation')
    parser.add_argument('--hparams', type=str, default='hparams.yaml')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    parser.add_argument('--distributed_run', type=str2bool, default=False)
    parser.add_argument('--multi_node', type=str2bool, default=False)
    parser.add_argument('--master_addr', type=str)
    parser.add_argument('--master_port',  type=int)
    parser.add_argument('--name', type=str, default='face_recon', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    parser.add_argument('--dataset_mode', type=str, default=None,
                        help='chooses how datasets are loaded. [None | flist]')
    parser.add_argument('--bfm_folder', type=str, default='deep3d/BFM')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')


    # renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)
    parser.add_argument('--use_opengl', type=str2bool, nargs='?', const=True, default=False, help='use opengl context or not')


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
    parser.add_argument('--isTrain', type=str2bool, default=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.set_defaults(
        focal=1015., center=112., camera_d=10., use_last_fc=False, z_near=5., z_far=15.
    )
    opt=parser.parse_args()
    if opt.local_rank==0 and not opt.debug:
        wandb.init(
            project='ProTalk',
            config=vars(opt))
    training_loop(opt)
