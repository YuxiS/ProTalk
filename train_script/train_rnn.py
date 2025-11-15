import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import torch.distributed as dist
import os 
import os.path as osp
import warnings
from torch.optim import lr_scheduler
import numpy as np
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
sys.path.append('..')
sys.path.append('.')
from hparams import create_hparams
# from models_easy import AudioEncoder
import wandb
from vqvae.models.vqvae import VQVAE
from CoeffDataset import AudioDataset, collate_fn, collate_fn_test
from torch.utils.data import DataLoader
from vqvae.pose_sampler.posesample import PoseSampler
from vqvae.datasets.DataPreFetecher import DataPrefetcher
import pdb

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
os.environ["CUDA_LAUNCH_BLOCKING"]='1'
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True




def load_datasets(hparams):
    trainset = AudioDataset(hparams.training_files, hparams, mode='train', dataset='Wild')
    valset = AudioDataset(hparams.validation_files, hparams, mode='val', dataset='Wild')

    if hparams.distributed_run:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    else:
        train_sampler = None
        val_sampler=None

    train_loader = DataLoader(trainset, num_workers=8, 
                              sampler=train_sampler,
                              batch_size=args.batch_size, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        valset, num_workers=8, sampler=None, batch_size=args.batch_size,
        drop_last=True, collate_fn=collate_fn)
    return train_loader, val_loader

def test(model, dataloader, logger, epoch):
    # model.eval()
    loss_list = []
    loss_pose_list = []
    loss_pose_delta_list = []
    with torch.no_grad():
        for batch in dataloader:
            mel, f0, energy, coeff_dynamic, coeff_static, coeff_crop, _, audio_length, _= batch
            mel = mel.cuda()
            f0 =  f0.cuda()
            energy = energy.cuda()
            coeff_dynamic = coeff_dynamic.cuda()
            coeff_static = coeff_static.cuda()
            coeff_crop = coeff_crop.cuda()
            pose_coeff = torch.cat((coeff_dynamic[:, :, 64:], coeff_crop), dim=2)
            target_index = []
            for t in range(0, pose_coeff.shape[1], 4):
                coeff_window = pose_coeff[:, t:t+8, :]
                if coeff_window.shape[1]<8:
                    paddings = torch.zeros(coeff_window.shape[0], 8-coeff_window.shape[1], coeff_window.shape[2]).cuda()
                    coeff_window = torch.cat([coeff_window, paddings], dim=1)
                with torch.no_grad():
                    z_e  = vae_model.encoder(coeff_window)
                    _, _, (_, min_encodings, min_encoding_indices) = vae_model.vector_quantization(z_e)
                    min_encoding_indices = min_encoding_indices.long()
                    target_index.append(min_encoding_indices)
            target_index = torch.cat(target_index, dim=1)
            prosody_data = torch.cat([f0, energy], dim=2) #[B, T, C]
            proso_index = pose_model(prosody_data, audio_length)
            pred_index = proso_index[:, ::8, :]
            # sampled_index = pose_model(mel[:, :, :80], audio_length)
            #################################################
            # pdb.set_trace()
            sampled_index = proso_index[:, ::4, :]
            res = []
            # pred_index = []
            B = prosody_data.shape[0]
            # pdb.set_trace()
            for t in range(sampled_index.shape[1]):
                sampled_index_temp = sampled_index[:, t, :]
                sampled_index_temp = sampled_index_temp.contiguous().view(-1, sampled_index_temp.shape[-1])
                min_index = torch.argmax(sampled_index_temp, dim=1)
                min_encodings = torch.zeros((min_index.shape[0], args.n_embeddings), dtype=torch.float).cuda()
                min_encodings.scatter_(1, min_index.unsqueeze(1), 1)
                e_weights = vae_model.vector_quantization.embedding.weight
                z_q = torch.matmul(min_encodings, e_weights)
                z_q = z_q.contiguous().view(B, -1, z_q.shape[-1])
                pose_seq_temp = vae_model.decoder(z_q)
                # res.append(pose_seq_temp)
                half = int(pose_seq_temp.shape[1]/2)
                if len(res)==0:
                    res.append(pose_seq_temp[:, :half, :])
                    res.append(pose_seq_temp[:, half:, :])
                else:
                    half = int(pose_seq_temp.shape[1]/2)
                    last = res.pop()
                    temp = (last + pose_seq_temp[:, :half, :])/2
                    res.append(temp)
                    res.append(pose_seq_temp[:, half:, :])
            # pdb.set_trace()
            pose_seq = torch.cat(res, dim=1)
            if pose_seq.shape[1]>pose_coeff.shape[1]:
                pose_seq = pose_seq[:, :pose_coeff.shape[1], :]
            elif pose_seq.shape[1]<pose_coeff.shape[1]:
                paddings = torch.zeros(pose_seq.shape[0], pose_coeff.shape[1]-pose_seq.shape[1], pose_seq.shape[2]).cuda()
                pose_coeff = torch.cat((pose_seq, paddings), dim=1)
            mask = torch.zeros_like(pose_seq).cuda()
            for i, l in enumerate(audio_length):
                mask[i, :l, :]=1.
            pose_seq = pose_seq * mask
            # with torch.no_grad():
            # z_e  = vae_model.encoder(pose_coeff)
            # _, _, _, min_encodings, min_encoding_indices = vae_model.vector_quantization(z_e)
            # min_encoding_indices = min_encoding_indices.view(-1).long()
            # prosody_data = torch.cat([f0, energy], dim=2) #[B, T, C]
            # sampled_index = pose_model(prosody_data, audio_length)
            # # sampled_index = pose_model(mel[:, :, :80], audio_length)
            # sampled_index = sampled_index.contiguous().view(-1, sampled_index.shape[-1])
            ###################################################
            # B = prosody_data.shape[0]
            # min_index = torch.argmax(sampled_index, dim=1)
            # min_encodings = torch.zeros((min_index.shape[0], args.n_embeddings), dtype=torch.float).cuda()
            # min_encodings.scatter_(1, min_index.unsqueeze(1), 1)
            # e_weights = vae_model.vector_quantization.embedding.weight
            # z_q = torch.matmul(min_encodings, e_weights)
            # z_q = z_q.contiguous().view(B, -1, z_q.shape[-1])
            # pose_seq = vae_model.decoder(z_q)
            pose_l1_loss = loss_l1(pose_seq, pose_coeff)
            pose_l1_delta_loss = loss_l1((pose_seq[:, 1:, :]-pose_seq[:, :-1, :]), (pose_coeff[:, 1:, :]-pose_coeff[:, :-1, :]))
            ###################################################
            crossentropy_loss = loss_fun(pred_index, target_index) 
            loss_pose_list.append(pose_l1_loss.item())
            loss_pose_delta_list.append(pose_l1_delta_loss.item())
            loss_list.append(crossentropy_loss.item())

    logger.add_scalar('Test cross entropy', np.mean(loss_list), epoch)
    logger.add_scalar('Test L1 Pose', np.mean(loss_pose_list), epoch)
    logger.add_scalar('Test L1 Pose Delta', np.mean(loss_pose_delta_list), epoch)
    print('Test Cross loss:{:6f} L1 Pose:{:6f} L1 pose delta: {:6f}'.format(np.mean(loss_list), np.mean(loss_pose_list), np.mean(loss_pose_delta_list)))
    # model.train()
    # dist.barrier()
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', type=str, default='/remote-home/yfsong/code/ProTalk/hparams.yaml')
    parser.add_argument("--batch_size", type=int, default= 512)
    parser.add_argument("--epochs", type=int, default= 500)
    parser.add_argument("--pose_dim", type=int, default=9)
    parser.add_argument("--n_hiddens", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--n_embeddings", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=.25)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default='/remote-home/yfsong/code/ProTalk/weights/headrnn')
    parser.add_argument('--distributed_run', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--vae_weight', type=str, default='/remote-home/yfsong/code/ProTalk/weights/vqvae/VQVAE-window-2024-01-09-06_30/vqvae_epoch_999.pth')

    args = parser.parse_args()
    hparams = create_hparams(yaml_file=args.hparams)
    hparams.local_rank = args.local_rank
    hparams.distributed_run = args.distributed_run

    if args.distributed_run:
        print("Distributed RUN!!!")
        dist.init_process_group(
            backend=hparams.dist_backend,
            init_method= 'env://'
        )
        torch.cuda.set_device(args.local_rank)

    vae_model = VQVAE(args.pose_dim, args.n_embeddings, args.embedding_dim, beta=args.beta).cuda()
    vae_model.encoder.load_state_dict(torch.load(args.vae_weight)['Encoder'])
    vae_model.decoder.load_state_dict(torch.load(args.vae_weight)['Decoder'])
    vae_model.vector_quantization.load_state_dict(torch.load(args.vae_weight)['CodeBook'])
    vae_model.eval()
    ########################################
    # pose_model = PoseSampler(2, args.n_embeddings, args.n_hiddens).cuda() # For prosody
    pose_model = PoseSampler(80, args.n_embeddings, args.n_hiddens).cuda() # For Mel
    # pose_model = AudioEncoder(hparams).cuda() #For exp 
    if args.distributed_run:
        pose_model = DDP(pose_model, device_ids=[args.local_rank])
    train_loader, val_loader = load_datasets(hparams)
    optimizer = optim.AdamW(pose_model.parameters(), lr=args.learning_rate, weight_decay=0.03, betas=(0.9, 0.999))
    step_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
    loss_fun = nn.CrossEntropyLoss(reduction='none')
    loss_l1 = nn.L1Loss()
    if args.local_rank == 0 and not args.debug:
        wandb.init(
            # set the wandb project where this run will be logged
            project="ProTalk",
            # track hyperparameters and run metadata
            name='PoseSampler-No_prosody',
            config=vars(args)
        )
        time_flag = time.strftime('%Y-%m-%d-%H_%M')
        # os.makedirs(osp.join(args.log_dir, 'PoseSampler' + time_flag), exist_ok=True)
        os.makedirs(osp.join(args.save_dir, 'PoseSampler' + time_flag), exist_ok=True)
        save_dir = osp.join(args.save_dir, 'PoseSampler' + time_flag)
    steps = 0
    loss_value = []
    loss_pose_value = []
    loss_pose_delta_value = []
    for epoch in range(args.epochs):
        if args.local_rank==0:
            print("Epoch:{}".format(epoch))
        prefetecher = DataPrefetcher(train_loader)
        batch = prefetecher.next()
        while batch is not None:
            mel, f0, energy, coeff_dynamic, coeff_static, coeff_crop, audio_length= batch 
        ########################################
            optimizer.zero_grad()
            pose_coeff = torch.cat((coeff_dynamic[:, :, 64:], coeff_crop), dim=2)
            # base_pose = pose_coeff[:, 0, :].unsqueeze(1)
            # pose_coeff = pose_coeff - base_pose
            # pose_coeff = coeff_dynamic[:, :, :64]
            #######################################
            target_index = []
            for t in range(0, pose_coeff.shape[1], 4):
                coeff_window = pose_coeff[:, t:t+8, :]
                if coeff_window.shape[1]<8:
                    paddings = torch.zeros(coeff_window.shape[0], 8-coeff_window.shape[1], coeff_window.shape[2]).cuda()
                    coeff_window = torch.cat([coeff_window, paddings], dim=1)
                with torch.no_grad():
                    z_e  = vae_model.encoder(coeff_window)
                    _, _, (_, min_encodings, min_encoding_indices) = vae_model.vector_quantization(z_e)
                    min_encoding_indices = min_encoding_indices.long()
                    target_index.append(min_encoding_indices)
            target_index = torch.cat(target_index, dim=1)
            ####################################
            ################################
            # prosody_data = torch.cat([f0, energy], dim=2) #[B, T, C] for prosody
            # proso_index = pose_model(prosody_data, audio_length)
            ####################################
            prosody_data = mel # for mel
            proso_index = pose_model(mel[:, :, :80], audio_length) # for mel
            ################################################
            # index_mask = torch.zeros(size=(proso_index.shape[0], proso_index.shape[1], 1),
            #                          device=proso_index.get_device())
            # for i, l in enumerate(audio_length):
            #     l = int(l//3)
            #     index_mask[i, :l, :]=1.
            # index_mask = index_mask[:, ::4, :].squeeze(2).flatten()
            ################################################
            sampled_index = proso_index[:, ::4, :]
            pred_index = proso_index[:, ::4, :]
            res = []
            # pred_index = []
            B = prosody_data.shape[0]
            for t in range(sampled_index.shape[1]):
                sampled_index_temp = sampled_index[:, t, :]
                sampled_index_temp = sampled_index_temp.contiguous().view(-1, sampled_index_temp.shape[-1])
                min_index = torch.argmax(sampled_index_temp, dim=1)
                min_encodings = torch.zeros((min_index.shape[0], args.n_embeddings), dtype=torch.float).cuda()
                min_encodings.scatter_(1, min_index.unsqueeze(1), 1)
                e_weights = vae_model.vector_quantization.embedding.weight
                z_q = torch.matmul(min_encodings, e_weights)
                z_q = z_q.contiguous().view(B, -1, z_q.shape[-1])
                pose_seq_temp = vae_model.decoder(z_q)
                # res.append(pose_seq_temp)
                half = int(pose_seq_temp.shape[1]/2)
                if len(res)==0:
                    res.append(pose_seq_temp[:, :half, :])
                    res.append(pose_seq_temp[:, half:, :])
                else:
                    last = res.pop()
                    temp = (last + pose_seq_temp[:, :half, :])/2.
                    res.append(temp)
                    res.append(pose_seq_temp[:, half:, :])
            pose_seq = torch.cat(res, dim=1)
            if pose_seq.shape[1]>pose_coeff.shape[1]: #如果预测的T>真实序列
                pose_seq = pose_seq[:, :pose_coeff.shape[1], :]
            elif pose_seq.shape[1]<pose_coeff.shape[1]:
                paddings = torch.zeros(pose_seq.shape[0], pose_coeff.shape[1]-pose_seq.shape[1], pose_seq.shape[2]).cuda()
                pose_coeff = torch.cat((pose_seq, paddings), dim=1)
            mask = torch.zeros_like(pose_seq).cuda()
            for i, l in enumerate(audio_length):
                mask[i, :l, :]=1.
            pose_seq = pose_seq * mask
            pose_coeff = pose_coeff * mask
            pose_l1_loss = loss_l1(pose_seq, pose_coeff)
            pose_l1_delta_loss = loss_l1((pose_seq[:, 1:, :]-pose_seq[:, :-1, :]), (pose_coeff[:, 1:, :]-pose_coeff[:, :-1, :]))
            #######################################################
            #################################################p#######
            pred_index = pred_index.contiguous().view(-1, pred_index.shape[-1])
            target_index = target_index.contiguous().view(-1)
            #########################################################
            # crossentropy_loss = loss_fun(sampled_index, min_encoding_indices)
            crossentropy_loss = loss_fun(pred_index, target_index)
            # crossentropy_loss[torch.isnan(crossentropy_loss)] = 0.
            loss = (crossentropy_loss).mean() + 5*pose_l1_loss+ pose_l1_delta_loss
            loss.backward()
            optimizer.step()
            batch = prefetecher.next()
            if not args.debug and args.local_rank == 0:
                wandb.log({'cross entropy': crossentropy_loss.mean().item()})
                wandb.log({'L1 pose': pose_l1_loss.item()})
                wandb.log({'L1 pose delta': pose_l1_delta_loss.item()})
                wandb.log({'Epoch': epoch})
            torch.cuda.empty_cache()
            if args.local_rank == 0 :
                print('Train loss: CrossEropy: {:6f} L1: {:6f}, L1_delta:{:6f}'.format(
                    crossentropy_loss.mean().item(), pose_l1_loss.item(), pose_l1_delta_loss.item()
                    )
                )
            
        if args.local_rank==0:
            step_scheduler.step()
        # if args.local_rank==0:
        #     test(pose_model, val_loader, log_writer, epoch)
        if args.local_rank==0 and (epoch+1) % args.save_interval==0: 
            checkpoints = os.path.join(save_dir, 'pose_sampler_epoch_{}.pth'.format(epoch))
            torch.save(pose_model.module.state_dict(), checkpoints)
        
