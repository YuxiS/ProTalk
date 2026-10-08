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
import torch.nn.functional as F
from tensorboardX import SummaryWriter
sys.path.append('..')
sys.path.append('.')
from hparams import create_hparams
from runtime_utils import str2bool
from vqvae.models.vqvae import VQVAE
from CoeffDataset import AudioDataset, collate_fn, collate_vq
from vqvae.datasets.DataPreFetecher import DataPrefetcher
from torch.utils.data import DataLoader
from utils import scale_function, unscale_function
import time
import wandb
import pdb
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
os.environ["CUDA_LAUNCH_BLOCKING"]='1'
warnings.filterwarnings('ignore')


# torch.autograd.detect_anomaly(True)
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)
# np.random.seed(42)

def load_datasets(hparams):
    trainset = AudioDataset(hparams.training_files, hparams, mode='train', dataset='Wild')
    valset = AudioDataset(hparams.validation_files, hparams, mode='val', dataset='Wild')

    if hparams.distributed_run:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    else:
        train_sampler = None
        val_sampler=None

    train_loader = DataLoader(trainset,  num_workers=8,
                              sampler=train_sampler,
                              batch_size=1024, pin_memory=True,
                              drop_last=True, collate_fn=collate_vq)
    val_loader = DataLoader(
        valset,  sampler=None, batch_size=64, pin_memory=True, num_workers=4,
        drop_last=True, collate_fn=collate_vq)
    return train_loader, val_loader

def test(model, test_loader, log_writer, epoch):
    with torch.no_grad():
        results = {
                'n_updates': 0,
                'recon_errors': [],
                'loss': [],
                'embedding_loss':[],
                'perplexities': [],
            }
        for batch in test_loader:
            mel, f0, energy, coeff_static, coeff_dynamic, coeff_crop = batch
            f0 =  f0.cuda()
            energy = energy.cuda()
            coeff_dynamic = coeff_dynamic.cuda()
            coeff_static = coeff_static.cuda()
            coeff_crop = coeff_crop.cuda()
            
            pose_coeff = torch.cat((coeff_dynamic[:, :, 64:], coeff_crop), dim=2)
            # pose_coeff = coeff_dynamic[:, :, :64] # for exp
            embedding_loss, pred_coeff, perplexity = model(pose_coeff)
            recon_loss = torch.mean((pred_coeff-pose_coeff)**2)

            loss = recon_loss + 5*embedding_loss
            results["recon_errors"].append(recon_loss.item())
            results["perplexities"].append(perplexity.item())
            results['embedding_loss'].append(embedding_loss.item())
            results["loss"].append(loss.item())
            results["n_updates"] += 1
        
        log_writer.add_scalar('Test recon_errors', np.mean(results['recon_errors']), epoch)
        log_writer.add_scalar('Test perplexities', np.mean(results['perplexities']), epoch)

        log_writer.add_scalar('Test loss', np.mean(results['loss']), epoch)
        print("Test loss:{:6f} perplexities:{:6f}  Test recon_loss:{:6f} Test embedding_loss:{:6f}".format(
            np.mean(results['loss']), 
            np.mean(results['perplexities']), 
            np.mean(results['recon_errors']),
            np.mean(results['embedding_loss'])))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', type=str, default='hparams.yaml')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default= 1000)
    parser.add_argument("--pose_dim", type=int, default=9)
    parser.add_argument("--n_hiddens", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--n_embeddings", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=.25)
    parser.add_argument('--local_rank', '--local-rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default='weights/vqvae')
    parser.add_argument('--distributed_run', type=str2bool, default=False)
    parser.add_argument('--log_dir', type=str, default='runs/train_vqvae')
    parser.add_argument('--debug', type=str2bool, default=False)

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

    model = VQVAE(args.pose_dim, args.n_embeddings, args.embedding_dim, beta=args.beta).cuda()
    if args.distributed_run:
        model = DDP(model, device_ids=[args.local_rank],find_unused_parameters=True)
    
    train_loader, val_loader = load_datasets(hparams)
    # train_prefetcher = DataPrefetcher(train_loader)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    step_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    if args.local_rank == 0 : #and not args.debug:
        wandb.init(
            # set the wandb project where this run will be logged
            project="ProTalk",
            # track hyperparameters and run metadata
            config=vars(args)
        )
        time_flag = time.strftime('%Y-%m-%d-%H_%M')
        os.makedirs(osp.join(args.log_dir, 'VQVAE-window-' + time_flag), exist_ok=True)
        os.makedirs(osp.join(args.save_dir, 'VQVAE-window-' + time_flag), exist_ok=True)
        save_dir = osp.join(args.save_dir, 'VQVAE-window-' + time_flag)
    # coeff_mean = torch.from_numpy(np.load(os.path.join(hparams.mean_std_root,  'mean.npy'))).view(1, 1, -1).float().cuda()
    # coeff_std = torch.from_numpy(np.load(os.path.join(hparams.mean_std_root, 'std.npy'))).view(1, 1, -1).float().cuda()
    results = {
            'n_updates': 0,
            'recon_errors': [],
            'loss': [],
            'perplexities': [],
            'embedding_loss':[]
        }

    for epoch in range(args.epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        if args.local_rank==0:
            print("Epoch:{}".format(epoch))
        time.sleep(0.003)
        for i, batch in enumerate(train_loader):
            
            # mel, f0, energy, coeff_dynamic, coeff_static, coeff_crop, frames, audio_length, frames_index= batch
            # pdb.set_trace()
            mel, f0, energy, coeff_static, coeff_dynamic, coeff_crop = batch
            f0 = f0.cuda()
            energy = energy.cuda()
            coeff_dynamic = coeff_dynamic.cuda()
            coeff_static = coeff_static.cuda()
            coeff_crop = coeff_crop.cuda()
            pose_coeff = torch.cat((coeff_dynamic[:, :, 64:], coeff_crop), dim=2)
            # pose_coeff = scale_function(pose_coeff)
            # pose_coeff = coeff_dynamic[:, :, :64] # for exp sample
            embedding_loss, pred_coeff, perplexity = model(pose_coeff, istrain=True)
            # embedding_loss, pred_coeff, ind = model(pose_coeff)
            recon_loss = F.mse_loss(pred_coeff, pose_coeff)
            optimizer.zero_grad()
            loss = recon_loss + 5*embedding_loss
            # pdb.set_trace()
            loss.backward()
            optimizer.step()
            if args.local_rank==0:
                results["recon_errors"].append(recon_loss.item())
                results["perplexities"].append(perplexity.item())
                results["loss"].append(loss.item())
                results['embedding_loss'].append(embedding_loss.item())
                results["n_updates"] += 1

            if args.local_rank==0 and (results["n_updates"]+1) % args.log_interval == 0:

                print("loss:{:6f} perplexities:{:6f} recon_loss:{:6f} embedding_loss:{:6f}".format(
                    np.mean(results['loss'][-args.log_interval:]), 
                    np.mean(results['perplexities'][-args.log_interval:]), 
                    np.mean(results['recon_errors'][-args.log_interval:]),
                    np.mean(results['embedding_loss'][-args.log_interval:])))
                wandb.log({'recon_errors':np.mean(results['loss'][-args.log_interval:])}, step=results['n_updates'])
                wandb.log({'perplexities':np.mean(results['perplexities'][-args.log_interval:])}, step=results['n_updates'])
                wandb.log({'embedding_loss':np.mean(results['embedding_loss'][-args.log_interval:])}, step=results['n_updates'])
                wandb.log({'loss':np.mean(results['loss'][-args.log_interval:])}, step=results['n_updates'])
        step_scheduler.step()
        if args.local_rank==0 and (epoch+1) % args.save_interval==0: 
            checkpoints = os.path.join(save_dir, 'vqvae_epoch_{}.pth'.format(epoch))
            if args.distributed_run:
                torch.save({
                    'Encoder':model.module.encoder.state_dict(),
                    'Decoder':model.module.decoder.state_dict(),
                    'CodeBook': model.module.vector_quantization.state_dict()}, checkpoints)
            else:
                torch.save({
                    'Encoder':model.encoder.state_dict(),
                    'Decoder':model.decoder.state_dict(),
                    'CodeBook': model.vector_quantization.state_dict()}, checkpoints)
