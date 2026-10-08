"""Train the existing expression/VQ-VAE/prosody-sampler networks."""
from protalk.paths import REPO_ROOT
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from protalk.config import create_hparams
from protalk.training.data import ClipDataset, PoseWindows, collate_clips, read_manifest
from protalk.training.objectives import expression_loss, pose_codes, sampler_loss
from protalk.training.checkpoints import atomic_save, capture_rng, restore_rng, inference_export


class EvaluationSampler(Sampler):
    def __init__(self, dataset, rank, world):
        self.indices = list(range(rank, len(dataset), world))
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def resolve_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    root = REPO_ROOT
    for key in ('hparams', 'train_manifest', 'val_manifest', 'statistics', 'output'):
        value = Path(config[key]).expanduser()
        config[key] = str(value if value.is_absolute() else root / value)
    for stage in ('expression', 'sampler'):
        for key in ('pirender_weight', 'bfm_folder', 'emotion_weight', 'vqvae_weight'):
            if key in config[stage]:
                value = Path(config[stage][key]).expanduser()
                config[stage][key] = str(value if value.is_absolute() else root / value)
    return config


def build_model(stage, hp, settings, resume=False):
    if stage == 'expression':
        from protalk.models.expression import ProsoResNet
        mode = settings['gst_init']
        if mode not in ('scratch', 'pretrained'):
            raise ValueError('gst_init must be scratch or pretrained')
        model = ProsoResNet(hp, 244, 2, 64, load_gst=mode == 'pretrained' and not resume,
                           train_gst=mode == 'scratch')
        if mode == 'pretrained':
            model.gst.requires_grad_(False)
            model.gst_transfrom.requires_grad_(False)
        return model
    if stage == 'vqvae':
        from protalk.models.pose.vqvae.vqvae import VQVAE
        model = VQVAE(hp.PoseModel.pose_dim, hp.PoseModel.n_embeddings,
                      hp.PoseModel.embedding_dim, hp.PoseModel.beta)
        model.vector_quantization.embedding.requires_grad_(False)
        model.vector_quantization.ema_w.requires_grad_(False)
        # Match the initial EMA numerator to a unit prior count per code.
        # Otherwise unused initial embeddings are divided by near-zero counts.
        model.vector_quantization.ema_cluster_size.fill_(1.)
        return model
    from protalk.models.pose.sampler.posesample import PoseSampler
    return PoseSampler(2, hp.PoseModel.n_embeddings, hp.PoseModel.n_hiddens)


def run(stage, config, device_name='cuda', resume=None, max_steps=None, initialize=None):
    output = Path(config['output']) / stage
    if not resume and output.exists() and any(output.glob('*.pth')):
        raise FileExistsError(f'Choose a new output directory or resume an existing run: {output}')
    world, rank = int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if device_name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; use --device cpu for a smoke test')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device(device_name)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    seed = config['seed'] + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    # Imports may set a historical CUDA seed; reseed after constructing the model below.
    hp = create_hparams(config['hparams'])
    root = REPO_ROOT
    if not Path(hp.gst_weight).is_absolute():
        hp.gst_weight = str(root / hp.gst_weight)
    settings = config[stage]
    if initialize and stage == 'vqvae':
        raise ValueError('--initialize supports expression/sampler exports; use --resume for VQ-VAE')
    if settings['batch_size'] < 2:
        raise ValueError('Use a per-rank batch size >= 2 for normalization layers')
    training_ids = {r['sample'] for r in read_manifest(config['train_manifest'])}
    validation_ids = {r['sample'] for r in read_manifest(config['val_manifest'])}
    if training_ids & validation_ids:
        raise ValueError('Training and validation samples overlap')
    train_clips = ClipDataset(config['train_manifest'], config['statistics'], config['max_frames'], True)
    val_clips = ClipDataset(config['val_manifest'], config['statistics'], config['max_frames'], False)
    train_data = PoseWindows(train_clips) if stage == 'vqvae' else train_clips
    val_data = PoseWindows(val_clips) if stage == 'vqvae' else val_clips
    train_sampler = DistributedSampler(train_data, world, rank, seed=config['seed']) if world > 1 else None
    collate = None if stage == 'vqvae' else collate_clips
    def loader(dataset, training=False):
        return DataLoader(dataset, batch_size=settings['batch_size'],
            sampler=train_sampler if training else EvaluationSampler(dataset, rank, world),
            shuffle=training and train_sampler is None, num_workers=config['workers'],
            pin_memory=device.type == 'cuda', drop_last=training,
            collate_fn=collate, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(config['seed']))
    train_loader, val_loader = loader(train_data, True), loader(val_data)
    if not len(train_loader):
        raise ValueError('Not enough training samples/windows for one full per-rank batch; reduce batch_size')
    model = build_model(stage, hp, settings, bool(resume or initialize)).to(device)
    visual = None
    if stage == 'expression' and settings['loss_profile'] == 'visual':
        from protalk.training.visual import VisualObjective
        visual = VisualObjective(settings, train_clips.stats, device)
    elif stage == 'expression' and settings['loss_profile'] != 'coefficients':
        raise ValueError('loss_profile must be coefficients or visual')
    vqvae = None
    if stage == 'sampler':
        vqvae = build_model('vqvae', hp, config['vqvae']).to(device)
        state = torch.load(settings['vqvae_weight'], map_location='cpu')
        for key, module in [('Encoder', vqvae.encoder), ('Decoder', vqvae.decoder), ('CodeBook', vqvae.vector_quantization)]:
            module.load_state_dict(state[key])
        vqvae.requires_grad_(False).eval()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    optimizer_type = torch.optim.Adam if stage == 'expression' else torch.optim.AdamW
    optimizer = optimizer_type([p for p in model.parameters() if p.requires_grad],
        lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, settings['scheduler_step'], settings['scheduler_gamma'])
    start_epoch, step, best = 0, 0, float('inf')
    architecture = {'pose_dim': hp.PoseModel.pose_dim, 'codes': hp.PoseModel.n_embeddings,
                    'embedding_dim': hp.PoseModel.embedding_dim, 'hidden': hp.PoseModel.n_hiddens}
    statistics_hash = hashlib.sha256(b''.join((Path(config['statistics']) / (name + '.npy')).read_bytes()
        for name in ('mean', 'std', 'mfcc_mean', 'mfcc_std'))).hexdigest()
    for dependency in ([state] if stage == 'sampler' else []):
        recorded = dependency.get('metadata', {}).get('statistics_hash')
        if recorded and recorded != statistics_hash:
            raise ValueError('VQ-VAE normalization statistics do not match the sampler dataset')
    if resume and initialize:
        raise ValueError('Use --resume or --initialize, not both')
    if resume:
        state = torch.load(resume, map_location='cpu')
        if state.get("partial_epoch"):
            raise ValueError("Partial smoke-test checkpoints cannot be resumed as completed epochs")
        if state['stage'] != stage or state['architecture'] != architecture or state['statistics_hash'] != statistics_hash:
            raise ValueError('Resume stage, architecture or training statistics do not match')
        old = state['config']
        for key in ('train_manifest', 'val_manifest', 'hparams', 'max_frames', 'seed', 'workers'):
            if old[key] != config[key]:
                raise ValueError(f'Resume configuration changed: {key}; use --initialize for a new run')
        old_settings = {k: v for k, v in old[stage].items() if k != 'epochs'}
        new_settings = {k: v for k, v in settings.items() if k != 'epochs'}
        if old_settings != new_settings:
            raise ValueError('Resume stage settings changed; use --initialize for a new run')
        if len(state['rng']) != world:
            raise ValueError('Resume requires the same world size for RNG restoration')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, step, best = state['epoch'] + 1, state['step'], state['best']
        restore_rng(state['rng'][rank])
    if initialize:
        state = torch.load(initialize, map_location='cpu')
        recorded = state.get('metadata', {}).get('statistics_hash')
        if recorded and recorded != statistics_hash:
            raise ValueError('Initialization checkpoint normalization statistics do not match')
        model.load_state_dict(state['audio_model'] if stage == 'expression' else state)
    if stage == 'vqvae':
        model.vector_quantization.sync_ema = world > 1
    wrapped = DistributedDataParallel(model, device_ids=[local_rank] if device.type == 'cuda' else None,
        broadcast_buffers=False, find_unused_parameters=stage == 'expression') if world > 1 else model
    output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (output / 'config.json').write_text(json.dumps(config, indent=2))
    def objective(batch, training):
        network = wrapped if training else model
        if stage == 'vqvae':
            target = batch.to(device)
            embedding_loss, predicted, perplexity = network(target, istrain=training)
            reconstruction = (predicted - target).square().mean()
            loss = reconstruction + settings['embedding_loss_weight'] * embedding_loss
            return loss, {'reconstruction': reconstruction, 'embedding': embedding_loss, 'perplexity': perplexity}, len(target)
        lengths = batch['lengths']
        mfcc, prosody, pose = [batch[k].to(device) for k in ('mfcc', 'prosody', 'pose')]
        if stage == 'sampler':
            target = pose_codes(vqvae, pose)
            loss, metrics = sampler_loss(network(prosody, lengths), target, lengths)
        else:
            prediction = network(mfcc, prosody, lengths)
            loss, metrics = expression_loss(prediction, batch['expression'].to(device), lengths,
                settings['coefficient_weight'], settings['delta_weight'])
            if visual:
                extra, extra_metrics = visual(prediction, batch)
                loss = loss + extra; metrics.update(extra_metrics)
        return loss, metrics, sum(lengths)
    for epoch in range(start_epoch, settings['epochs']):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        train_loader.generator.manual_seed(config['seed'] + epoch)
        model.train()
        if stage == 'expression' and settings['gst_init'] == 'pretrained':
            model.gst.eval()
        train_totals = {}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics, count = objective(batch, True)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_gradient'], error_if_nonfinite=True)
            optimizer.step(); step += 1
            train_totals = {'loss': float(loss.detach()), 'gradient_norm': float(norm),
                            **{k: float(v.detach()) for k, v in metrics.items()}}
            if rank == 0:
                print(json.dumps({'stage': stage, 'epoch': epoch, 'step': step, 'train': train_totals}), flush=True)
            if max_steps and step >= max_steps:
                break
        # Give all ranks the same normalization buffers before uneven validation batches.
        if world > 1:
            for buffer in model.buffers():
                dist.broadcast(buffer, 0)
        rng = capture_rng()
        random.seed(config["seed"] + 10000 + rank)
        np.random.seed(config["seed"] + 10000 + rank)
        torch.manual_seed(config["seed"] + 10000 + rank)
        model.eval()
        total = torch.zeros(2, dtype=torch.float64, device=device)
        with torch.no_grad():
            for batch in val_loader:
                loss, _, count = objective(batch, False)
                total += torch.tensor([float(loss) * count, count], device=device)
        if world > 1:
            dist.all_reduce(total)
        restore_rng(rng)
        validation = float(total[0] / total[1])
        if not np.isfinite(validation):
            raise FloatingPointError('Nonfinite validation loss')
        improved = validation < best
        best = min(best, validation)
        scheduler.step()
        rng_states = [None] * world
        if world > 1:
            dist.all_gather_object(rng_states, capture_rng())
        else:
            rng_states[0] = capture_rng()
        if rank == 0:
            metadata = {'stage': stage, 'epoch': epoch, 'pose_scale': 1., 'target': 'reference_displacement',
                        'architecture': architecture, 'statistics_hash': statistics_hash}
            state = {'stage': stage, 'epoch': epoch, 'step': step, 'best': best, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rng': rng_states,
                'config': config, 'architecture': architecture, 'statistics_hash': statistics_hash,
                'partial_epoch': bool(max_steps and step >= max_steps)}
            atomic_save(state, output / 'last-training.pth')
            exported = inference_export(stage, model, metadata)
            atomic_save(exported, output / 'last-inference.pth')
            if improved:
                atomic_save(state, output / 'best-training.pth')
                atomic_save(exported, output / 'best-inference.pth')
            entry = {'epoch': epoch, 'step': step, 'validation_loss': validation, 'best': best,
                     'learning_rate': optimizer.param_groups[0]['lr'], 'last_train_batch': train_totals,
                     'partial_epoch': bool(max_steps and step >= max_steps)}
            with (output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(entry) + '\n')
            print(json.dumps(entry), flush=True)
        if max_steps and step >= max_steps:
            break
    if world > 1:
        dist.destroy_process_group()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['expression', 'vqvae', 'sampler'])
    parser.add_argument('--config', default='configs/train.yaml')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--resume')
    parser.add_argument('--initialize', help='Warm-start an expression/sampler export with a fresh optimizer')
    parser.add_argument('--max-steps', type=int, help='Smoke test only: stop early and mark the partial epoch')
    args = parser.parse_args()
    run(args.stage, resolve_config(args.config), args.device, args.resume, args.max_steps, args.initialize)


if __name__ == '__main__':
    main()
