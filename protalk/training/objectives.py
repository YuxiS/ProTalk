import torch
import torch.nn.functional as F


def mask_for(lengths, frames, device):
    return torch.arange(frames, device=device)[None] < torch.as_tensor(lengths, device=device)[:, None]


def expression_loss(predicted, target, lengths, coefficient_weight=.5, delta_weight=1.):
    mask = mask_for(lengths, target.shape[1], target.device)
    distances = torch.linalg.vector_norm(predicted - target + 1e-6, dim=-1)
    coefficient = distances[mask].mean()
    delta_mask = mask[:, 1:]
    delta = torch.linalg.vector_norm((predicted[:, 1:] - predicted[:, :-1]) -
                                    (target[:, 1:] - target[:, :-1]) + 1e-6, dim=-1)[delta_mask].mean()
    return coefficient_weight * coefficient + delta_weight * delta, {'coefficient': coefficient, 'delta': delta}


@torch.no_grad()
def pose_codes(vqvae, pose):
    frames = pose.shape[1]
    windows = []
    for start in range(0, frames, 4):
        window = pose[:, start:start + 8]
        windows.append(F.pad(window, (0, 0, 0, 8 - window.shape[1])))
    windows = torch.stack(windows, 1)
    batch, steps = windows.shape[:2]
    encoded = vqvae.encoder(windows.flatten(0, 1))
    _, _, (_, _, indices) = vqvae.vector_quantization(encoded, istrain=False)
    if indices.numel() != batch * steps:
        raise ValueError('The pose codebook must encode one code per eight-frame window')
    return indices.reshape(batch, steps)


def sampler_loss(logits, targets, lengths):
    selected = logits[:, ::4]
    if selected.shape[:2] != targets.shape:
        raise ValueError('Sampler logits and target-code lengths differ')
    mask = mask_for(lengths, logits.shape[1], logits.device)[:, ::4]
    loss = F.cross_entropy(selected[mask], targets[mask])
    accuracy = (selected.argmax(-1)[mask] == targets[mask]).float().mean()
    return loss, {'cross_entropy': loss, 'code_accuracy': accuracy}
