"""Optional visual fine-tuning through a frozen PIRender and expression network."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def selected_frames(video, indices):
    import cv2
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f'Cannot read video: {video}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isclose(fps, 30., atol=.05):
        capture.release()
        raise ValueError(f'Expected aligned 30-fps video, got {fps}: {video}')
    wanted, found = set(indices), {}
    index = 0
    try:
        while index <= max(indices):
            ok, image = capture.read()
            if not ok:
                break
            if index in wanted:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if image.shape[:2] != (256, 256):
                    raise ValueError('Visual training requires aligned 256x256 videos')
                found[index] = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 127.5 - 1
            index += 1
    finally:
        capture.release()
    if any(i not in found for i in indices):
        raise ValueError(f'Video and coefficient lengths do not match: {video}')
    return torch.stack([found[i] for i in indices])


class VisualObjective:
    def __init__(self, settings, statistics, device):
        from protalk.third_party.face_utils.renders import PIRenderFaceGenerator
        from protalk.third_party.deep3d.models.bfm import ParametricFaceModel
        from protalk.third_party.emoca.expression_loss import ExpressionLossNet
        self.settings, self.device = settings, device
        for key in ('pirender_weight', 'emotion_weight'):
            if not Path(settings[key]).is_file():
                raise FileNotFoundError(f"Visual training requires {key}: {settings[key]}")
        self.renderer = PIRenderFaceGenerator().to(device)
        self.renderer.load_state_dict(torch.load(settings['pirender_weight'], map_location='cpu')['net_G_ema'])
        self.renderer.requires_grad_(False).eval()
        self.emotion = ExpressionLossNet(pretrained=False).to(device)
        weights = torch.load(settings['emotion_weight'], map_location='cpu')['state_dict']
        weights = dict(weights)
        if 'linear.weight' in weights:
            weights['linear.0.weight'] = weights.pop('linear.weight')
            weights['linear.0.bias'] = weights.pop('linear.bias')
        missing, _ = self.emotion.load_state_dict(weights, strict=False)
        if any(key.startswith('backbone.') for key in missing):
            raise ValueError('Expression-loss checkpoint is missing backbone weights')
        self.emotion.requires_grad_(False).eval()
        self.face = ParametricFaceModel(settings['bfm_folder'], is_train=False)
        self.face.to(device)
        self.std = torch.as_tensor(statistics['std'], device=device)

    def landmarks(self, coeff):
        split = self.face.split_coeff(coeff)
        shape = self.face.compute_shape(split['id'], split['exp'])
        rotation = self.face.compute_rotation(split['angle'])
        transformed = self.face.transform(shape, rotation, split['trans'])
        landmarks = self.face.get_landmarks(self.face.to_image(self.face.to_camera(transformed)))
        return torch.stack([landmarks[..., 0] + 16, 224 - landmarks[..., 1] + 16], -1)

    def __call__(self, prediction, batch):
        from protalk.third_party.emoca.lossfunc import weighted_landmark_loss
        components = []
        for b, sample in enumerate(batch['samples']):
            if not sample['video']:
                raise ValueError('Visual profile requires a video path in every prepared manifest record')
            length = batch['lengths'][b]
            raw = batch['coeff'][b, :length].to(self.device)
            base = batch['base_coeff'][b].to(self.device)
            predicted = torch.cat([raw[:, :80],
                prediction[b, :length] * self.std[80:144] + base[80:144], raw[:, 144:]], -1)
            count = min(self.settings['frames_per_clip'], length)
            indices = torch.linspace(0, length - 1, count).long().tolist()
            images = selected_frames(sample['video'], [0] + [sample['start'] + i for i in indices]).to(self.device)
            source, target = images[:1], images[1:]
            crop = batch['crop'][b, :length].to(self.device)
            semantic = torch.cat([predicted[:, 80:144], predicted[:, 224:227], predicted[:, 254:257],
                                  crop[:, :1], crop[:, 1:] * 256], -1)
            generated = []
            for i in indices:
                window = torch.arange(i - 13, i + 14, device=self.device).clamp(0, length - 1)
                # forward_train keeps gradients to semantic coefficients through the frozen renderer.
                generated.append(self.renderer.forward_train(source, semantic[window].T.unsqueeze(0)))
            generated = torch.cat(generated)
            predicted_landmarks, target_landmarks = self.landmarks(predicted[indices]), self.landmarks(raw[indices])
            landmark_loss = weighted_landmark_loss(predicted_landmarks, target_landmarks)
            # Preserve temporal adjacency for the mouth-motion loss; do not difference sparse frame samples.
            all_pred, all_target = self.landmarks(predicted), self.landmarks(raw)
            lip_delta = F.mse_loss((all_pred[1:] - all_pred[:-1])[:, 48:68],
                                  (all_target[1:] - all_target[:-1])[:, 48:68])
            emotion = F.mse_loss(self.emotion(generated), self.emotion(target).detach())
            image = F.smooth_l1_loss(generated, target)
            components.append(torch.stack([landmark_loss, lip_delta, emotion, image]))
        values = torch.stack(components).mean(0)
        weights = values.new_tensor([self.settings['landmark_weight'], self.settings['lip_delta_weight'],
                                     self.settings['emotion_loss_weight'], self.settings['image_weight']])
        return (values * weights).sum(), dict(zip(['landmark', 'lip_delta', 'emotion', 'image'], values))
