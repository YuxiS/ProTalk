import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from training.data import Moments, validate_sample
from training.objectives import expression_loss, sampler_loss
from training.smoke import smoke


class TrainingTests(unittest.TestCase):
    def test_masked_losses_ignore_padding_and_its_gradients(self):
        prediction = torch.randn(2, 12, 64, requires_grad=True)
        target = torch.randn_like(prediction)
        expected, _ = expression_loss(prediction, target, [8, 12])
        altered = target.clone(); altered[0, 8:] = 1e8
        actual, _ = expression_loss(prediction, altered, [8, 12])
        self.assertTrue(torch.equal(expected, actual))
        actual.backward()
        self.assertTrue(torch.equal(prediction.grad[0, 8:], torch.zeros_like(prediction.grad[0, 8:])))
        logits = torch.randn(2, 12, 4, requires_grad=True)
        codes = torch.zeros(2, 3, dtype=torch.long)
        loss, _ = sampler_loss(logits, codes, [8, 12])
        altered_codes = codes.clone(); altered_codes[0, 2] = 3
        other, _ = sampler_loss(logits, altered_codes, [8, 12])
        self.assertTrue(torch.equal(loss, other))

    def test_streaming_statistics_and_constant_channels(self):
        moments = Moments()
        values = np.asarray([[1., 2.], [3., 2.], [5., 2.]])
        moments.update(values[:1]); moments.update(values[1:])
        mean, std = moments.result()
        np.testing.assert_allclose(mean, values.mean(0))
        self.assertEqual(std[1], 1.)
        self.assertAlmostEqual(std[0], values.std(0)[0], places=6)

    def test_invalid_time_alignment_is_rejected(self):
        sample = {'mfcc': np.zeros((8, 244)), 'prosody': np.zeros((8, 2)),
                  'coeff': np.zeros((8, 257)), 'crop': np.zeros((8, 3))}
        with self.assertRaises(ValueError):
            validate_sample(sample)

    def test_real_models_train_resume_and_export(self):
        with tempfile.TemporaryDirectory(prefix='protalk-training-test-') as directory:
            outputs = smoke(directory)
            import json
            metrics = [json.loads(line) for line in (outputs['vqvae'] / 'metrics.jsonl').read_text().splitlines()]
            self.assertLess(metrics[0]['validation_loss'], 100.)
            state = torch.load(outputs['expression'] / 'best-training.pth', map_location='cpu')
            self.assertEqual(state['config']['expression']['gst_init'], 'scratch')
            self.assertFalse(state['partial_epoch'])

    def test_raw_audio_features_have_the_inference_dimensions(self):
        from scipy.io.wavfile import write
        from training.prepare import audio_features
        from hparams import create_hparams
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tone.wav'
            samples = np.sin(2 * np.pi * 220 * np.arange(6000) / 22050).astype(np.float32)
            write(path, 22050, samples)
            mfcc, prosody = audio_features(str(path), 8, create_hparams())
            self.assertEqual(mfcc.shape, (8, 244))
            self.assertEqual(prosody.shape, (24, 2))
            self.assertTrue(np.isfinite(mfcc).all() and np.isfinite(prosody).all())

    def test_frozen_renderer_still_backpropagates_to_semantics(self):
        torch.set_num_threads(1)
        from face_utils.renders import PIRenderFaceGenerator
        renderer = PIRenderFaceGenerator().requires_grad_(False).eval()
        semantics = torch.randn(1, 73, 27, requires_grad=True)
        image = torch.randn(1, 3, 64, 64)
        output = renderer.forward_train(image, semantics)
        output.square().mean().backward()
        self.assertIsNotNone(semantics.grad)
        self.assertGreater(float(semantics.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in renderer.parameters()))
        with torch.no_grad():
            inference_output = renderer(image, semantics.detach())
        self.assertTrue(torch.equal(output.detach(), inference_output))


if __name__ == '__main__':
    unittest.main()
