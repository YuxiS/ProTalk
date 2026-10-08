import argparse
import ast
import subprocess
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from protalk.config import create_hparams
from protalk.runtime import align_coefficients, minmax_normalize, str2bool
from protalk.evaluation.beat_scores import bas, sbas
from protalk.models.pose.vqvae.quantizer import VectorQuantizer, VectorQuantizerEMA
from protalk.models.pose.generate import VAE
from protalk.models.pose.sampler.posesample import PoseSampler

ROOT = Path(__file__).resolve().parents[1]


def source_function(relative_path, name, namespace):
    """Exercise the original math without loading unrelated GPU/face dependencies."""
    tree = ast.parse((ROOT / relative_path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative_path, "exec"), namespace)
    return namespace[name]


class RuntimeTests(unittest.TestCase):
    def test_boolean_false_is_false(self):
        self.assertFalse(str2bool("False"))
        self.assertTrue(str2bool("true"))
        with self.assertRaises(argparse.ArgumentTypeError):
            str2bool("maybe")

    def test_config_has_no_missing_text_dependency(self):
        hp = create_hparams()
        self.assertEqual(hp.PoseModel.pose_dim, 9)
        self.assertEqual(hp.PoseModel.n_embeddings, 1024)

    def test_normalization_constant_and_normal_signal(self):
        self.assertTrue(torch.equal(minmax_normalize(torch.zeros(10)), torch.zeros(10)))
        x = torch.tensor([2., 4., 6.])
        self.assertTrue(torch.equal(minmax_normalize(x), torch.tensor([0., .5, 1.])))

    def test_alignment_trims_and_rejects_short_input(self):
        exp, pose = align_coefficients(torch.zeros(1, 10, 64), torch.zeros(1, 8, 9))
        self.assertEqual(exp.shape[1], pose.shape[1])
        self.assertEqual(exp.shape[1], 8)
        with self.assertRaises(ValueError):
            align_coefficients(exp[:, :4], pose)

    def test_bas_and_sbas_preserve_historical_formulas(self):
        audio = np.array([0., 20., 30.])
        motion = np.array([0., 12.])
        historical = np.mean([np.exp(-np.min((motion - b)**2)/2/64) for b in audio])
        reverse = np.mean([np.exp(-np.min((audio - b)**2)/2/64) for b in motion])
        self.assertAlmostEqual(bas(audio, motion), historical)
        self.assertAlmostEqual(sbas(audio, motion), historical + reverse)
        self.assertEqual(sbas([0], [0]), 2.)
        with self.assertRaises(ValueError):
            bas([], [0])
        with self.assertRaises(ValueError):
            bas([0], [0], sigma=0)

    def test_gaussian_smoothing_keeps_channels_and_length(self):
        kernel_fn = source_function("protalk/inference/smoothing.py", "gaussian_kernel", {"np": np, "torch": torch, "nn": torch.nn})
        kernel = kernel_fn(2, 2, 5, std=3)
        self.assertTrue(torch.allclose(kernel.sum(-1), torch.ones(2, 1)))
        x = torch.stack([torch.ones(9), torch.full((9,), 3.)]).unsqueeze(0)
        y = F.conv1d(x, kernel, groups=2)
        y = torch.cat([y[:, :, :1].repeat(1, 1, 2), y, y[:, :, -1:].repeat(1, 1, 2)], dim=2)
        self.assertTrue(torch.allclose(x, y))

    def test_vector_quantizer_cpu_and_gradient(self):
        model = VectorQuantizer(4, 3, .25)
        x = torch.randn(2, 5, 3, requires_grad=True)
        quantized, loss, _ = model(x)
        (loss + quantized.sum()).backward()
        self.assertEqual(quantized.device.type, "cpu")
        self.assertIsNotNone(x.grad)

    def test_ema_and_sampler_cpu(self):
        model = VectorQuantizerEMA(4, 3, .25)
        x = torch.randn(2, 5, 3)
        self.assertTrue(torch.equal(model.ema_w, model.embedding.weight))
        quantized, loss, _ = model(x, istrain=False)
        self.assertEqual(quantized.shape, x.shape)
        self.assertTrue(torch.isfinite(loss))
        weight_id = id(model.embedding.weight)
        ema_id = id(model.ema_w)
        model(x, istrain=True)
        self.assertEqual(id(model.embedding.weight), weight_id)
        self.assertEqual(id(model.ema_w), ema_id)
        self.assertTrue(torch.isfinite(model.embedding.weight).all())
        generator = VAE(2, 4, 3, 16, 9)
        indices = torch.tensor([1, 3])
        self.assertTrue(torch.equal(generator.generate_samplers(indices), generator.codebook.embedding.weight[indices]))
        sampler = PoseSampler(2, 4, 16).eval()
        with torch.no_grad():
            output = sampler(torch.randn(2, 24, 2), [8, 6])
        self.assertEqual(tuple(output.shape), (2, 8, 4))

    def test_python_sources_and_shell_syntax(self):
        tracked = subprocess.check_output(["git", "ls-files", "*.py", "*.sh"], cwd=ROOT, text=True).splitlines()
        for relative in tracked:
            if ".ipynb_checkpoints" in relative:
                continue
            if relative.endswith(".py"):
                ast.parse((ROOT / relative).read_text(), filename=relative)
            else:
                subprocess.run(["bash", "-n", str(ROOT / relative)], check=True)


if __name__ == "__main__":
    unittest.main()
