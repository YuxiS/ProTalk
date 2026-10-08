"""Ensure reorganized entry points and imports work independently of old root files."""
import ast
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class LayoutTests(unittest.TestCase):
    def test_supported_modules_do_not_import_archive(self):
        for path in (ROOT / 'protalk').rglob('*.py'):
            if 'third_party' in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                self.assertFalse(any(name == 'archive' or name.startswith('archive.') for name in names), str(path))

    def test_installed_command_help_outside_checkout(self):
        from protalk.__main__ import COMMANDS
        with tempfile.TemporaryDirectory() as directory:
            for command in [None, *COMMANDS]:
                arguments = [sys.executable, '-m', 'protalk']
                if command:
                    arguments.append(command)
                result = subprocess.run(arguments + ['--help'], cwd=directory, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)
            result = subprocess.run([sys.executable, '-c',
                'from protalk.config import create_hparams; assert create_hparams().PoseModel.pose_dim == 9'],
                cwd=directory, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_deep3d_dynamic_loader_uses_its_package(self):
        from protalk.third_party.deep3d import models
        fake_class = type('FakeModel', (models.BaseModel,), {})
        module = types.ModuleType('fake_model')
        module.FakeModel = fake_class
        with patch.object(models.importlib, 'import_module', return_value=module) as load:
            self.assertIs(models.find_model_using_name('fake'), fake_class)
            load.assert_called_once_with('protalk.third_party.deep3d.models.fake_model')


if __name__ == '__main__':
    unittest.main()
