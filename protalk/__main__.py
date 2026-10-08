"""Public command-line entry for the maintained ProTalk workflow."""
import argparse
import importlib
import sys

COMMANDS = {
    'prepare': ('protalk.training.prepare', 'Cache features and fit training statistics'),
    'manifest': ('protalk.training.manifest', 'Build a JSONL data manifest'),
    'extract': ('protalk.training.extract', 'Extract face coefficients from aligned videos'),
    'train': ('protalk.training.run', 'Train expression, VQ-VAE or pose sampler'),
    'generate': ('protalk.inference.__main__', 'Generate a talking-head video'),
    'doctor': ('protalk.training.doctor', 'Check dependencies and local assets'),
    'smoke': ('protalk.training.smoke', 'Exercise real networks using synthetic data'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Use protalk COMMAND --help for command options.')
    parser.add_argument('command', choices=COMMANDS, help='; '.join(f'{k}: {v[1]}' for k, v in COMMANDS.items()))
    args = parser.parse_args(sys.argv[1:2])
    sys.argv = [sys.argv[0] + ' ' + args.command] + sys.argv[2:]
    importlib.import_module(COMMANDS[args.command][0]).main()


if __name__ == '__main__':
    main()
