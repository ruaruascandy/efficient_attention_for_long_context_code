"""Reproduce the recorded baseline with three independent initialisation seeds.

All training finishes before the test set is opened by evaluate.py.
Existing run directories are rejected to preserve the original run evidence.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def invoke(args, logfile):
    print('Running:', ' '.join(str(x) for x in args), flush=True)
    with logfile.open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, '-u'] + [str(x) for x in args], stdout=log,
                       stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, default=Path('reproduced_runs'))
    p.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    for seed in a.seeds:
        directory = a.output / f'seed_{seed}'
        if directory.exists():
            p.error(f'Run already exists: {directory}. Choose a new --output path.')
    a.output.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        invoke(['train.py', '--data', a.data, '--output', a.output / f'seed_{seed}',
                '--seed', seed, '--epochs', a.epochs, '--threads', a.threads, '--device', a.device],
               a.output / f'train_seed_{seed}.log')
    for seed in a.seeds:
        directory = a.output / f'seed_{seed}'
        invoke(['evaluate.py', '--data', a.data, '--checkpoint', directory / 'best.pt',
                '--output', directory / 'evaluation', '--threads', a.threads, '--device', a.device],
               a.output / f'evaluate_seed_{seed}.log')
    print('Training and evaluation completed.', flush=True)
