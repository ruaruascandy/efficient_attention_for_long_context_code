"""One frozen checkpoint, all test items; inference-time window is an ablation."""
import argparse
import json
from pathlib import Path
from collections import Counter
import torch
from dataset import load_rows
from model import load_model
from evaluation import predict_rows, benchmark, environment, wilson, write_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--modes', nargs='+', choices=['full', 'window'], default=['full', 'window'])
    p.add_argument('--window', type=int, default=64)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    model, metadata = load_model(a.checkpoint, a.device)
    if metadata.get('is_overfit_diagnostic'):
        p.error('Use the trained held-out baseline checkpoint, not the overfit diagnostic')
    train, test = load_rows(a.data / 'train.jsonl'), load_rows(a.data / 'test.jsonl')
    majority = Counter(r['answer'] for r in train).most_common(1)[0][0]
    a.output.mkdir(parents=True, exist_ok=True)
    summary, per_sample, timings = [], [], []
    for mode in a.modes:
        pred, loss = predict_rows(model, test, a.batch_size, a.device, mode, a.window)
        for row, prediction in zip(test, pred):
            per_sample.append({'id': row['id'], 'family_id': row['family_id'], 'family': row['family'],
                               'byte_length': row['byte_length'], 'answer': row['answer'],
                               'prediction': prediction, 'correct': int(prediction == row['answer']),
                               'mode': mode, 'seed': metadata['seed'],
                               'training_mode': metadata['mode']})
        for length in sorted({r['byte_length'] for r in test}):
            rows_at_length = [r for r in test if r['byte_length'] == length]
            timing = benchmark(model, rows_at_length, a.device, mode, a.window, a.batch_size)
            timings.append({'mode': mode, 'byte_length': length, 'seed': metadata['seed'], **timing})
            for family in ['all', 'chain', 'call', 'branch']:
                selected = [x for x in per_sample if x['mode'] == mode and x['byte_length'] == length and (family == 'all' or x['family'] == family)]
                correct = sum(x['correct'] for x in selected)
                low, high = wilson(correct, len(selected))
                summary.append({'seed': metadata['seed'], 'training_mode': metadata['mode'],
                    'inference_mode': mode, 'family': family, 'byte_length': length, 'n': len(selected),
                    'correct': correct, 'accuracy': correct / len(selected), 'wilson95_low': low, 'wilson95_high': high,
                    'train_majority_accuracy': sum(x['answer'] == majority for x in selected) / len(selected),
                    'uniform_random_expected_accuracy': .1,
                    'interpretation': 'trained baseline' if mode == metadata['mode'] else 'inference-only mask ablation; no retraining',
                    'length_regime': 'trained length' if length == 256 else 'untrained length extrapolation',
                    'median_model_amortized_ms_per_sample': timing['median_model_amortized_ms_per_sample'],
                    'gpu_peak_allocated_mib': timing['gpu_peak_allocated_mib']})
    write_csv(a.output / 'summary.csv', summary)
    write_csv(a.output / 'predictions.csv', per_sample)
    (a.output / 'timings.json').write_text(json.dumps(timings, indent=2), encoding='utf-8')
    (a.output / 'evaluation_config.json').write_text(json.dumps({
        'checkpoint': str(a.checkpoint), 'selected_epoch': metadata['selected_epoch'],
        'selected_validation_accuracy': metadata['validation_accuracy'],
        'environment': environment(a.device, a.threads), 'window': a.window,
        'note': 'Length variants share family_id; do not treat lengths as independent samples.'}, indent=2), encoding='utf-8')
    for r in summary:
        if r['family'] == 'all':
            print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
