"""Train the baseline. Reads train/validation only; NEVER opens test.jsonl."""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
import torch
from torch.nn import functional as F
from dataset import load_rows
from evaluation import environment, predict_rows, synchronize, write_csv
from model import TinyCodeTransformer, collate_packed, pack_code, save_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--mode', choices=['full', 'window'], default='full')
    p.add_argument('--window', type=int, default=64)
    p.add_argument('--overfit', type=int, default=0, help='Use N train rows as both train and validation, diagnostic only')
    p.add_argument('--steps', type=int, default=0, help='Override epoch schedule with fixed optimizer steps')
    args = p.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA device unavailable. Install an appropriate GPU PyTorch build or use --device cpu.')
    if min(args.epochs, args.batch_size, args.threads) < 1:
        p.error('Epochs, batch-size, threads must be positive')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    # CUDA deterministic matmul requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup.
    train = load_rows(args.data / 'train.jsonl')
    validation = [r for r in load_rows(args.data / 'validation.jsonl') if r['byte_length'] == 256]
    if args.overfit:
        # Balanced and reproducible small set, one per (family, label) before extras.
        selected, used = [], set()
        for row in train:
            key = (row['family'], row['answer'])
            if key not in used:
                selected.append(row)
                used.add(key)
        train = selected[:args.overfit]
        if len(train) != args.overfit:
            p.error('Overfit diagnostic supports at most 30 rows')
        validation = train
    args.output.mkdir(parents=True, exist_ok=True)
    model = TinyCodeTransformer().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    packed = [pack_code(r['code']) for r in train]
    per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = args.steps or args.epochs * per_epoch
    epochs = math.ceil(total_steps / per_epoch)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'model': asdict(model.config), 'parameters': sum(p.numel() for p in model.parameters()),
                   'training_samples': len(train), 'validation_samples': len(validation),
                   'total_steps': total_steps, 'environment': environment(args.device, args.threads),
                   'checkpoint_rule': 'lowest validation cross entropy (train loss for overfit diagnostic)',
                   'dataset_manifest': json.loads((args.data / 'manifest.json').read_text()),
                   'is_overfit_diagnostic': bool(args.overfit)})
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    initial, initial_loss = predict_rows(model, validation, args.batch_size, args.device, args.mode, args.window)
    initial_accuracy = sum(p == r['answer'] for p, r in zip(initial, validation)) / len(validation)
    history, step, best_loss = [], 0, float('inf')
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train)))
        random.Random(args.seed * 100000 + epoch).shuffle(order)
        loss_sum, correct, n = 0., 0, 0
        for offset in range(0, len(order), args.batch_size):
            if step >= total_steps:
                break
            ix = order[offset:offset + args.batch_size]
            batch = collate_packed([packed[i] for i in ix], args.device)
            y = torch.tensor([train[i]['answer'] for i in ix], device=args.device)
            # Fixed LR for tiny-set fitting; warmup + cosine for the held-out experiment.
            if args.overfit:
                lr = args.lr
            else:
                warmup = min(20, max(1, total_steps // 10))
                factor = (step + 1) / warmup if step < warmup else .1 + .9 * .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
                lr = args.lr * factor
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            logits = model(*batch, mode=args.mode, window=args.window)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(ix)
            correct += int((logits.argmax(-1) == y).sum().item())
            n += len(ix)
            step += 1
        # Do not evaluate all 30 tiny-set epochs individually: every 20 steps is enough.
        if args.overfit and step % 20 != 0 and step < total_steps:
            continue
        predictions, val_loss = predict_rows(model, validation, args.batch_size, args.device, args.mode, args.window)
        val_accuracy = sum(p == r['answer'] for p, r in zip(predictions, validation)) / len(validation)
        synchronize(args.device)
        row = {'epoch': epoch, 'step': step, 'train_loss_online': loss_sum / n,
               'train_accuracy_online': correct / n, 'validation_loss': val_loss,
               'validation_accuracy': val_accuracy, 'elapsed_seconds': time.perf_counter() - start, 'lr': lr}
        history.append(row)
        write_csv(args.output / 'history.csv', history)
        if val_loss < best_loss:
            best_loss = val_loss
            save_model(args.output / 'best.pt', model, {**config, 'selected_epoch': epoch, 'selected_step': step,
                        'validation_loss': val_loss, 'validation_accuracy': val_accuracy})
        print(json.dumps(row), flush=True)
    synchronize(args.device)
    summary = {'initial_validation_accuracy': initial_accuracy, 'initial_validation_loss': initial_loss,
               'best_validation_loss': best_loss, 'training_seconds_including_validation_and_checkpoints': time.perf_counter() - start,
               'optimizer_steps': step, 'history': history}
    (args.output / 'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
