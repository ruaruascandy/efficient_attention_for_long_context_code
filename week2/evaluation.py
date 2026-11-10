"""Model-only and preprocessing-inclusive timing; CPU/GPU results are distinct."""
from __future__ import annotations
import csv
import json
import platform
import statistics
import time
from collections import Counter
from pathlib import Path
import psutil
import torch
from torch.nn import functional as F
from model import collate_packed, pack_code


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def environment(device, threads):
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'torch': str(torch.__version__), 'device': str(device), 'threads': threads,
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
            'cpu': platform.processor() or platform.machine(),
            'precision': 'float32', 'torch_compile': False}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def predict_rows(model, rows, batch_size=32, device='cpu', mode='full', window=64):
    model.eval()
    predictions, losses = [], []
    # Bucket by source length to avoid an unnecessarily padded mixed-length batch.
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]['code'].encode()))
    output = {}
    for start in range(0, len(order), batch_size):
        ix = order[start:start + batch_size]
        items = [rows[i] for i in ix]
        batch = collate_packed([pack_code(x['code']) for x in items], device)
        y = torch.tensor([x['answer'] for x in items], device=device)
        logits = model(*batch, mode=mode, window=window)
        pred = logits.argmax(-1).cpu().tolist()
        loss = F.cross_entropy(logits, y, reduction='none').cpu().tolist()
        for i, p, l in zip(ix, pred, loss):
            output[i] = (p, l)
    for i in range(len(rows)):
        p, l = output[i]
        predictions.append(p)
        losses.append(l)
    return predictions, statistics.mean(losses)


def wilson(correct, n):
    z, p = 1.96, correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / (1 + z * z / n)
    return centre - half, centre + half


@torch.inference_mode()
def benchmark(model, rows, device, mode='full', window=64, batch_size=16, repeats=10):
    model.eval()
    rows = rows[:batch_size]
    batch = collate_packed([pack_code(x['code']) for x in rows], device)
    for _ in range(3):
        model(*batch, mode=mode, window=window)
    synchronize(device)
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    inference_ms, pipeline_ms = [], []
    for _ in range(repeats):
        synchronize(device)
        t = time.perf_counter()
        logits = model(*batch, mode=mode, window=window)
        synchronize(device)
        inference_ms.append((time.perf_counter() - t) * 1000)
        synchronize(device)
        t = time.perf_counter()
        packed = collate_packed([pack_code(x['code']) for x in rows], device)
        model(*packed, mode=mode, window=window).argmax(-1).cpu().tolist()
        synchronize(device)
        pipeline_ms.append((time.perf_counter() - t) * 1000)
    length = batch[0].shape[1]
    c = model.config
    links = sum(min(i + 1, window) if mode == 'window' else i + 1 for i in range(length))
    return {'batch_size': len(rows), 'sequence_tokens_including_cls': length,
            'median_model_batch_ms': statistics.median(inference_ms),
            'median_model_amortized_ms_per_sample': statistics.median(inference_ms) / len(rows),
            'median_pipeline_batch_ms': statistics.median(pipeline_ms),
            'samples_per_second': 1000 * len(rows) / statistics.median(inference_ms),
            'logical_attention_connections_per_sample_all_layers_heads': links * c.layers * c.heads,
            'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20 if str(device).startswith('cuda') else None,
            'cpu_process_rss_mib_after': psutil.Process().memory_info().rss / 2**20,
            'timing_repeats': repeats, 'warmup_batches': 3,
            'execution': 'dense SDPA with boolean mask; NOT a sparse kernel',
            'pipeline_scope': 'UTF-8 encoding + CPU packing + device transfer + mask + model + argmax; excludes disk I/O',
            'model_batch_ms_raw': inference_ms, 'pipeline_batch_ms_raw': pipeline_ms}
