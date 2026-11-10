"""Predict final result from source text with a saved neural classifier."""
import argparse
from pathlib import Path
import torch
from model import load_model, collate_packed, pack_code

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--code-file', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    m, _ = load_model(a.checkpoint, a.device)
    m.eval()
    with torch.inference_mode():
        prediction = m(*collate_packed([pack_code(a.code_file.read_text(encoding='utf-8'))], a.device)).argmax(-1).item()
    print(prediction)
