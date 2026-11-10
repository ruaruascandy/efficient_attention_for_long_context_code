"""Four-layer causal Transformer; byte IDs match the existing Week-2 adapter."""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F

PAD_ID, CLS_ID, VOCAB_SIZE = 256, 257, 258


@dataclass
class ModelConfig:
    layers: int = 4
    hidden: int = 128
    heads: int = 4
    ffn: int = 512
    max_position: int = 4096
    classes: int = 10
    dropout: float = 0.0


def pack_code(code):
    ids = list(code.encode('utf-8')) + [CLS_ID]
    return {'input_ids': ids, 'position_ids': list(range(len(ids))), 'readout_index': len(ids) - 1}


def collate_packed(items, device='cpu'):
    max_length = max(len(x['input_ids']) for x in items)
    ids = torch.full((len(items), max_length), PAD_ID, dtype=torch.long)
    positions = torch.zeros_like(ids)
    readout = torch.empty(len(items), dtype=torch.long)
    for i, x in enumerate(items):
        n = len(x['input_ids'])
        if len(x['position_ids']) != n or not 0 <= x['readout_index'] < n:
            raise ValueError('Malformed packed input')
        ids[i, :n] = torch.tensor(x['input_ids'])
        positions[i, :n] = torch.tensor(x['position_ids'])
        readout[i] = x['readout_index']
    return tuple(x.to(device) for x in (ids, positions, readout))


def attention_mask(ids, mode='full', window=64):
    n = ids.shape[1]
    index = torch.arange(n, device=ids.device)
    distance = index[:, None] - index[None, :]
    allowed = distance >= 0
    if mode == 'window':
        if window < 1:
            raise ValueError('Window must be positive')
        allowed &= distance < window
    elif mode != 'full':
        raise ValueError(f'Unknown attention mode {mode}')
    # SDPA boolean True means allowed. This mask is applied in EVERY layer.
    return allowed[None, None] & (ids != PAD_ID)[:, None, None, :]


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.width, self.dropout = c.heads, c.hidden // c.heads, c.dropout
        self.ln1, self.ln2 = nn.LayerNorm(c.hidden), nn.LayerNorm(c.hidden)
        self.qkv, self.proj = nn.Linear(c.hidden, 3 * c.hidden), nn.Linear(c.hidden, c.hidden)
        self.ffn = nn.Sequential(nn.Linear(c.hidden, c.ffn), nn.GELU(), nn.Linear(c.ffn, c.hidden))

    def forward(self, x, mask):
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, n, 3, self.heads, self.width).permute(2, 0, 3, 1, 4).unbind(0)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
              dropout_p=self.dropout if self.training else 0.0)
        x = x + self.proj(h.transpose(1, 2).reshape(b, n, d))
        return x + self.ffn(self.ln2(x))


class TinyCodeTransformer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if c.hidden % c.heads or c.hidden % 2:
            raise ValueError('hidden must be even and divisible by heads')
        self.embedding = nn.Embedding(VOCAB_SIZE, c.hidden, padding_idx=PAD_ID)
        # Separate constructors give independent initialisation for all layers.
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm, self.classifier = nn.LayerNorm(c.hidden), nn.Linear(c.hidden, c.classes)
        pos = torch.arange(c.max_position + 1).float()[:, None]
        freq = torch.exp(torch.arange(0, c.hidden, 2).float() * (-math.log(10000.0) / c.hidden))
        pe = torch.zeros(c.max_position + 1, c.hidden)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
        self.register_buffer('positions', pe, persistent=False)

    def states(self, ids, position_ids, mode='full', window=64):
        if position_ids.min() < 0 or position_ids.max() > self.config.max_position:
            raise ValueError('Original source positions exceed configured maximum')
        mask = attention_mask(ids, mode, window)
        x = self.embedding(ids) + self.positions[position_ids]
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)

    def forward(self, ids, position_ids, readout_index, mode='full', window=64):
        h = self.states(ids, position_ids, mode, window)
        return self.classifier(h[torch.arange(len(ids), device=ids.device), readout_index])


def save_model(path, model, metadata):
    torch.save({'state_dict': model.state_dict(), 'config': asdict(model.config), 'metadata': metadata}, path)


def load_model(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = TinyCodeTransformer(ModelConfig(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint['metadata']
