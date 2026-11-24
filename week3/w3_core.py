"""Week-3 probe: model-independent pieces.

Everything here runs without model weights: data loading, Qwen tokenizer,
prompt construction, answer readout positions, byte-span -> token mapping,
attention masks, reachability analysis, and paired statistics.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

W3 = Path(__file__).resolve().parent
PROJECT = W3.parent
DATA_DIR = PROJECT / "week1" / "data"
TOK_DIR = W3 / "_ref" / "tokenizer"
DIGITS = tuple("0123456789")


# --------------------------------------------------------------------------- data
def load_split(split: str, data_dir: Path = DATA_DIR) -> list[dict]:
    path = Path(data_dir) / f"{split}.jsonl"
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    for r in rows:
        if any(ord(c) > 127 for c in r["code"]):
            raise ValueError(f"{r['id']}: non-ASCII source, byte offsets would not map")
    return rows


def load_tokenizer(tok_dir: Path = TOK_DIR):
    from transformers import PreTrainedTokenizerFast

    cfg = json.loads((Path(tok_dir) / "tokenizer_config.json").read_text(encoding="utf-8"))
    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(tok_dir) / "tokenizer.json"),
        chat_template=cfg.get("chat_template"),
        bos_token=cfg.get("bos_token"),
        eos_token=cfg.get("eos_token"),
        pad_token=cfg.get("pad_token"),
    )
    return tok


def digit_ids(tok) -> list[int]:
    """The ten single-token digits. Qwen's pre-tokenizer splits every digit, so
    this is valid -- but assert it rather than trust it."""
    ids = []
    for d in DIGITS:
        enc = tok.encode(d, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"digit {d!r} is not a single token: {enc}")
        ids.append(enc[0])
    if len(set(ids)) != 10:
        raise ValueError("digits do not map to ten distinct token ids")
    return ids


# ------------------------------------------------------------------------ prompts
@dataclass
class Prompt:
    text: str            # the string that goes into the tokenizer
    code_start: int      # char offset where the program starts
    code_end: int        # char offset where the program ends (exclusive)
    force_space: bool    # append one "臓" token before reading the answer
    style: str


def build_prompt(tok, code: str, style: str = "completion") -> Prompt:
    """Two fixed surfaces; pick one on validation, report both counts separately."""
    body = code.rstrip("\n")
    if style == "completion":
        # The trailing space of "# Output: " is its own token (id 25), so it is
        # appended as an input token instead of being scored as part of the prefix.
        return Prompt(body + "\nprint(result)\n# Output:", 0, len(body), True, style)
    if style == "chat":
        msgs = [
            {"role": "system", "content": "You are a precise Python interpreter."},
            {"role": "user", "content": (
                f"{body}\n\nWhat is the final integer value of {chr(96)}result{chr(96)}? "
                "Reply with a single digit 0-9."
            )},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        start = text.index(body)
        return Prompt(text, start, start + len(body), False, style)
    raise ValueError(f"unknown prompt style: {style}")


def encode_prompt(tok, prompt: Prompt) -> tuple[list[int], list[tuple[int, int]]]:
    enc = tok(prompt.text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = list(enc["offset_mapping"])
    if prompt.force_space:
        space_id = tok.encode(" ", add_special_tokens=False)
        if len(space_id) != 1:
            raise ValueError("space is not a single token")
        ids.append(space_id[0])
        offsets.append((len(prompt.text), len(prompt.text)))
    return ids, offsets


def span_token_positions(
    offsets: Sequence[tuple[int, int]],
    prompt_code_start: int,
    spans: Iterable[Sequence[int]],
) -> list[list[int]]:
    """Map byte spans (== char spans, ASCII asserted) to sequence positions.

    These are positions in the encoded prompt, not vocabulary ids.
    """
    out = []
    for start, end in spans:
        s, e = prompt_code_start + int(start), prompt_code_start + int(end)
        hits = [i for i, (a, b) in enumerate(offsets) if b > a and a < e and b > s]
        out.append(hits)
    return out


# -------------------------------------------------------------------------- masks
def _tri(n: int, device=None):
    import torch

    i = torch.arange(n, device=device)[:, None]
    j = torch.arange(n, device=device)[None, :]
    return i, j


def mask_full(n: int, device=None):
    i, j = _tri(n, device)
    return j <= i


def mask_window(n: int, w: int, device=None):
    if w < 1:
        raise ValueError("window must be >= 1")
    i, j = _tri(n, device)
    return (j <= i) & (j > i - w)


def mask_local_global(n: int, w: int, globals_idx: Sequence[int], device=None):
    """Longformer-style: globals see all previous tokens and are seen by all later ones."""
    m = mask_window(n, w, device)
    m = m.clone()
    for g in sorted({int(x) for x in globals_idx if 0 <= int(x) < n}):
        m[g, : g + 1] = True   # a global token reads every earlier token
        m[g:, g] = True        # every later token reads the global token
    return m


def mask_line_block(n: int, line_starts: Sequence[int], block_lines: int, sinks: int = 0, device=None):
    """Block-diagonal over groups of `block_lines` source lines: a true cutoff.

    Consecutive blocks never see each other, so stacking L layers cannot relay
    information across blocks (unlike a sliding window).
    """
    import torch

    line_of = torch.zeros(n, dtype=torch.long, device=device)
    starts = list(line_starts) + [n]
    for ln in range(len(starts) - 1):
        line_of[starts[ln]:starts[ln + 1]] = ln
    block_of = line_of // max(1, int(block_lines))
    i, j = _tri(n, device)
    m = (block_of[:, None] == block_of[None, :]) & (j <= i)
    if sinks:
        s = torch.arange(min(sinks, n), device=device)
        m[s, :] = j[s, :] <= i[s, :]
        m[:, s] = j[:, s] <= i[:, s]
    return m


def line_starts(code: str) -> list[int]:
    """Char offset of every '\n' + which starts a line, as token-agnostic layout.

    Returned in characters; the caller converts to token positions via offsets.
    """
    starts = [0]
    for k, ch in enumerate(code):
        if ch == "\n" and k + 1 < len(code):
            starts.append(k + 1)
    return starts


def char_to_token_starts(code_starts: Sequence[int], offsets: Sequence[tuple[int, int]], base: int) -> list[int]:
    """First token index at or after each source-line character offset."""
    out = []
    for cs in code_starts:
        target = base + int(cs)
        idx = next((i for i, (a, b) in enumerate(offsets) if b > target), len(offsets) - 1)
        out.append(idx)
    return out


def connection_count(mask) -> int:
    return int(mask.sum().item())


# --------------------------------------------------------------------- reachability
def reach_hops(mask, readout: int, max_hops: int) -> dict[int, int]:
    """Exact influence reachability over a *constant* per-layer mask.

    mask[i, j] is True when position i may attend to j. Information flows from j
    to i, so position j can influence the readout after k hops when there is a
    chain j -> ... -> readout of length k. Residual and MLP are position-wise,
    so they do not move information between positions.

    Returns {token_index: minimum hops} for every token that can reach `readout`
    within `max_hops` layers.
    """
    import numpy as np

    if hasattr(mask, "detach"):
        mask = mask.detach().cpu()
    m = np.asarray(mask, dtype=bool)
    n = m.shape[0]
    packed = np.packbits(m, axis=1)  # (n, ceil(n/8)); bit order is symmetric under pack/unpack

    hops = {int(readout): 0}
    seen = np.zeros(n, dtype=bool)
    seen[readout] = True
    frontier = np.array([int(readout)])

    for depth in range(1, int(max_hops) + 1):
        nxt = np.unpackbits(np.bitwise_or.reduce(packed[frontier], axis=0))[:n].astype(bool)
        nxt &= ~seen
        idx = np.nonzero(nxt)[0]
        if idx.size == 0:
            break
        for i in idx:
            hops[int(i)] = depth
        seen |= nxt
        frontier = idx
    return hops


def gold_visibility(mask, gold_tokens: Iterable[int], readout: int, layers: int) -> dict:
    hops = reach_hops(mask, readout, layers)
    gold = [int(t) for t in gold_tokens]
    reached = [t for t in gold if t in hops]
    direct = [t for t in gold if hops.get(t) == 1]
    return {
        "gold_n": len(gold),
        "gold_visible": len(reached),
        "gold_visible_frac": len(reached) / max(1, len(gold)),
        "gold_direct": len(direct),
        "gold_direct_frac": len(direct) / max(1, len(gold)),
        "gold_min_hops": min((hops[t] for t in reached), default=None),
        "gold_max_hops": max((hops[t] for t in reached), default=None),
        "unreachable_gold": len(gold) - len(reached),
    }


def receptive_field_window(window: int, layers: int) -> int:
    """Tokens visible to the final position of a causal sliding-window stack."""
    return 1 + int(layers) * (int(window) - 1)


# ------------------------------------------------------------------------- mutation
def lengthen_program(
    code: str,
    gold_spans: Sequence[Sequence[int]],
    target_bytes: int,
    answer: int,
    seed: int = 0,
    prefix: str = "w",
) -> dict:
    """Pad an existing program to `target_bytes` with new irrelevant assignments.

    The week-1 generator has a 256-name pool and cannot reach 4 KB, so longer
    variants are built here by inserting fresh `wNNNNN = d` lines before the
    final `result = ...` line. The gold spans are shifted, never recomputed, and
    the label is re-derived with the week-1 interpreter as a hard check.
    """
    import random
    import sys

    sys.path.insert(0, str(PROJECT / "week1"))
    from dataset import interpret  # noqa: E402

    if target_bytes < len(code):
        raise ValueError("lengthening can only grow a program")
    rng = random.Random(seed)
    lines = code.splitlines(keepends=True)
    if not lines[-1].startswith("result = "):
        raise ValueError("last line is not the result assignment")
    body = lines[:-1]

    line_offsets, cursor = [], 0
    for ln in lines:
        line_offsets.append(cursor)
        cursor += len(ln.encode())

    inserts: dict[int, list[str]] = {}
    # Never insert inside a multi-line gold statement (the function body); only at
    # line starts that lie strictly between statements.
    allowed_pos = [
        i for i in range(len(body) + 1)
        if not any(int(s) < line_offsets[i] < int(e) for s, e in gold_spans)
    ]
    if not allowed_pos:
        raise ValueError("no legal insertion point")
    remaining = target_bytes - len(code.encode())
    used = set()
    n = 0
    while remaining >= 10:
        name = f"{prefix}{n:05d}"
        n += 1
        if name in used or name in code:
            continue
        used.add(name)
        text = f"{name} = {rng.randint(0, 9)}\n"
        if len(text.encode()) > remaining:
            break
        pos = rng.choice(allowed_pos)
        inserts.setdefault(pos, []).append(text)
        remaining -= len(text.encode())
    if remaining >= 1:
        # A line of (remaining - 1) spaces is exactly `remaining` bytes.
        inserts.setdefault(rng.choice(allowed_pos), []).append(" " * max(0, remaining - 1) + "\n")
        remaining = 0

    new_parts, mapping, shift = [], [], 0
    for i, ln in enumerate(body):
        for text in inserts.get(i, []):
            new_parts.append(text)
            shift += len(text.encode())
        mapping.append((line_offsets[i], shift))
        new_parts.append(ln)
    for text in inserts.get(len(body), []):
        new_parts.append(text)
        shift += len(text.encode())
    mapping.append((line_offsets[-1], shift))
    new_parts.append(lines[-1])

    new_code = "".join(new_parts)
    old_starts = [m[0] for m in mapping]
    import bisect

    new_spans = []
    for start, end in gold_spans:
        k = max(0, bisect.bisect_right(old_starts, int(start)) - 1)
        delta = mapping[k][1]
        new_spans.append([int(start) + delta, int(end) + delta])
        if new_code[new_spans[-1][0]:new_spans[-1][1]] != code[int(start):int(end)]:
            raise AssertionError("gold span shifted incorrectly")
    if interpret(new_code) != int(answer):
        raise AssertionError("lengthened program no longer computes the label")
    return {"code": new_code, "gold_static_spans": new_spans,
            "byte_length": len(new_code.encode()), "answer": int(answer)}


def mutate_gold_value(code: str, spans: Sequence[Sequence[int]]) -> tuple[str, int] | None:
    """Length-preserving counterfactual: change one integer literal inside the
    first gold span and re-derive the label with the week-1 interpreter."""
    import re
    import sys

    sys.path.insert(0, str(PROJECT / "week1"))
    from dataset import interpret  # noqa: E402

    for start, end in spans:
        chunk = code[int(start):int(end)]
        nums = list(re.finditer(r"(?<![A-Za-z_])\d+", chunk))
        if not nums:
            continue
        m = nums[-1]
        old = m.group(0)
        new = str((int(old) + 1) % 10)
        cand = code[:int(start)] + chunk[:m.start()] + new + chunk[m.end():] + code[int(end):]
        if len(cand) != len(code):
            continue
        try:
            return cand, int(interpret(cand))
        except Exception:
            continue
    return None


def drop_spans(code: str, spans: Sequence[Sequence[int]]) -> str:
    keep, cursor = [], 0
    for start, end in sorted((int(s), int(e)) for s, e in spans):
        if start < cursor:
            continue
        keep.append(code[cursor:start])
        cursor = end
    keep.append(code[cursor:])
    return "".join(keep)


# --------------------------------------------------------------------------- stats
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    if b + c == 0:
        return 1.0
    n = b + c
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_delta_ci(correct_a: Sequence[int], correct_b: Sequence[int], iters: int = 10000, seed: int = 0):
    """Item-level bootstrap CI for accuracy(a) - accuracy(b), paired."""
    import random

    if len(correct_a) != len(correct_b):
        raise ValueError("paired samples must have equal length")
    rng = random.Random(seed)
    n = len(correct_a)
    d = [int(a) - int(b) for a, b in zip(correct_a, correct_b)]
    point = sum(d) / n
    means = []
    for _ in range(iters):
        means.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters) - 1]
    return point, lo, hi


def accuracy(correct: Sequence[int]) -> float:
    return sum(correct) / max(1, len(correct))
