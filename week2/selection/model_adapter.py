"""Byte-token input adapter; no PyTorch dependency or model execution.

Use these inputs BEFORE the first model layer. Selecting only the final layer's
keys could leak information through hidden states computed from discarded code.
"""
from week2_mvp import Analysis

PAD_ID = 256
CLS_ID = 257
VOCAB_SIZE = 258


def pack_selected(a: Analysis, selected: set[int]) -> dict:
    if not a.local <= selected:
        raise ValueError("Every method must retain the fixed local/query blocks")
    if not selected <= set(range(len(a.blocks))):
        raise ValueError("Invalid block ID")
    ids, positions = [], []
    offset = 0
    for b, block in enumerate(a.blocks):
        if b in selected:
            ids.extend(block)
            positions.extend(range(offset, offset + len(block)))
        offset += len(block)
    # Explicit readout token after the code. Its position is consistent across methods.
    ids.append(CLS_ID)
    positions.append(len(a.raw))
    return {"input_ids": ids, "position_ids": positions,
            "readout_index": len(ids) - 1,
            "attention_note": "Use the same causal policy for all methods; pad separately"}
