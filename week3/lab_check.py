"""Lab-machine self-check. Run this first; it tells you exactly what will break.

    python lab_check.py
    python lab_check.py --model /data/models/Qwen2.5-Coder-7B-Instruct

Checks, in order: python/torch/transformers versions, GPU and VRAM, the data and
interpreter files this repo needs, the bundled tokenizer, and -- most important --
whether a 4D boolean attention mask is actually honoured by this transformers
version. That last one is a functional test on a tiny random Qwen2, so it is the
authoritative answer rather than a version-number guess.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OK, WARN, BAD = "PASS", "WARN", "FAIL"
rows: list[tuple[str, str, str]] = []


def add(status: str, what: str, detail: str) -> None:
    rows.append((status, what, detail))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="optional: local path to a Qwen checkpoint")
    a = ap.parse_args()

    add(OK, "python", f"{platform.python_version()} on {platform.platform()}")
    if sys.version_info < (3, 9):
        add(BAD, "python version", "3.9+ required")

    try:
        import torch

        add(OK, "torch", f"{torch.__version__}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            gb = p.total_memory / 2**30
            add(OK, "device", f"{p.name}, {gb:.1f} GB, {torch.cuda.device_count()} visible")
            if gb < 18:
                add(WARN, "vram", "7B bf16 needs ~17 GB with activations; use 3B or --dtype float16")
            if p.major < 8:
                add(WARN, "compute capability", f"sm_{p.major}{p.minor}: no bf16, use --dtype float16")
        else:
            add(WARN, "cuda", "not available; everything will run on CPU (very slow)")
    except Exception as e:  # noqa: BLE001
        add(BAD, "torch", f"import failed: {e}")
        return report()

    try:
        import transformers
        import tokenizers

        add(OK, "transformers/tokenizers", f"{transformers.__version__} / {tokenizers.__version__}")
        if not transformers.__version__.startswith("4.57"):
            add(WARN, "transformers version", "verified on 4.57.1; the mask test below decides")
    except Exception as e:  # noqa: BLE001
        add(BAD, "transformers", f"import failed: {e}")
        return report()

    # files this repo needs on the lab machine
    need = [HERE.parent / "week1" / "dataset.py",
            HERE.parent / "week1" / "data" / "test.jsonl",
            HERE.parent / "week1" / "data" / "validation.jsonl",
            HERE / "_ref" / "tokenizer" / "tokenizer.json"]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        add(BAD, "required files", "missing: " + ", ".join(missing))
        add(WARN, "fix", "copy week1/dataset.py, week1/data/*.jsonl and week3/_ref/tokenizer/ "
                          "(see LAB_SETUP_ZH.md); or pass --data-dir for the data only")
    else:
        add(OK, "required files", "week1 data + interpreter + tokenizer present")

    free = shutil.disk_usage(HERE).free / 2**30
    add(OK if free > 20 else WARN, "disk", f"{free:.1f} GB free here (7B weights need ~16 GB)")

    # tokenizer + prompt + readout surface
    try:
        import w3_core as core

        tok = core.load_tokenizer()
        dids = core.digit_ids(tok)
        add(OK, "tokenizer", f"{len(tok)} ids; digits are single tokens: {len(dids) == 10}")
        rows_ = core.load_split("test")
        sample = next(r for r in rows_ if r["byte_length"] == 256)
        prompt = core.build_prompt(tok, sample["code"], "completion")
        ids, _ = core.encode_prompt(tok, prompt)
        add(OK, "prompt", f"256 B program -> {len(ids)} tokens (completion style)")
        if len(tok.encode("# Output: ", add_special_tokens=False)) < 1:
            add(BAD, "readout", "cannot tokenise '# Output: '")
    except Exception as e:  # noqa: BLE001
        add(BAD, "tokenizer/prompt", f"{e}")
        return report()

    # the decisive check: does this stack honour a custom 4D boolean mask?
    try:
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=512, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=1024, tie_word_embeddings=False)
        m = Qwen2ForCausalLM(cfg).eval()
        ids_t = torch.randint(3, 500, (1, 24))
        n = 24

        def run(mask):
            kw = {} if mask is None else {"attention_mask": mask[None, None]}
            with torch.no_grad():
                return m(input_ids=ids_t, use_cache=False, **kw).logits[0, -1].float()

        base = run(None)
        full = run(core.mask_full(n))
        win = run(core.mask_window(n, 3))
        if not torch.allclose(base, full, atol=1e-4):
            add(BAD, "4D mask passthrough", "explicit full mask != no mask: this version rewrites masks")
        elif (base - win).abs().max().item() < 1e-3:
            add(BAD, "4D mask passthrough", "window mask has no effect: masks are being ignored")
        else:
            add(OK, "4D mask passthrough", "full==no-mask, window changes logits")
    except Exception as e:  # noqa: BLE001
        add(BAD, "4D mask passthrough", f"{e}")

    if a.model:
        p = Path(a.model)
        if p.exists():
            total = sum(f.stat().st_size for f in p.glob("*.safetensors")) / 2**30
            add(OK, "weights", f"{p} ({total:.1f} GB)")
        else:
            add(WARN, "weights", f"{p} not found (HF id? then it must download at load time)")

    return report()


def report() -> int:
    width = max(len(w) for _, w, _ in rows)
    for status, what, detail in rows:
        print(f"[{status}] {what:<{width}}  {detail}")
    failed = [r for r in rows if r[0] == BAD]
    print()
    if failed:
        print(f"{len(failed)} blocking problem(s). Fix those before downloading weights.")
        print("See LAB_SETUP_ZH.md -> Troubleshooting.")
        return 1
    print("Environment OK. Next: python -m unittest -v test_week3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
