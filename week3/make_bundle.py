"""Build the transfer bundle for the lab machine.

The bash one-liner is fine on Linux but not in cmd.exe: cmd has no `\\` line
continuation, single quotes are literal, so `--exclude='week3/_ref/qwen05b'`
matches nothing and tar happily compresses the 4 GB of local weights -- which
looks exactly like a hang.

This script builds the same bundle without needing a shell:

    python make_bundle.py            ->  dist/week3_lab.tgz   (~5 MB, POSIX paths)

Contents: week3/ (code, docs, tokenizer) + week1/dataset.py + week1/data/{test,validation}.jsonl
Excluded: _ref/qwen05b, _ref/qwen15b, results/, tmp_*, __pycache__, .hf
"""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "week3_lab.tgz"

EXCLUDE_DIRS = {"__pycache__", ".hf", "results", "_ref"}
KEEP_IN_REF = {"tokenizer"}
EXTRA = [
    "week1/dataset.py",
    "week1/data/test.jsonl",
    "week1/data/validation.jsonl",
]


def wanted(rel: Path) -> bool:
    parts = rel.parts
    if any(p.startswith("tmp_") for p in parts):
        return False
    if "__pycache__" in parts or ".hf" in parts or "results" in parts:
        return False
    if "_ref" in parts:
        i = parts.index("_ref")
        # keep _ref/tokenizer/**, drop _ref/qwen05b, _ref/qwen15b, _ref/fetch_model.py
        return len(parts) > i + 1 and parts[i + 1] in KEEP_IN_REF
    return True


def main() -> int:
    files: list[tuple[Path, str]] = []
    for p in sorted((ROOT / "week3").rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if wanted(rel):
            files.append((p, rel.as_posix()))
    for name in EXTRA:
        p = ROOT / name
        if not p.exists():
            print(f"missing required file: {name}", file=sys.stderr)
            return 1
        files.append((p, name))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OUT, "w:gz", format=tarfile.PAX_FORMAT) as tf:
        for path, arcname in files:
            info = tf.gettarinfo(str(path), arcname=arcname)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o755 if arcname.endswith(".sh") else 0o644
            with path.open("rb") as fh:
                tf.addfile(info, fh)

    size = OUT.stat().st_size / 2**20
    print(f"{OUT}  {size:.1f} MB  {len(files)} files")
    # sanity: nothing from the local weights, and the runner is executable
    with tarfile.open(OUT) as tf:
        names = tf.getnames()
        leaked = [n for n in names if "qwen05b" in n or "qwen15b" in n]
        mode = tf.getmember("week3/run_all.sh").mode
    if leaked:
        print("weights leaked into bundle:", leaked[:3], file=sys.stderr)
        return 1
    print(f"run_all.sh mode: {oct(mode)}   tokenizer files: {sum('_ref/tokenizer' in n for n in names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
