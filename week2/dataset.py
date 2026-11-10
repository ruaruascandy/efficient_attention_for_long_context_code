"""Balanced, generated Python output prediction. No external dataset downloads.

Labels are 0..9; template IDs, annotations, and answers are NEVER model inputs.
The three splits use disjoint operator-composition skeletons (0..3/4..5/6..7).
This small grammar is a diagnostic dataset, not a real-code benchmark.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import operator
import random
from collections import Counter
from pathlib import Path

FAMILIES = ('chain', 'call', 'branch')
TEMPLATES = {'train': (0, 1, 2, 3), 'validation': (4, 5), 'test': (6, 7)}


def interpret(code: str) -> int:
    """Evaluate ONLY the supported AST subset. No eval/exec, imports, or I/O."""
    env, functions = {}, {}
    binary = {ast.Add: operator.add, ast.Sub: operator.sub}
    comparisons = {ast.Gt: operator.gt, ast.LtE: operator.le}

    def expr(n, local):
        if isinstance(n, ast.Constant) and type(n.value) is int:
            return n.value
        if isinstance(n, ast.Name):
            return local[n.id] if n.id in local else env[n.id]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return (-1 if isinstance(n.op, ast.USub) else 1) * expr(n.operand, local)
        if isinstance(n, ast.BinOp) and type(n.op) in binary:
            return binary[type(n.op)](expr(n.left, local), expr(n.right, local))
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in comparisons:
            return comparisons[type(n.ops[0])](expr(n.left, local), expr(n.comparators[0], local))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and not n.keywords:
            fn = functions[n.func.id]
            if len(n.args) != len(fn.args.args):
                raise ValueError('Wrong argument count')
            args = {p.arg: expr(a, local) for p, a in zip(fn.args.args, n.args)}
            return statement(fn.body[0], args)
        raise ValueError(f'Unsupported expression: {type(n).__name__}')

    def statement(n, local):
        if isinstance(n, ast.Return):
            return expr(n.value, local)
        if isinstance(n, ast.If) and len(n.body) == len(n.orelse) == 1:
            return statement((n.body if expr(n.test, local) else n.orelse)[0], local)
        raise ValueError('Unsupported function body')

    for n in ast.parse(code).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
            if name in env or name in functions:
                raise ValueError('Rebinding is unsupported')
            env[name] = expr(n.value, {})
        elif isinstance(n, ast.FunctionDef):
            a = n.args
            if (n.decorator_list or n.returns or a.defaults or a.kwonlyargs or a.posonlyargs
                    or a.vararg or a.kwarg or len(n.body) != 1 or n.name in functions or n.name in env):
                raise ValueError('Unsupported function signature')
            # This interpreter is for local generated programs; recursion is disallowed.
            if any(isinstance(x, ast.Call) for x in ast.walk(n)):
                raise ValueError('Nested calls/recursion unsupported')
            functions[n.name] = n
        else:
            raise ValueError(f'Unsupported statement: {type(n).__name__}')
    return env['result']


def make_sample(seed: int, family: str, template: int, answer: int, length: int, split: str) -> dict:
    if family not in FAMILIES or template not in range(8) or answer not in range(10):
        raise ValueError('Invalid family/template/label')
    rng = random.Random(seed)
    names = iter(rng.sample([f'v{i:03}' for i in range(256)], 256))
    fresh = lambda: next(names)
    src, mid, out, fn, offset, flag, threshold = [fresh() for _ in range(7)]
    units = []
    def add(text, relevant=True):
        units.append((text, relevant))
    def sign(bit):
        return '-' if template & (1 << bit) else '+'
    def signed(x, bit):
        return -x if template & (1 << bit) else x

    if family == 'chain':
        values = [rng.randint(1, 3) for _ in range(3)]
        x = answer - sum(signed(v, i) for i, v in enumerate(values))
        add(f'{src} = {x}\n')
        add(f'{mid} = {src} {sign(0)} {values[0]}\n')
        add(f'{out} = {mid} {sign(1)} {values[1]}\n')
        final = f'result = {out} {sign(2)} {values[2]}\n'
    elif family == 'call':
        k, j = rng.randint(1, 3), rng.randint(1, 3)
        reverse = bool(template & 4)
        a = (-1 if template & 1 else 1) if reverse else 1
        b = k if reverse else signed(k, 0)
        b += signed(j, 1)
        x = (answer - b) // a
        add(f'{src} = {x}\n')
        add(f'{offset} = {k}\n')
        expression = f'{offset} {sign(0)} p' if reverse else f'p {sign(0)} {offset}'
        add(f'def {fn}(p):\n    return ({expression}) {sign(1)} {j}\n')
        add(f'{out} = {fn}({src})\n')
        final = f'result = {out}\n'
    else:
        cut, value, k = rng.randint(2, 7), rng.randint(0, 9), rng.randint(1, 3)
        comparison = '<=' if template & 4 else '>'
        taken = value <= cut if template & 4 else value > cut
        delta = signed(k, 0 if taken else 1)
        add(f'{src} = {answer - delta}\n')
        add(f'{offset} = {k}\n')
        add(f'{flag} = {value}\n')
        add(f'{threshold} = {cut}\n')
        add(f'def {fn}(p):\n    if {flag} {comparison} {threshold}:\n'
            f'        return p {sign(0)} {offset}\n    else:\n        return p {sign(1)} {offset}\n')
        add(f'{out} = {fn}({src})\n')
        final = f'result = {out}\n'

    # Place independent distractors in random gaps; preserve relevant execution order.
    remaining = length - sum(len(t.encode()) for t, _ in units) - len(final.encode())
    if remaining < 0:
        raise ValueError('Requested length is too short')
    while remaining >= 10:
        text = f'{fresh()} = {rng.randint(0, 9)}\n'
        units.insert(rng.randrange(len(units) + 1), (text, False))
        remaining -= len(text)
    # Whitespace is before the final assignment; never truncate a Python program.
    if remaining:
        units.append((' ' * (remaining - 1) + '\n', False))
    units.append((final, True))
    code, spans, cursor = '', [], 0
    for text, relevant in units:
        if relevant:
            spans.append([cursor, cursor + len(text.rstrip('\n').encode())])
        code += text
        cursor += len(text.encode())
    if cursor != length or interpret(code) != answer:
        raise AssertionError('Generated program/label mismatch')
    return {'id': f'{split}_{family}_t{template}_s{seed}_l{length}',
            'family_id': f'{split}_{family}_t{template}_s{seed}',
            'skeleton_id': f'{family}_t{template}', 'family': family,
            'split': split, 'seed': seed, 'code': code, 'target': 'result',
            'answer': answer, 'byte_length': length, 'gold_static_spans': spans,
            'max_dependency_distance_bytes': length - min(s for s, _ in spans)}


def generate(output: Path, train_reps=10, validation_reps=2, test_reps=3,
             train_lengths=(256,), eval_lengths=(256, 512, 1024), seed=20261110):
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'dataset_version': 'week1-v1', 'seed': seed,
                'task': 'Exact final value of result, integer classes 0..9',
                'tokenizer': 'UTF-8 bytes + CLS (lengths exclude CLS)',
                'split_protocol': 'disjoint operator-composition skeletons; small synthetic grammar only',
                'templates': TEMPLATES, 'splits': {}}
    for si, (split, reps) in enumerate([('train', train_reps), ('validation', validation_reps), ('test', test_reps)]):
        rows = []
        for fi, family in enumerate(FAMILIES):
            for template in TEMPLATES[split]:
                for rep in range(reps):
                    for label in range(10):
                        key = seed + si * 10000000 + fi * 1000000 + template * 100000 + rep * 10 + label
                        for length in train_lengths if split == 'train' else eval_lengths:
                            rows.append(make_sample(key, family, template, label, length, split))
        random.Random(seed + si).shuffle(rows)
        path = output / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8', newline='\n')
        manifest['splits'][split] = {'samples': len(rows), 'labels': dict(Counter(r['answer'] for r in rows)),
                                    'lengths': dict(Counter(r['byte_length'] for r in rows)),
                                    'skeleton_ids': sorted({r['skeleton_id'] for r in rows}),
                                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8', newline='\n')
    return manifest


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('data'))
    p.add_argument('--train-reps', type=int, default=10)
    p.add_argument('--validation-reps', type=int, default=2)
    p.add_argument('--test-reps', type=int, default=3)
    p.add_argument('--train-lengths', type=int, nargs='+', default=[256])
    p.add_argument('--eval-lengths', type=int, nargs='+', default=[256, 512, 1024])
    p.add_argument('--seed', type=int, default=20261110)
    a = p.parse_args()
    print(json.dumps(generate(a.output, a.train_reps, a.validation_reps, a.test_reps,
                              a.train_lengths, a.eval_lengths, a.seed), indent=2))
