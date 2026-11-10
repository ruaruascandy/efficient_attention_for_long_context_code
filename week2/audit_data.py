"""Audit the actual shipped data, not just a few unit-test examples."""
import ast
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from dataset import load_rows, interpret
sys.path.insert(0, str(Path(__file__).parent / 'selection'))
from week2_mvp import analyze, span_blocks


def structural_signature(row):
    tree = ast.parse(row['code'])
    lines = row['code'].encode().splitlines(keepends=True)
    offsets, p = [], 0
    for line in lines:
        offsets.append(p)
        p += len(line)
    starts = {start for start, _ in row['gold_static_spans']}
    tree.body = [s for s in tree.body if offsets[s.lineno - 1] + s.col_offset in starts]
    names = {}
    def name(x):
        if x not in names:
            names[x] = f'n{len(names)}'
        return names[x]
    class Normalise(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.Name(id=name(node.id), ctx=node.ctx)
        def visit_FunctionDef(self, node):
            node.name = name(node.name)
            self.generic_visit(node)
            return node
        def visit_arg(self, node):
            node.arg = name(node.arg)
            return node
        def visit_Constant(self, node):
            return ast.Constant(value=1)
        def visit_UnaryOp(self, node):
            if isinstance(node.operand, ast.Constant):
                return ast.Constant(value=1)
            return self.generic_visit(node)
    return ast.dump(Normalise().visit(tree), include_attributes=False)


def main():
    root = Path('data')
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = {s: load_rows(root / f'{s}.jsonl') for s in ['train', 'validation', 'test']}
    report = {'checks': {}, 'sample_count': sum(map(len, rows.values())),
              'scope': 'Local generated fixed-grammar programs only; no arbitrary user code executed'}
    signatures = {}
    for split, samples in rows.items():
        assert hashlib.sha256((root / f'{split}.jsonl').read_bytes()).hexdigest() == manifest['splits'][split]['sha256']
        assert len({s['code'] for s in samples}) == len(samples)
        signatures[split] = {structural_signature(s) for s in samples}
        for sample in samples:
            # Independent CPython cross-check on every shipped generated program.
            namespace = {'__builtins__': {}}
            exec(compile(sample['code'], '<local-generated-audit>', 'exec'), namespace)
            assert namespace['result'] == sample['answer'] == interpret(sample['code'])
            a = analyze(sample['code'], 64)
            extracted = set().union(*(a.node_blocks[n] for n in a.distances))
            gold = set().union(*(span_blocks(x, y, 64) for x, y in sample['gold_static_spans']))
            assert extracted == gold
            assert len(sample['code'].encode()) == sample['byte_length']
        for length in {r['byte_length'] for r in samples}:
            for family in {r['family'] for r in samples}:
                c = Counter(r['answer'] for r in samples if r['byte_length'] == length and r['family'] == family)
                assert len(c) == 10 and len(set(c.values())) == 1
        report['checks'][split] = {'samples': len(samples), 'unique_normalised_relevant_ASTs': len(signatures[split]),
                                  'labels_match_python': True, 'annotations_match_week2_parser': True,
                                  'classes_balanced_within_family_and_length': True, 'sha256_matches': True,
                                  'byte_lengths_exact': True, 'exact_duplicates': 0}
    for x, y in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not signatures[x] & signatures[y]
        assert not {r['code'] for r in rows[x]} & {r['code'] for r in rows[y]}
        assert not {r['family_id'] for r in rows[x]} & {r['family_id'] for r in rows[y]}
    report['normalised_relevant_AST_overlap_between_splits'] = 0
    report['caveat'] = 'AST separation is syntactic; semantically equivalent programs can remain. This is limited grammar generalisation.'
    Path('results').mkdir(exist_ok=True)
    Path('results/data_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8', newline='\n')
    Path('examples').mkdir(exist_ok=True)
    for family in ['chain', 'call', 'branch']:
        sample = next(r for r in rows['train'] if r['family'] == family and r['answer'] == 7)
        Path(f'examples/{family}.py').write_text(sample['code'], encoding='utf-8', newline='\n')
        Path(f'examples/{family}.json').write_text(json.dumps(sample, indent=2), encoding='utf-8', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
