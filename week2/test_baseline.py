"""Checks intended to catch label, split, masking, position and checkpoint bugs."""
import json
import random
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
import torch
from dataset import FAMILIES, TEMPLATES, generate, interpret, load_rows, make_sample
from model import (ModelConfig, TinyCodeTransformer, attention_mask, collate_packed,
                   pack_code, save_model, load_model)
sys.path.insert(0, str(Path(__file__).parent / 'selection'))
from week2_mvp import analyze, select, span_blocks
from model_adapter import pack_selected


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_labels_match_independent_python(self):
        # Only execute this test's locally generated fixed-grammar programs.
        for family in FAMILIES:
            for template in range(8):
                for label in range(10):
                    s = make_sample(7000 + 100 * template + label, family, template, label, 256, 'unit')
                    scope = {'__builtins__': {}}
                    exec(compile(s['code'], '<generated-unit-test>', 'exec'), scope)
                    self.assertEqual(scope['result'], label)
                    self.assertEqual(interpret(s['code']), label)
                    self.assertEqual(len(s['code'].encode()), 256)

    def test_splits_balance_deduplicate_and_group(self):
        with tempfile.TemporaryDirectory() as d:
            generate(Path(d), 1, 1, 1)
            rows = {s: load_rows(Path(d) / f'{s}.jsonl') for s in TEMPLATES}
            for s, samples in rows.items():
                self.assertEqual(len({x['code'] for x in samples}), len(samples))
                self.assertEqual(len(set(Counter(x['answer'] for x in samples).values())), 1)
            for x, y in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
                for field in ['skeleton_id', 'family_id', 'code']:
                    self.assertFalse({r[field] for r in rows[x]} & {r[field] for r in rows[y]})
            for samples in rows.values():
                grouped = {}
                for s in samples:
                    grouped.setdefault(s['family_id'], set()).add(s['answer'])
                self.assertTrue(all(len(v) == 1 for v in grouped.values()))

    def test_model_input_independent_of_answer_and_gold(self):
        s = make_sample(14, 'branch', 2, 7, 256, 'unit')
        before = pack_code(s['code'])
        s['answer'], s['gold_static_spans'] = 0, []
        self.assertEqual(before, pack_code(s['code']))
        self.assertEqual(set(before), {'input_ids', 'position_ids', 'readout_index'})

    def test_every_layer_causality(self):
        torch.manual_seed(1)
        m = TinyCodeTransformer().eval()
        ids, pos, _ = collate_packed([pack_code('x = 2\nresult = x\n')])
        changed = ids.clone()
        changed[:, 10] = (int(changed[0, 10]) + 1) % 256
        with torch.no_grad():
            for mode in ['full', 'window']:
                a, b = m.states(ids, pos, mode, 4), m.states(changed, pos, mode, 4)
                torch.testing.assert_close(a[:, :10], b[:, :10], rtol=0, atol=1e-6)

    def test_padding_does_not_change_logits(self):
        torch.manual_seed(2)
        m = TinyCodeTransformer().eval()
        a = pack_code('x = 1\nresult = x\n')
        b = pack_code('unused = 2\nx = 1\nresult = x\n')
        with torch.no_grad():
            one = m(*collate_packed([a]))[0]
            padded = m(*collate_packed([a, b]))[0]
        torch.testing.assert_close(one, padded, rtol=1e-5, atol=1e-6)

    def test_window_connections_and_all_layers(self):
        ids, _, _ = collate_packed([pack_code('abcd')])
        mask = attention_mask(ids, 'window', 2)[0, 0]
        self.assertEqual(int(mask.sum()), 9)
        self.assertFalse(mask[4, 2])
        self.assertTrue(mask[4, 3])
        self.assertFalse(mask[0, 1])
        m = TinyCodeTransformer().eval()
        # A four-layer width-2 network cannot carry token 0 into position 5.
        ids, pos, _ = collate_packed([pack_code('abcde')])
        changed = ids.clone()
        changed[0, 0] = ord('z')
        with torch.no_grad():
            torch.testing.assert_close(m.states(ids, pos, 'window', 2)[:, 5],
                m.states(changed, pos, 'window', 2)[:, 5], rtol=0, atol=1e-6)

    def test_backward_and_checkpoint_roundtrip(self):
        torch.manual_seed(3)
        m = TinyCodeTransformer()
        batch = collate_packed([pack_code('result = 3\n')])
        loss = torch.nn.functional.cross_entropy(m(*batch), torch.tensor([3]))
        loss.backward()
        for block in m.blocks:
            self.assertTrue(torch.isfinite(block.qkv.weight.grad).all())
            self.assertGreater(float(block.qkv.weight.grad.abs().sum()), 0)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'test.pt'
            save_model(path, m, {'unit_test': True})
            restored, meta = load_model(path)
            torch.testing.assert_close(m(*batch), restored(*batch), rtol=0, atol=0)
            self.assertTrue(meta['unit_test'])

    def test_week2_parser_gold_and_adapter_compatibility(self):
        for family in FAMILIES:
            for t in range(8):
                s = make_sample(300 + t, family, t, t, 512, 'unit')
                a = analyze(s['code'], block_size=64)
                actual = set().union(*(a.node_blocks[n] for n in a.distances))
                expected = set().union(*(span_blocks(x, y, 64) for x, y in s['gold_static_spans']))
                self.assertEqual(actual, expected)
                for method in ['lexical_topk', 'structure_topk', 'evidence_groups']:
                    selected = select(a, method, budget=128)
                    packed = pack_selected(a, selected)
                    self.assertEqual(packed['position_ids'][-1], 512)
                    self.assertEqual(packed['input_ids'][:-1], [a.raw[i] for i in packed['position_ids'][:-1]])
                    self.assertLessEqual(sum(len(a.blocks[i]) for i in selected - a.local), 128)
        m = TinyCodeTransformer().eval()
        with torch.no_grad():
            self.assertEqual(tuple(m(*collate_packed([packed])).shape), (1, 10))

    def test_interpreter_rejects_unsupported_code(self):
        for code in ['import os\nresult = 1', 'result = open("x")', 'result = [1][0]', 'x = 1\nx = 2\nresult = x']:
            with self.subTest(code=code), self.assertRaises((ValueError, KeyError)):
                interpret(code)


if __name__ == '__main__':
    unittest.main()
