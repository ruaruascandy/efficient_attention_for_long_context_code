"""Week-3 tests: no model weights required (a tiny random Qwen2 stands in).

    python -m unittest -v test_week3
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w3_core as core  # noqa: E402
import w3_run  # noqa: E402

TOK = core.load_tokenizer()
DIDS = core.digit_ids(TOK)


class TestTokenizer(unittest.TestCase):
    def test_digits_are_single_tokens(self):
        self.assertEqual(len(DIDS), 10)
        self.assertEqual(len(set(DIDS)), 10)
        for d, i in zip(core.DIGITS, DIDS):
            self.assertEqual(TOK.convert_ids_to_tokens(i), d)

    def test_space_is_its_own_token(self):
        # The trap in the original plan: "# Output: " ends with a standalone space.
        ids = TOK.encode("# Output: ", add_special_tokens=False)
        self.assertEqual(TOK.convert_ids_to_tokens(ids[-1]), "\u0120")


class TestPrompt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = core.load_split("test")
        cls.row = next(r for r in rows if r["byte_length"] == 256)
        cls.code = cls.row["code"]

    def test_completion_prompt_map(self):
        p = core.build_prompt(TOK, self.code, "completion")
        ids, offsets = core.encode_prompt(TOK, p)
        self.assertTrue(p.force_space)
        self.assertEqual(len(ids), len(offsets))
        # the forced space token carries a zero-width offset
        self.assertEqual(offsets[-1], (len(p.text), len(p.text)))
        spans = core.span_token_positions(offsets, p.code_start, self.row["gold_static_spans"])
        self.assertTrue(all(spans))
        gold = sorted({t for s in spans for t in s})
        self.assertTrue(gold)
        # every mapped position must overlap a real gold span in the text
        for gi in gold:
            a, b = offsets[gi]
            self.assertTrue(any(a < p.code_start + e and b > p.code_start + s
                                for s, e in self.row["gold_static_spans"]), (gi, a, b))
        # positions are indices into ids, not vocabulary ids
        self.assertLess(max(gold), len(ids))

    def test_chat_prompt_map(self):
        p = core.build_prompt(TOK, self.code, "chat")
        ids, offsets = core.encode_prompt(TOK, p)
        self.assertFalse(p.force_space)
        spans = core.span_token_positions(offsets, p.code_start, self.row["gold_static_spans"])
        self.assertTrue(all(spans))
        self.assertGreater(p.code_start, 0)

    def test_span_inside_code(self):
        p = core.build_prompt(TOK, self.code, "completion")
        ids, offsets = core.encode_prompt(TOK, p)
        first = self.row["gold_static_spans"][0]
        pos = core.span_token_positions(offsets, p.code_start, [first])[0]
        text = TOK.decode([ids[i] for i in pos])
        self.assertIn("=", text)


class TestMasks(unittest.TestCase):
    def test_causal(self):
        for maker in (lambda n: core.mask_full(n), lambda n: core.mask_window(n, 4)):
            m = maker(9)
            for i in range(9):
                for j in range(9):
                    if j > i:
                        self.assertFalse(bool(m[i, j]))

    def test_window_counts(self):
        n = 10
        m = core.mask_window(n, 3)
        self.assertEqual(core.connection_count(m), sum(min(3, i + 1) for i in range(n)))

    def test_random_mask_matches_window_budget(self):
        # random=W must be a pure placement control: identical per-row degree.
        n = 60
        for w in (1, 4, 16):
            r = core.mask_random(n, w, seed=3)
            win = core.mask_window(n, w)
            self.assertTrue(torch.equal(r.sum(1), win.sum(1)), w)
            self.assertEqual(core.connection_count(r), core.connection_count(win), w)
            self.assertEqual(int((r & ~core.mask_full(n)).sum()), 0, w)  # causal only
            self.assertTrue(torch.equal(r, core.mask_random(n, w, seed=3)))  # reproducible
        self.assertFalse(torch.equal(core.mask_random(n, 8, seed=3),
                                     core.mask_random(n, 8, seed=4)))

    def test_window_receptive_field_matches_bfs(self):
        for w in (1, 2, 3, 5, 8):
            for layers in (1, 2, 3, 7):
                n = 40
                m = core.mask_window(n, w)
                hops = core.reach_hops(m, n - 1, layers)
                rf = core.receptive_field_window(w, layers)
                self.assertEqual(min(hops), max(0, n - rf), (w, layers, min(hops), rf))

    def test_block_diagonal_is_a_true_cutoff(self):
        # every position sits on its own line, blocks of 2 lines, 30 layers
        n = 40
        line_starts = list(range(0, n, 2))
        m = core.mask_line_block(n, line_starts, 2)
        hops = core.reach_hops(m, n - 1, 30)
        block_of_readout = (n - 1) // 2 // 2
        for tok in hops:
            self.assertEqual(tok // 2 // 2, block_of_readout)

    def test_global_bridges_blocks(self):
        n = 40
        line_starts = list(range(0, n, 2))
        m = core.mask_line_block(n, line_starts, 2)          # no bridge
        self.assertNotIn(0, core.reach_hops(m, n - 1, 30))
        m2 = core.mask_local_global(n, 2, [2, 20])           # globals bridge
        self.assertIn(0, core.reach_hops(m2, n - 1, 30))

    def test_gold_visibility_full(self):
        n = 20
        gv = core.gold_visibility(core.mask_full(n), [0, 5, 19], n - 1, 1)
        self.assertEqual(gv["gold_visible_frac"], 1.0)
        self.assertEqual(gv["gold_max_hops"], 1)

    def test_gold_visibility_window_cut(self):
        # W=2, 4 layers -> receptive field 5, so a 64-token sequence cannot relay
        # evidence from positions 0..2 to position 63.
        n = 64
        gv = core.gold_visibility(core.mask_window(n, 2), [0, 1, 2], n - 1, 4)
        self.assertEqual(gv["gold_visible"], 0)
        self.assertEqual(gv["gold_visible_frac"], 0.0)

    def test_line_starts(self):
        self.assertEqual(core.line_starts("a\nb\nc"), [0, 2, 4])
        self.assertEqual(core.line_starts("a\n"), [0])


class TestMutation(unittest.TestCase):
    def test_gold_value_changes_answer_and_length(self):
        row = next(r for r in core.load_split("test") if r["family"] == "chain")
        mut = core.mutate_gold_value(row["code"], row["gold_static_spans"])
        self.assertIsNotNone(mut)
        new_code, new_answer = mut
        self.assertEqual(len(new_code), len(row["code"]))
        self.assertNotEqual(new_answer, row["answer"])

    def test_drop_spans(self):
        row = next(r for r in core.load_split("test"))
        out = core.drop_spans(row["code"], row["gold_static_spans"])
        self.assertLess(len(out), len(row["code"]))
        self.assertEqual(len(out), len(row["code"]) - sum(e - s for s, e in row["gold_static_spans"]))

    def test_mutations_stay_in_the_10_way_readout(self):
        # Regression: mutating a literal used to push the label to 10 and crash p_gold.
        for r in core.load_split("test")[:60]:
            mut = core.mutate_gold_value(r["code"], r["gold_static_spans"])
            self.assertIsNotNone(mut, r["id"])
            code, answer = mut
            self.assertEqual(len(code), len(r["code"]), r["id"])
            self.assertTrue(0 <= answer <= 9, (r["id"], answer))
            self.assertNotEqual(answer, r["answer"], r["id"])


class TestStats(unittest.TestCase):
    def test_wilson(self):
        lo, hi = core.wilson(0, 10)
        self.assertAlmostEqual(lo, 0.0, places=5)
        self.assertLess(hi, 0.4)
        lo, hi = core.wilson(5, 10)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_mcnemar(self):
        self.assertEqual(core.mcnemar_exact(0, 0), 1.0)
        self.assertLess(core.mcnemar_exact(1, 20), 0.01)
        self.assertGreater(core.mcnemar_exact(10, 10), 0.9)

    def test_paired_delta(self):
        a = [1, 1, 1, 1, 0]
        b = [0, 0, 0, 0, 0]
        point, lo, hi = core.paired_delta_ci(a, b, iters=2000)
        self.assertAlmostEqual(point, 0.8)
        self.assertLessEqual(lo, point)
        self.assertGreaterEqual(hi, point)


class TestMaskedForward(unittest.TestCase):
    """A tiny random Qwen2 proves that the 4D mask reaches SDPA and means what we think."""

    @classmethod
    def setUpClass(cls):
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=512, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=1024, tie_word_embeddings=False)
        cls.model = Qwen2ForCausalLM(cfg).eval()
        cls.n = 24
        cls.ids = torch.randint(3, 500, (1, cls.n))

    def logits(self, mask=None):
        kw = {"use_cache": False}
        if mask is not None:
            kw["attention_mask"] = mask[None, None]
        with torch.no_grad():
            return self.model(input_ids=self.ids, **kw).logits[0, -1]

    def test_explicit_full_mask_equals_no_mask(self):
        a = self.logits(None)
        b = self.logits(core.mask_full(self.n))
        self.assertTrue(torch.allclose(a.float(), b.float(), atol=1e-4))

    def test_window_mask_changes_logits(self):
        a = self.logits(core.mask_full(self.n))
        b = self.logits(core.mask_window(self.n, 3))
        self.assertGreater((a.float() - b.float()).abs().max().item(), 1e-3)

    def test_self_only_mask_changes_logits(self):
        n = self.n
        m = torch.eye(n, dtype=torch.bool)
        a = self.logits(core.mask_full(n))
        b = self.logits(m)
        self.assertGreater((a.float() - b.float()).abs().max().item(), 1e-3)

    def test_digit_readout_shape(self):
        probs, margin, ent = w3_run.digit_scores(self.model, list(self.ids[0]), core.mask_full(self.n), "cpu", DIDS)
        self.assertEqual(probs.shape, (10,))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertGreaterEqual(margin, 0.0)
        self.assertGreater(ent, -1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
