"""Meaningful correctness checks for the restricted analysis and budget logic."""
import unittest

from model_adapter import CLS_ID, pack_selected
from week2_mvp import analyze, build_groups, generate_sample, select, span_blocks


class Week2Tests(unittest.TestCase):
    def test_argument_control_and_return_edges(self):
        code = ("a = 7\nlimit = 5\nbonus = 3\n"
                "def f(p):\n    if p > limit:\n        return p + bonus\n"
                "    else:\n        return p - bonus\n"
                "b = f(a)\nresult = b + 1\n")
        a = analyze(code)
        self.assertEqual(a.nodes["b"].deps, {"f": "call_return", "a": "argument"})
        self.assertEqual(a.nodes["f"].deps, {"limit": "control", "bonus": "return_value"})
        self.assertNotIn("p", a.nodes["f"].deps)

    def test_function_global_resolved_at_call(self):
        a = analyze("def f(p):\n    return p + offset\noffset = 3\nx = 2\nresult = f(x)\n")
        self.assertIn("offset", a.distances)

    def test_parameter_shadows_global(self):
        a = analyze("p = 99\nx = 2\ndef f(p):\n    return p + 1\nresult = f(x)\n")
        self.assertNotIn("p", a.distances)
        self.assertIn("x", a.distances)

    def test_calls_do_not_merge_arguments(self):
        a = analyze("x = 2\ny = 5\ndef f(p):\n    return p + 1\na = f(x)\nb = f(y)\nresult = a\n")
        self.assertNotIn("y", a.distances)
        self.assertNotIn("b", a.distances)
        self.assertIn("x", a.distances)

    def test_generated_labels_and_independent_gold(self):
        for i in range(90):
            s = generate_sample(i, ["chain", "call", "branch"][i % 3])
            # Execute ONLY this test's locally generated fixed-grammar programs.
            namespace = {"__builtins__": {}}
            exec(compile(s["code"], "<generated-self-test>", "exec"), namespace)
            self.assertEqual(namespace["result"], s["answer"])
            a = analyze(s["code"], 32)
            extracted = set().union(*(a.node_blocks[n] for n in a.distances))
            gold = set().union(*(span_blocks(x, y, 32) for x, y in s["gold_static_spans"]))
            self.assertEqual(extracted, gold)

    def test_budget_local_and_adapter_invariants(self):
        for i in range(24):
            s = generate_sample(i, ["chain", "call", "branch"][i % 3])
            a = analyze(s["code"], 32)
            for budget in [0, 31, 32, 65, 128, 256]:
                for method in ["lexical_topk", "structure_topk", "evidence_groups"]:
                    chosen = select(a, method, budget)
                    self.assertTrue(a.local <= chosen)
                    self.assertLessEqual(sum(len(a.blocks[b]) for b in chosen - a.local), budget)
                    packed = pack_selected(a, chosen)
                    self.assertEqual(packed["input_ids"][-1], CLS_ID)
                    self.assertEqual(packed["position_ids"][-1], len(a.raw))
                    self.assertEqual(packed["position_ids"], sorted(set(packed["position_ids"])))
                    original = [a.raw[p] for p in packed["position_ids"][:-1]]
                    self.assertEqual(original, packed["input_ids"][:-1])

    def test_no_gold_or_answer_argument(self):
        s = generate_sample(10, "branch")
        a = analyze(s["code"])
        before = select(a, "evidence_groups", 256)
        s["answer"] = 999999
        s["gold_static_spans"] = []
        self.assertEqual(before, select(analyze(s["code"]), "evidence_groups", 256))

    def test_group_overlap_counts_once(self):
        a = analyze("x = 1\na = x + 2\nb = x + 3\nresult = a + b\n", 8)
        groups = build_groups(a)
        self.assertTrue(any(g & h for i, (_, g, _) in enumerate(groups) for _, h, _ in groups[i + 1:]))
        chosen = select(a, "evidence_groups", 24)
        self.assertLessEqual(sum(len(a.blocks[b]) for b in chosen - a.local), 24)

    def test_utf8_offsets(self):
        a = analyze("变量 = 2\nresult = 变量 + 1\n", 8)
        n = a.nodes["变量"]
        self.assertEqual(a.raw[n.start:n.end].decode("utf-8"), "变量 = 2")

    def test_unsupported_cases_fail(self):
        cases = [
            "x = 1\nx = 2\nresult = x\n",
            "import os\nresult = 1\n",
            "x = [1]\nresult = x[0]\n",
            "def f(p):\n    return f(p)\nresult = f(1)\n",
            "def f(p):\n    return p + z\na = f(1)\nz = 2\nresult = a\n",
            "def f(p):\n    q = p + 1\n    return q\nresult = f(1)\n",
            "def f(p=1):\n    return p\nresult = f()\n",
        ]
        for code in cases:
            with self.subTest(code=code), self.assertRaises(ValueError):
                analyze(code)


if __name__ == "__main__":
    unittest.main()
