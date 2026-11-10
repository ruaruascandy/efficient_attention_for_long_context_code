#!/usr/bin/env python3
"""Week-2 selection prototype (Python standard library only, Python >= 3.10).

Measures CONSERVATIVE STATIC EVIDENCE retention, not model answer accuracy.
Uses an explicit byte tokenizer (one UTF-8 byte = one token), not a Qwen tokenizer.
Supported language is deliberately narrow; unsupported programs fail explicitly.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import re
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Node:
    name: str
    start: int
    end: int
    kind: str
    deps: dict[str, str] = field(default_factory=dict)


@dataclass
class Analysis:
    raw: bytes
    nodes: dict[str, Node]
    blocks: list[bytes]
    node_blocks: dict[str, set[int]]
    local: set[int]
    target: str
    distances: dict[str, int]


def loaded_names(expr: ast.AST) -> set[str]:
    return {x.id for x in ast.walk(expr)
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}


def validate_expr(expr: ast.AST, allow_call: bool) -> None:
    allowed = (ast.Name, ast.Load, ast.Constant, ast.BinOp, ast.Add, ast.Sub,
               ast.Mult, ast.Compare, ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq,
               ast.NotEq, ast.UnaryOp, ast.USub, ast.UAdd)
    for x in ast.walk(expr):
        if isinstance(x, ast.Call):
            if not allow_call or not isinstance(x.func, ast.Name) or x.keywords:
                raise ValueError("Only positional calls to named, non-recursive functions are supported")
        elif not isinstance(x, allowed):
            raise ValueError(f"Unsupported expression: {type(x).__name__}")
        if isinstance(x, ast.Constant) and type(x.value) not in (int, bool):
            raise ValueError("Only integer and boolean literals are supported")


def function_uses(fn: ast.FunctionDef) -> tuple[set[str], set[str]]:
    """Keep an entire small function as one semantic node; no fake full PDG."""
    if (fn.decorator_list or fn.returns or fn.args.defaults or fn.args.kw_defaults
            or fn.args.kwonlyargs or fn.args.posonlyargs or fn.args.vararg
            or fn.args.kwarg or any(a.annotation for a in fn.args.args)):
        raise ValueError("Function must have plain positional parameters")
    value_names, control_names = set(), set()

    def ret(x: ast.AST) -> None:
        if not isinstance(x, ast.Return) or x.value is None:
            raise ValueError("Function bodies support return, or terminal if/else with one return each")
        validate_expr(x.value, allow_call=False)
        value_names.update(loaded_names(x.value))

    if len(fn.body) != 1:
        raise ValueError("Function body must be a single return or a single if/else")
    body = fn.body[0]
    if isinstance(body, ast.If):
        validate_expr(body.test, allow_call=False)
        control_names.update(loaded_names(body.test))
        if len(body.body) != 1 or len(body.orelse) != 1:
            raise ValueError("Both branches must contain one return")
        ret(body.body[0])
        ret(body.orelse[0])
    else:
        ret(body)
    parameters = {x.arg for x in fn.args.args}
    return value_names - parameters, control_names - parameters


def span_blocks(start: int, end: int, block_size: int) -> set[int]:
    return set(range(start // block_size, (end - 1) // block_size + 1))


def analyze(code: str, block_size: int = 64, target: str = "result") -> Analysis:
    """Read source only. No answer, gold nodes, or runtime execution enters here.

    All module bindings must be unique. Function globals must exist before each
    call. These restrictions make context-independent global edges well-defined.
    """
    if block_size < 1:
        raise ValueError("block_size must be positive")
    raw = code.encode("utf-8")
    lines = raw.splitlines(keepends=True)
    offsets, total = [], 0
    for line in lines:
        offsets.append(total)
        total += len(line)
    tree = ast.parse(code)
    nodes, definitions, functions = {}, {}, {}
    statements = tree.body
    if not statements:
        raise ValueError("Empty program")
    for i, stmt in enumerate(statements):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name, kind = stmt.targets[0].id, "assign"
            validate_expr(stmt.value, allow_call=True)
        elif isinstance(stmt, ast.FunctionDef):
            name, kind = stmt.name, "function"
            function_uses(stmt)
            functions[name] = stmt
        else:
            raise ValueError(f"Unsupported top-level statement: {type(stmt).__name__}")
        if name in definitions:
            raise ValueError(f"Rebinding is outside this prototype's subset: {name}")
        start = offsets[stmt.lineno - 1] + stmt.col_offset
        end = offsets[stmt.end_lineno - 1] + stmt.end_col_offset
        nodes[name] = Node(name, start, end, kind)
        definitions[name] = i
    if target not in nodes or definitions[target] != len(statements) - 1 or nodes[target].kind != "assign":
        raise ValueError("Target must be the last top-level assignment")

    for i, stmt in enumerate(statements):
        name = stmt.name if isinstance(stmt, ast.FunctionDef) else stmt.targets[0].id
        if isinstance(stmt, ast.FunctionDef):
            value_names, control_names = function_uses(stmt)
            for used in value_names | control_names:
                if used not in nodes or nodes[used].kind != "assign":
                    raise ValueError(f"Undefined/non-value function global: {used}")
                role = "control+return" if used in value_names & control_names else (
                    "control" if used in control_names else "return_value")
                nodes[name].deps[used] = role
        else:
            calls = [x for x in ast.walk(stmt.value) if isinstance(x, ast.Call)]
            called = {x.func.id for x in calls}
            arguments = set().union(*(loaded_names(a) for call in calls for a in call.args)) if calls else set()
            for used in loaded_names(stmt.value):
                if used not in nodes or definitions[used] >= i:
                    raise ValueError(f"Undefined/forward reference at {name}: {used}")
                if nodes[used].kind == "function" and used not in called:
                    raise ValueError("Function values/aliases are unsupported")
                nodes[name].deps[used] = "call_return" if used in called else (
                    "argument" if used in arguments else "data")
            for call in calls:
                fn = functions.get(call.func.id)
                if fn is None or len(call.args) != len(fn.args.args):
                    raise ValueError("Call must match a defined function's positional arity")
                values, controls = function_uses(fn)
                if any(definitions[g] >= i for g in values | controls):
                    raise ValueError("Function global must be assigned before the call")

    # Reject cycles, even in dead code, rather than silently using a broken graph.
    visiting, visited = set(), set()
    def visit(name: str) -> None:
        if name in visiting:
            raise ValueError("Cyclic dependency is outside the supported subset")
        if name in visited:
            return
        visiting.add(name)
        for dep in nodes[name].deps:
            visit(dep)
        visiting.remove(name)
        visited.add(name)
    for name in nodes:
        visit(name)

    distances, queue = {target: 0}, deque([target])
    while queue:
        name = queue.popleft()
        for dep in nodes[name].deps:
            if dep not in distances:
                distances[dep] = distances[name] + 1
                queue.append(dep)
    blocks = [raw[i:i + block_size] for i in range(0, len(raw), block_size)]
    node_blocks = {name: span_blocks(n.start, n.end, block_size) for name, n in nodes.items()}
    # Same query-containing blocks are free/fixed for all budgeted selectors.
    local = node_blocks[target] | {len(blocks) - 1}
    return Analysis(raw, nodes, blocks, node_blocks, local, target, distances)


def build_groups(a: Analysis, hops: int = 2) -> list[tuple[str, frozenset[int], float]]:
    """All source nodes are anchors; gold annotations never enter this function."""
    if hops < 0:
        raise ValueError("hops must be non-negative")
    groups = []
    for anchor in a.nodes:
        seen, frontier = {anchor}, {anchor}
        for _ in range(hops):
            frontier = {d for n in frontier for d in a.nodes[n].deps} - seen
            seen |= frontier
        blocks = set().union(*(a.node_blocks[n] for n in seen)) - a.local
        weight = 1.0 / (1 + a.distances[anchor]) if anchor in a.distances else 0.0
        if blocks:
            groups.append((anchor, frozenset(blocks), weight))
    return groups


def rank_select(a: Analysis, scores: dict[int, float], budget: int, chosen=None) -> set[int]:
    chosen = set(a.local if chosen is None else chosen)
    used = sum(len(a.blocks[b]) for b in chosen - a.local)
    for b in sorted(range(len(a.blocks)), key=lambda b: (-scores.get(b, 0.0), b)):
        if b not in chosen and used + len(a.blocks[b]) <= budget:
            chosen.add(b)
            used += len(a.blocks[b])
    return chosen


def select(a: Analysis, method: str, budget: int, hops: int = 2) -> set[int]:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if method == "full":
        return set(range(len(a.blocks)))
    if method == "local_only":
        return set(a.local)
    if method == "lexical_topk":
        target = a.nodes[a.target]
        query_names = set(re.findall(rb"[A-Za-z_]\w*", a.raw[target.start:target.end]))
        scores = {}
        for b, raw in enumerate(a.blocks):
            names = set(re.findall(rb"[A-Za-z_]\w*", raw))
            scores[b] = len(names & query_names) / math.sqrt(max(1, len(names)))
        return rank_select(a, scores, budget)

    groups = build_groups(a, hops)
    # Structural Top-k receives the SAME groups and weights, flattened to blocks.
    scores = {b: 0.0 for b in range(len(a.blocks))}
    for _, blocks, weight in groups:
        for b in blocks:
            scores[b] += weight / len(blocks)
    if method == "structure_topk":
        return rank_select(a, scores, budget)
    if method != "evidence_groups":
        raise ValueError(f"Unknown method: {method}")

    chosen, used = set(a.local), 0
    while True:
        best, best_key = None, (-1.0, -1.0)
        for _, blocks, weight in groups:
            new = set(blocks) - chosen
            cost = sum(len(a.blocks[b]) for b in new)
            if not new or cost + used > budget or weight == 0:
                continue
            union = chosen | new
            # Reward newly completed groups, counted once. Shared blocks are free.
            gain = sum(w for _, g, w in groups if g <= union and not g <= chosen)
            key = (gain / cost, gain)
            if gain > 0 and key > best_key:
                best, best_key = new, key
        if best is None:
            break
        used += sum(len(a.blocks[b]) for b in best)
        chosen |= best
    # Identical fill rule avoids wasting budget after no complete group fits.
    return rank_select(a, scores, budget, chosen)


def generate_sample(seed: int, family: str, distractors: int = 24) -> dict:
    """Annotations are recorded by construction, independently of analyze()."""
    rng, used_names = random.Random(seed), {"result"}
    def fresh() -> str:
        while True:
            name = "v" + "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=5))
            if name not in used_names:
                used_names.add(name)
                return name

    source, derived, fn, threshold, offset = (fresh() for _ in range(5))
    x, plus, cut = rng.randint(1, 15), rng.randint(1, 7), rng.randint(4, 12)
    # Each unit has independent generator dependencies used only to order source.
    units = {}
    def add(name, text, deps=(), gold=False):
        units[name] = {"text": text, "deps": set(deps), "gold": gold}
    add(source, f"{source} = {x}\n", gold=True)
    if family == "chain":
        middle = fresh()
        add(middle, f"{middle} = {source} + {plus}\n", [source], True)
        add(derived, f"{derived} = {middle} * 2\n", [middle], True)
        answer = (x + plus) * 2 + 1
    elif family in ("call", "branch"):
        add(offset, f"{offset} = {plus}\n", gold=True)
        if family == "call":
            text = f"def {fn}(p):\n    return p + {offset}\n"
            prerequisites = [offset]
            answer = x + plus + 1
        else:
            add(threshold, f"{threshold} = {cut}\n", gold=True)
            text = (f"def {fn}(p):\n    if p > {threshold}:\n"
                    f"        return p + {offset}\n    else:\n        return p - {offset}\n")
            prerequisites = [threshold, offset]
            answer = (x + plus if x > cut else x - plus) + 1
        add(fn, text, prerequisites, True)
        add(derived, f"{derived} = {fn}({source})\n", [fn, source], True)
    else:
        raise ValueError(f"Unknown family: {family}")

    noise_names = []
    for _ in range(distractors):
        name = fresh()
        if noise_names and rng.random() < 0.55:
            parent = rng.choice(noise_names)
            add(name, f"{name} = {parent} + {rng.randint(1, 15)}\n", [parent])
        else:
            add(name, f"{name} = {rng.randint(1, 30)}\n")
        noise_names.append(name)

    emitted, chunks, gold_spans = set(), [], []
    position = 0
    while len(emitted) < len(units):
        ready = [n for n, u in units.items() if n not in emitted and u["deps"] <= emitted]
        name = rng.choice(ready)
        unit = units[name]
        chunks.append(unit["text"])
        if unit["gold"]:
            gold_spans.append([position, position + len(unit["text"].rstrip("\n").encode())])
        position += len(unit["text"].encode())
        emitted.add(name)
    final = f"result = {derived} + 1\n"
    chunks.append(final)
    gold_spans.append([position, position + len(final.rstrip("\n").encode())])
    return {"id": f"{family}_{seed}", "family": family, "seed": seed,
            "split": "smoke_only", "code": "".join(chunks), "target": "result",
            "answer": answer, "gold_static_spans": gold_spans}


def evaluate(samples: list[dict], block_size: int, budgets: list[int], hops: int):
    rows, first = [], None
    methods = ["lexical_topk", "structure_topk", "evidence_groups"]
    for sample in samples:
        start = time.perf_counter()
        a = analyze(sample["code"], block_size, sample["target"])
        parse_ms = 1000 * (time.perf_counter() - start)
        # Gold metadata is only accessed AFTER analysis; select() receives none.
        gold = set().union(*(span_blocks(s, e, block_size) for s, e in sample["gold_static_spans"]))
        remote_gold = gold - a.local
        required = sum(len(a.blocks[b]) for b in remote_gold)
        for budget in budgets:
            for method in methods:
                start = time.perf_counter()
                chosen = select(a, method, budget, hops)
                selection_ms = 1000 * (time.perf_counter() - start)
                remote_cost = sum(len(a.blocks[b]) for b in chosen - a.local)
                assert remote_cost <= budget and a.local <= chosen
                row = {"sample_id": sample["id"], "family": sample["family"],
                       "method": method, "budget_byte_tokens": budget,
                       "remote_byte_tokens": remote_cost,
                       "local_byte_tokens": sum(len(a.blocks[b]) for b in a.local),
                       "complete_static_evidence": int(gold <= chosen),
                       "remote_evidence_recall": len(remote_gold & chosen) / len(remote_gold) if remote_gold else 1.0,
                       "gold_fits_budget": int(required <= budget),
                       "analysis_ms": parse_ms, "selection_ms": selection_ms,
                       "selected_blocks": sorted(chosen)}
                rows.append(row)
        if first is None:
            first = {"sample": sample, "nodes": {n: {"kind": v.kind, "deps": v.deps,
                      "blocks": sorted(a.node_blocks[n])} for n, v in a.nodes.items()},
                     "local_blocks": sorted(a.local), "gold_static_blocks": sorted(gold),
                     "groups": [{"anchor": n, "blocks": sorted(g), "weight": w}
                                for n, g, w in build_groups(a, hops)],
                     "selections": [r for r in rows if r["sample_id"] == sample["id"]]}
    summary = []
    for budget in budgets:
        for method in methods:
            for family in ["all", "chain", "call", "branch"]:
                group = [r for r in rows if r["method"] == method
                         and r["budget_byte_tokens"] == budget
                         and (family == "all" or r["family"] == family)]
                if not group:
                    continue
                feasible = [r for r in group if r["gold_fits_budget"]]
                summary.append({"method": method, "family": family,
                    "budget_byte_tokens": budget, "n": len(group),
                    "static_complete_rate": statistics.mean(r["complete_static_evidence"] for r in group),
                    "mean_remote_recall": statistics.mean(r["remote_evidence_recall"] for r in group),
                    "feasible_fraction": len(feasible) / len(group),
                    "complete_rate_when_feasible": statistics.mean(r["complete_static_evidence"] for r in feasible) if feasible else "NA",
                    "mean_remote_tokens": statistics.mean(r["remote_byte_tokens"] for r in group),
                    "median_analysis_ms": statistics.median(r["analysis_ms"] for r in group),
                    "median_selection_ms": statistics.median(r["selection_ms"] for r in group)})
    return rows, summary, first


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 384])
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("out"))
    args = parser.parse_args()
    if args.samples < 1 or args.block_size < 1 or args.hops < 0 or any(b < 0 for b in args.budgets):
        parser.error("Invalid non-positive size or negative budget/hops")
    samples = [generate_sample(args.seed + i, ["chain", "call", "branch"][i % 3], 16 + i % 17)
               for i in range(args.samples)]
    rows, summary, first = evaluate(samples, args.block_size, args.budgets, args.hops)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "smoke_samples.jsonl").write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8")
    (args.output / "example_trace.json").write_text(json.dumps(first, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {"seed": args.seed, "samples": args.samples, "block_size": args.block_size,
              "budgets": args.budgets, "hops": args.hops, "tokenizer": "UTF-8 bytes",
              "measurement": "static_evidence_retention_only", "model_trained": False,
              "warning": "Smoke templates only; not a benchmark or novelty/acceleration claim"}
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_csv(args.output / "per_sample.csv", rows)
    write_csv(args.output / "summary.csv", summary)
    print("Static evidence retention only; no neural-model accuracy measured.")
    print("budget  method            complete  recall  feasible")
    for r in summary:
        if r["family"] == "all":
            print(f"{r['budget_byte_tokens']:6d}  {r['method']:17s} "
                  f"{r['static_complete_rate']:8.3f} {r['mean_remote_recall']:7.3f} {r['feasible_fraction']:9.3f}")
    print(f"Saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
