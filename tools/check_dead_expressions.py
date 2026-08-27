#!/usr/bin/env python
"""
Flag statement-level expressions whose value is silently discarded.

This guards against a specific failure mode: ``autoflake
--remove-unused-variable`` rewrites ``x = f()`` into a bare ``f()``
when ``x`` is unused, which detaches a computed result from the code
that was meant to consume it. That is how the normalization log-det
correction in ``flow_matching_sbi.log_prob`` was lost.

ruff's B015/B018 cover bare comparisons, attribute access and
constants, but not a unary or binary operation wrapping a call
(``-torch.sum(...)``) -- the shape that matters most here, because it
looks like arithmetic that is being used.

Exits non-zero if any are found.
"""

import ast
import sys

# Expression kinds that cannot have a useful effect as a bare statement.
DISCARDED = (
    ast.UnaryOp,
    ast.BinOp,
    ast.BoolOp,
    ast.Compare,
    ast.Subscript,
    ast.Name,
    ast.Attribute,
    ast.IfExp,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.Set,
)


def docstring_nodes(tree):
    """Expr nodes that are legitimate docstrings, by identity."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0]))
    return out


def check(path):
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as err:
        print(f"{path}:{err.lineno}: could not parse: {err.msg}")
        return 1

    skip = docstring_nodes(tree)
    lines = source.splitlines()
    found = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr) or id(node) in skip:
            continue
        value = node.value
        if isinstance(value, ast.Constant) and value.value is Ellipsis:
            continue  # `...` is a deliberate placeholder
        if isinstance(value, DISCARDED) or (
            isinstance(value, ast.Constant) and not isinstance(value.value, str)
        ):
            text = lines[node.lineno - 1].strip()
            print(f"{path}:{node.lineno}: result of this expression is discarded: {text}")
            found += 1

    return found


def main(argv):
    total = sum(check(path) for path in argv[1:] if path.endswith(".py"))
    if total:
        print(
            f"\n{total} discarded expression(s). Assign the result and use it, "
            f"or delete the line."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
