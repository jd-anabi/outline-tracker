"""The scan of the imports in a parsed Python file, shared by the rule tests of tests/repo/.

tests/repo/test_repo_rules.py (the boundaries of Qt and the model libraries), tests/repo/test_repo_rules_test_code.py
(the imports of conftest.py and of test modules) and tests/repo/test_layers.py (the layers of the package) all
read imports through `imports`. A test module imports nothing from another test module, so the scan lies here.
"""

import ast


def imports(tree: ast.AST):
    """Yield (line, module, names) for every absolute import in a parsed file: the dotted name of
    the module, and for `from module import a, b` the names ("a", "b"), else ().

    Covers `import a.b`, `from a.b import c`, and `import_module("a.b")` / `__import__("a.b")`
    with a literal name, at any depth (lazy imports inside functions count too).
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, ()
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.lineno, node.module, tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.Call) and node.args:
            func, first = node.func, node.args[0]
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            literal = isinstance(first, ast.Constant) and isinstance(first.value, str)
            if called in ("import_module", "__import__") and literal:
                yield node.lineno, first.value, ()
