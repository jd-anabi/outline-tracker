"""Fixtures of the tests of tests/core/ that more than one test file needs.

`schema_docs` is the module core/schema_docs.py, `readme` the text of README.txt that it builds and
`outputs` the text of docs/OUTPUTS.md. They are used by tests/core/test_schema_docs.py,
tests/core/test_schema_docs_readme.py and tests/core/test_schema_docs_outputs.py.
"""

import importlib

import pytest
from layout_helpers import module


@pytest.fixture(scope="module")
def schema_docs():
    """The module under test, imported here so that a missing module fails the tests that need it."""
    return importlib.import_module(module("schema docs"))


@pytest.fixture(scope="module")
def readme(schema_docs) -> str:
    return schema_docs.readme_text()


@pytest.fixture(scope="module")
def outputs(schema_docs) -> str:
    return schema_docs.outputs_markdown()
