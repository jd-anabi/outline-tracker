"""Fixtures of the tests of tests/core/ that more than one test file needs.

`schema_docs` is the module core/schema_docs.py; the three files tests/core/test_schema_docs.py,
test_schema_docs_readme.py and test_schema_docs_outputs.py use it. `readme` is the text of README.txt that it
builds and `outputs` the text of docs/OUTPUTS.md; the last two files use them. tests/core/test_schema_texts.py
has a fixture `readme` of its own (the text of `schema.readme_text()`), which is the one its tests get.
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
