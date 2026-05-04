"""Unit tests for structured employee retrieval.

Strategy:
- Do not connect to a real PostgreSQL; verify via ``safe_search_employees``
  with an in-memory duck-typed store that:
  - DB exceptions / a None store both return an empty list ("miss-is-fine" semantics).
  - ``employee_records_to_documents`` output has metadata that matches
    the RAG merge convention.
- ``EmployeeStore.search`` returns empty when query/department/title are all empty.
"""

from __future__ import annotations

from backend.rag.employee_retriever import (
    _build_query_patterns,
    _extract_name_hint,
    EmployeeRecord,
    EmployeeStore,
    employee_records_to_documents,
    safe_search_employees,
)


class _BoomStore:
    def search(self, *_args, **_kwargs):  # pragma: no cover - exercised below
        raise RuntimeError("db is sad")


def test_safe_search_returns_empty_when_store_is_none():
    assert safe_search_employees(None, "alice") == []


def test_safe_search_swallows_store_errors():
    assert safe_search_employees(_BoomStore(), "alice") == []


def test_employee_store_search_short_circuits_on_empty_filters():
    """When query is empty and no department/title filter is given, the DB is not even hit."""
    store = EmployeeStore(connection_string="postgresql://invalid:5432/db")
    assert store.search("", department=None, title=None) == []
    assert store.search(None) == []


def test_records_to_documents_metadata_shape():
    records = [
        EmployeeRecord(
            employee_id="E9001",
            name="Test User",
            department="Engineering",
            title="Senior Engineer",
            email="test.user@example.com",
        )
    ]
    docs = employee_records_to_documents(records, query="Test")
    assert len(docs) == 1
    doc = docs[0]
    # page_content must contain the core fields so the RAG generation
    # step can cite them directly.
    assert "Test User" in doc.page_content
    assert "Engineering" in doc.page_content
    assert "Senior Engineer" in doc.page_content
    # metadata must satisfy the downstream rag chain convention.
    assert doc.metadata["document_role"] == "employee_structured"
    assert doc.metadata["context_id"] == "employee_E9001"
    assert doc.metadata["source"] == "employee_directory"
    assert doc.metadata["employee_id"] == "E9001"
    assert doc.metadata["match_query"] == "Test"


def test_extract_name_hint_from_chinese_self_intro_sentence():
    # Chinese self-introduction kept as Unicode escapes; this fixture
    # exercises the bilingual name-hint extraction.
    query = (
        "\u6211\u53eb Alice Carter\uff0c\u8981\u53bb\u5916\u5730\u51fa\u5dee\uff0c"
        "\u6839\u636e\u6211\u7684\u516c\u53f8\u4fe1\u606f\u751f\u6210\u5e94\u8be5\u6ce8\u610f\u7684 policy"
    )
    assert _extract_name_hint(query) == "Alice Carter"


def test_build_query_patterns_prioritizes_name_tokens():
    query = (
        "\u6211\u53eb Alice Carter\uff0c\u8981\u53bb\u5916\u5730\u51fa\u5dee\uff0c"
        "\u6839\u636e\u6211\u7684\u516c\u53f8\u4fe1\u606f\u751f\u6210\u5e94\u8be5\u6ce8\u610f\u7684 policy"
    )
    patterns = _build_query_patterns(query)
    # The top priority should be the name phrase, avoiding a wholesale
    # ``%...%`` match against the entire sentence.
    assert patterns[0] == "Alice Carter"
    # Key tokens should also be split out to improve fuzzy match recall.
    assert "Alice" in patterns
    assert "Carter" in patterns
