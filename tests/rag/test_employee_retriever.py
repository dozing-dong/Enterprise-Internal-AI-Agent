"""员工结构化检索的单元测试。

策略：
- 不连接真实 PostgreSQL，通过 ``safe_search_employees`` 与一个内存版
  duck-typed store 验证：
  - DB 异常 / store 为空都返回空列表（“查不到可忽略”语义）。
  - ``employee_records_to_documents`` 输出的 metadata 满足 RAG 合并约定。
- ``EmployeeStore.search`` 对空 query/department/title 直接返回空。
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
    """空 query 且无 department/title 过滤时，连 DB 都不会去访问。"""
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
    # page_content 必须包含核心字段，便于 RAG generate 时直接引用。
    assert "Test User" in doc.page_content
    assert "Engineering" in doc.page_content
    assert "Senior Engineer" in doc.page_content
    # metadata 必须满足下游 rag chain 的约定：
    assert doc.metadata["document_role"] == "employee_structured"
    assert doc.metadata["context_id"] == "employee_E9001"
    assert doc.metadata["source"] == "employee_directory"
    assert doc.metadata["employee_id"] == "E9001"
    assert doc.metadata["match_query"] == "Test"


def test_extract_name_hint_from_chinese_self_intro_sentence():
    query = "我叫 Alice Carter，要去外地出差，根据我的公司信息生成应该注意的 policy"
    assert _extract_name_hint(query) == "Alice Carter"


def test_build_query_patterns_prioritizes_name_tokens():
    query = "我叫 Alice Carter，要去外地出差，根据我的公司信息生成应该注意的 policy"
    patterns = _build_query_patterns(query)
    # 第一优先应是姓名短语，避免只拿整句 `%...%` 去匹配。
    assert patterns[0] == "Alice Carter"
    # 关键 token 也应被拆出来，提升模糊匹配命中率。
    assert "Alice" in patterns
    assert "Carter" in patterns
