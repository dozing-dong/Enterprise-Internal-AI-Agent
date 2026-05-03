"""员工结构化信息检索（PostgreSQL）。

设计要点：
- 与 ``backend.rag.retrievers`` / ``backend.storage.history`` 共用同一个
  PG 连接配置，避免引入额外服务依赖。
- 启动期通过 ``CREATE TABLE IF NOT EXISTS`` 自动建表，对部署友好。
- 查询行为为模糊匹配（``ILIKE``）：把同一个 ``query`` 同时尝试匹配
  ``name`` / ``department`` / ``title`` / ``employee_id`` / ``email``，
  并允许调用方再叠加 ``department`` / ``title`` 作为精确过滤。
- 失败容忍：DB 不可用、表为空、SQL 异常都返回空列表，不向上抛，
  以满足“RAG 必查但查不到可忽略”的语义。
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

from backend.config import (
    EMPLOYEE_LOOKUP_TOP_K,
    EMPLOYEE_TABLE,
    PGVECTOR_CONNECTION,
)
from backend.storage.history import _normalize_pg_connection_string
from backend.types import RagDocument


logger = logging.getLogger(__name__)

_LOOKUP_STOPWORDS = {
    "i",
    "im",
    "i'm",
    "my",
    "name",
    "is",
    "am",
    "to",
    "the",
    "a",
    "an",
    "and",
    "or",
    "please",
    "policy",
    "policies",
    "trip",
    "travel",
    "business",
}


def _extract_name_hint(query: str) -> str | None:
    """从自然语言里提取可能的人名（英文字母为主）。"""
    patterns = [
        r"(?:我叫|我是)\s*([A-Za-z][A-Za-z'.-]*(?:\s+[A-Za-z][A-Za-z'.-]*){0,2})",
        r"(?:my\s+name\s+is|i\s+am)\s*([A-Za-z][A-Za-z'.-]*(?:\s+[A-Za-z][A-Za-z'.-]*){0,2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        value = re.sub(r"\s+", " ", match.group(1)).strip(" ,.!?\"'“”")
        if value:
            return value
    return None


def _build_query_patterns(query: str) -> list[str]:
    """把整句查询拆成更可命中的 pattern 列表。"""
    clean_query = query.strip()
    if not clean_query:
        return []

    patterns: list[str] = []
    seen: set[str] = set()

    def _push(value: str) -> None:
        candidate = value.strip()
        if not candidate:
            return
        lowered = candidate.lower()
        if lowered in seen:
            return
        seen.add(lowered)
        patterns.append(candidate)

    # 1) 优先尝试显式姓名提示，如“我叫 Alice Carter”
    name_hint = _extract_name_hint(clean_query)
    if name_hint:
        _push(name_hint)

    # 2) 再尝试完整 query
    _push(clean_query)

    # 3) 最后拆分英文 token，降低整句匹配失败概率
    for token in re.findall(r"[A-Za-z][A-Za-z'.-]{1,}", clean_query):
        lowered = token.lower()
        if lowered in _LOOKUP_STOPWORDS:
            continue
        if len(token) < 3:
            continue
        _push(token)

    return patterns


@dataclass(slots=True)
class EmployeeRecord:
    """单条员工记录。字段顺序固定，便于上游序列化。"""

    employee_id: str
    name: str
    department: str
    title: str
    email: str

    def to_dict(self) -> dict[str, str]:
        return {
            "employee_id": self.employee_id,
            "name": self.name,
            "department": self.department,
            "title": self.title,
            "email": self.email,
        }

    def to_text(self) -> str:
        """转成自然语言段落，便于塞进 RAG 上下文。"""
        return (
            f"Employee {self.employee_id}: {self.name}, "
            f"department={self.department}, title={self.title}, "
            f"email={self.email}."
        )


class EmployeeStore:
    """基于 PG 的员工目录读写客户端。

    业务上只读（生产数据维护通过 SQL 或运营脚本完成），
    本类只暴露 ``ensure_table`` / ``upsert_many`` / ``search``，
    便于初始化时灌入演示数据。
    """

    def __init__(
        self,
        *,
        connection_string: str = PGVECTOR_CONNECTION,
        table_name: str = EMPLOYEE_TABLE,
    ) -> None:
        self._connection_string = connection_string
        self._table_name = table_name
        self._table_initialized = False
        self._init_lock = threading.Lock()

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise ImportError(
                "缺少 psycopg 依赖，无法使用 EmployeeStore。"
            ) from exc
        return psycopg.connect(_normalize_pg_connection_string(self._connection_string))

    def _ensure_table(self, connection) -> None:
        if self._table_initialized:
            return
        with self._init_lock:
            if self._table_initialized:
                return
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table_name} (
                        employee_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        department TEXT NOT NULL,
                        title TEXT NOT NULL,
                        email TEXT NOT NULL DEFAULT ''
                    );
                    """
                )
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_name
                    ON {self._table_name} (name);
                    """
                )
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_department
                    ON {self._table_name} (department);
                    """
                )
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table_name}_title
                    ON {self._table_name} (title);
                    """
                )
            connection.commit()
            self._table_initialized = True

    def ensure_table(self) -> None:
        """显式建表（不要求一定成功，调用方可吞异常）。"""
        with self._connect() as connection:
            self._ensure_table(connection)

    def upsert_many(self, records: list[EmployeeRecord]) -> int:
        """批量写入员工数据，存在则覆盖（按 employee_id 主键）。"""
        if not records:
            return 0

        payload = [
            (
                rec.employee_id,
                rec.name,
                rec.department,
                rec.title,
                rec.email,
            )
            for rec in records
        ]

        with self._connect() as connection:
            self._ensure_table(connection)
            with connection.cursor() as cursor:
                cursor.executemany(
                    f"""
                    INSERT INTO {self._table_name}
                        (employee_id, name, department, title, email)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (employee_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        department = EXCLUDED.department,
                        title = EXCLUDED.title,
                        email = EXCLUDED.email;
                    """,
                    payload,
                )
            connection.commit()
        return len(payload)

    def count(self) -> int:
        try:
            with self._connect() as connection:
                self._ensure_table(connection)
                with connection.cursor() as cursor:
                    cursor.execute(f"SELECT COUNT(*) FROM {self._table_name};")
                    row = cursor.fetchone()
                    return int(row[0]) if row else 0
        except Exception:  # pragma: no cover - defensive
            logger.exception("EmployeeStore.count failed")
            return 0

    def search(
        self,
        query: str | None,
        *,
        department: str | None = None,
        title: str | None = None,
        limit: int = EMPLOYEE_LOOKUP_TOP_K,
    ) -> list[EmployeeRecord]:
        """模糊匹配员工。

        - ``query``：在 ``name`` / ``department`` / ``title`` /
          ``employee_id`` / ``email`` 上做 ``ILIKE``，任一命中即返回。
        - ``department`` / ``title``：可选的精确过滤（``ILIKE``）。
        - DB 不可用或异常时返回空列表，调用方应视为“没有员工证据”。
        """
        clean_query = (query or "").strip()
        clean_department = (department or "").strip()
        clean_title = (title or "").strip()
        safe_limit = max(1, min(int(limit or EMPLOYEE_LOOKUP_TOP_K), 50))

        if not clean_query and not clean_department and not clean_title:
            return []

        clauses: list[str] = []
        params: list[Any] = []

        if clean_query:
            query_patterns = _build_query_patterns(clean_query)
            if query_patterns:
                pattern_clauses: list[str] = []
                for pattern in query_patterns:
                    like_value = f"%{pattern}%"
                    pattern_clauses.append(
                        "(name ILIKE %s OR department ILIKE %s OR title ILIKE %s "
                        "OR employee_id ILIKE %s OR email ILIKE %s)"
                    )
                    params.extend([like_value] * 5)
                clauses.append("(" + " OR ".join(pattern_clauses) + ")")

        if clean_department:
            clauses.append("department ILIKE %s")
            params.append(f"%{clean_department}%")

        if clean_title:
            clauses.append("title ILIKE %s")
            params.append(f"%{clean_title}%")

        where_sql = " AND ".join(clauses) if clauses else "TRUE"

        try:
            with self._connect() as connection:
                self._ensure_table(connection)
                with connection.cursor() as cursor:
                    cursor.execute(
                        f"""
                        SELECT employee_id, name, department, title, email
                        FROM {self._table_name}
                        WHERE {where_sql}
                        ORDER BY name ASC, employee_id ASC
                        LIMIT %s;
                        """,
                        (*params, safe_limit),
                    )
                    rows = cursor.fetchall()
        except Exception:
            logger.exception("EmployeeStore.search failed; treating as miss")
            return []

        return [
            EmployeeRecord(
                employee_id=str(row[0] or ""),
                name=str(row[1] or ""),
                department=str(row[2] or ""),
                title=str(row[3] or ""),
                email=str(row[4] or ""),
            )
            for row in rows
        ]


def employee_records_to_documents(
    records: list[EmployeeRecord],
    *,
    query: str | None = None,
) -> list[RagDocument]:
    """把员工记录包装成 RagDocument，便于合入 RAG ``retrieved_docs``。

    每条员工对应一个独立 doc，``document_role=employee_structured``
    便于下游识别这是结构化数据来源（而非自由文本知识库片段）。
    """
    docs: list[RagDocument] = []
    for record in records:
        metadata: dict[str, Any] = {
            "source": "employee_directory",
            "context_id": f"employee_{record.employee_id}",
            "document_role": "employee_structured",
            "title": f"Employee profile - {record.name}",
            "employee_id": record.employee_id,
            "department": record.department,
            "job_title": record.title,
            "email": record.email,
        }
        if query:
            metadata["match_query"] = query
        docs.append(
            RagDocument(page_content=record.to_text(), metadata=metadata)
        )
    return docs


def safe_search_employees(
    store: EmployeeStore | None,
    query: str | None,
    *,
    department: str | None = None,
    title: str | None = None,
    limit: int = EMPLOYEE_LOOKUP_TOP_K,
) -> list[EmployeeRecord]:
    """辅助函数：``store`` 为空或抛错都视为查不到。

    供 RAG chain 的“必查”节点使用，避免把基础设施故障传染到主流程。
    """
    if store is None:
        return []
    try:
        return store.search(
            query, department=department, title=title, limit=limit
        )
    except Exception:
        logger.exception("safe_search_employees failed; returning empty result")
        return []


DEFAULT_DEMO_EMPLOYEES: list[EmployeeRecord] = [
    EmployeeRecord(
        employee_id="E1001",
        name="Alice Carter",
        department="Engineering",
        title="Senior Backend Engineer",
        email="alice.carter@example.com",
    ),
    EmployeeRecord(
        employee_id="E1002",
        name="Bob Anderson",
        department="Engineering",
        title="Engineering Manager",
        email="bob.anderson@example.com",
    ),
    EmployeeRecord(
        employee_id="E1003",
        name="Cathy Brown",
        department="Human Resources",
        title="HR Business Partner",
        email="cathy.brown@example.com",
    ),
    EmployeeRecord(
        employee_id="E1004",
        name="David Miller",
        department="Finance",
        title="Finance Lead",
        email="david.miller@example.com",
    ),
    EmployeeRecord(
        employee_id="E1005",
        name="Eva Davis",
        department="Sales",
        title="Account Executive",
        email="eva.davis@example.com",
    ),
    EmployeeRecord(
        employee_id="E1006",
        name="Frank Wilson",
        department="Legal",
        title="Senior Counsel",
        email="frank.wilson@example.com",
    ),
    EmployeeRecord(
        employee_id="E1007",
        name="Grace Taylor",
        department="Information Technology",
        title="IT Support Specialist",
        email="grace.taylor@example.com",
    ),
    EmployeeRecord(
        employee_id="E1008",
        name="Henry Thompson",
        department="Engineering",
        title="Frontend Engineer",
        email="henry.thompson@example.com",
    ),
]


def seed_default_employees(store: EmployeeStore | None = None) -> int:
    """把内置 demo 员工写入表。失败时返回 0，不向上抛。"""
    target = store or EmployeeStore()
    try:
        return target.upsert_many(DEFAULT_DEMO_EMPLOYEES)
    except Exception:
        logger.exception("seed_default_employees failed")
        return 0
