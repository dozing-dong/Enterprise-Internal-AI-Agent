from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RagDocument:
    """项目内统一使用的文档结构。"""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
