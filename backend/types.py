from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RagDocument:
    """Unified document structure used across the project."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
