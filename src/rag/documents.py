from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Document:
    """
    Lightweight wrapper representing one chunk of API knowledge.
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
