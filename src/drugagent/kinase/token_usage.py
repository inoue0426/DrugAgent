from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TokenUsage:
    """Track token usage across LLM calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    breakdown: list = field(default_factory=list)

    def add(
        self, usage: Any, tag: str = "", meta: Optional[Dict[str, Any]] = None
    ) -> None:
        """Accumulate token usage from an SDK response object.

        Args:
            usage: Azure/OpenAI SDK response.usage.
            tag: Optional tag for breakdown tracking.
            meta: Optional metadata for the breakdown entry.
        """
        if usage is None:
            return
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        tt = int(getattr(usage, "total_tokens", 0) or (pt + ct))
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.total_tokens += tt
        self.calls += 1
        if tag:
            self.breakdown.append(
                {
                    "tag": tag,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": tt,
                    "meta": meta or {},
                }
            )
