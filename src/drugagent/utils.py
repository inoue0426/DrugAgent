#!/usr/bin/env python
# coding: utf-8
"""Utility helpers for DrugAgent."""

from __future__ import annotations

import asyncio
import random
from itertools import combinations
from typing import List, Optional

from openai import RateLimitError

from drugagent import config as dag_config
from drugagent.config import ALL_EVIDENCE_AGENTS, LABEL_ORDER


def _normalize_label(label: str) -> str:
    """Normalize a label string to a known categorical value.

    Args:
        label: Raw label string.

    Returns:
        Uppercased label or an empty string if unknown.
    """
    value = str(label).strip().upper()
    dash_chars = [
        chr(0x2010),
        chr(0x2011),
        chr(0x2012),
        chr(0x2013),
        chr(0x2014),
        chr(0x2015),
        chr(0x2212),
    ]
    for dash in dash_chars:
        value = value.replace(dash, "-")
    value = value.replace(" ", "_").replace("-", "_")
    while "__" in value:
        value = value.replace("__", "_")
    if value in {"MODERATE_HIGH", "MODERATE_LOW"}:
        value = "MODERATE"
    return value if value in LABEL_ORDER else ""


def _normalize_fusion_label(label: str | None) -> Optional[str]:
    """Normalize labels into fusion space (binary or multi-class).

    Args:
        label: Raw label string.

    Returns:
        Normalized fusion label or None if unknown.
    """
    if not label:
        return None
    value = str(label).strip().lower().replace("_", "-").replace(" ", "-")
    if dag_config.BINARY_MODE:
        mapping = {
            "active": "Active",
            "inactive": "Inactive",
            "strong": "Active",
            "moderate": "Active",
            "moderate-low": "Active",
            "moderate-high": "Active",
            "weak": "Inactive",
            "low": "Inactive",
            "insufficient": "Inactive",
            "none": "Inactive",
        }
        return mapping.get(value)
    mapping = {
        "strong": "Strong",
        "moderate": "Moderate",
        "moderate-low": "Moderate",
        "moderate-high": "Moderate",
        "weak": "Low",
        "low": "Low",
        "insufficient": "Low",
        "none": "Low",
    }
    return mapping.get(value)


def normalize_enabled_agents(enabled_agents: List[str] | None) -> List[str]:
    """Normalize enabled agent list to known evidence agent names.

    Args:
        enabled_agents: Possibly mixed-case agent names.

    Returns:
        Normalized list of agent names.
    """
    if not enabled_agents:
        return ALL_EVIDENCE_AGENTS[:]
    normalized: List[str] = []
    for agent in enabled_agents:
        name = str(agent).strip()
        for candidate in ALL_EVIDENCE_AGENTS:
            if name.lower() == candidate.lower():
                if candidate not in normalized:
                    normalized.append(candidate)
    return normalized


def config_id_from_enabled(enabled_agents: List[str]) -> str:
    """Build a stable config id from enabled agents.

    Args:
        enabled_agents: List of enabled agent names.

    Returns:
        Config id string.
    """
    if not enabled_agents:
        return "none"
    ordered = [a.lower() for a in ALL_EVIDENCE_AGENTS if a in enabled_agents]
    return "_".join(ordered)


def generate_ablation_configs() -> List[List[str]]:
    """Generate all non-empty ablation combinations.

    Returns:
        List of enabled agent lists.
    """
    combos: List[List[str]] = []
    for size in range(1, len(ALL_EVIDENCE_AGENTS) + 1):
        for subset in combinations(ALL_EVIDENCE_AGENTS, size):
            combos.append(list(subset))
    return combos


def _is_rate_limit_exception(exc: Exception) -> bool:
    """Check if an exception is an API rate limit.

    Args:
        exc: Exception raised by a tool or client.

    Returns:
        True if the exception indicates a rate limit.
    """
    if isinstance(exc, RateLimitError):
        return True
    msg = str(exc)
    return (
        ("RateLimit" in msg)
        or ("RateLimitReached" in msg)
        or ("Error code: 429" in msg)
    )


def _retry_after_seconds(exc: Exception, default: float = 3.0) -> float:
    """Extract retry-after delay from exception or return default.

    Args:
        exc: Exception raised by a tool or client.
        default: Default delay to use when retry-after is unavailable.

    Returns:
        Delay in seconds.
    """
    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None) or {}
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if ra:
            try:
                return float(ra)
            except Exception:
                pass
        ra_ms = headers.get("retry-after-ms") or headers.get("Retry-After-Ms")
        if ra_ms:
            try:
                return float(ra_ms) / 1000.0
            except Exception:
                pass
    return default


async def _sleep_with_backoff(exc: Exception, attempt: int) -> None:
    """Sleep with exponential backoff and jitter for retry handling.

    Args:
        exc: Exception raised by a tool or client.
        attempt: Current retry attempt count.
    """
    base_wait = _retry_after_seconds(exc, default=3.0)
    exp_wait = 1.7**attempt
    wait = max(base_wait, exp_wait) + random.uniform(0, 0.75)
    await asyncio.sleep(wait)
