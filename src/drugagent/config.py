#!/usr/bin/env python
# coding: utf-8
"""Shared configuration and prompts for DrugAgent."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from openai import AzureOpenAI

from config_utils import get_reasoning_settings, load_azure_openai_config



def _env_path(env_key: str) -> Optional[Path]:
    """Resolve a path from an environment variable if set.

    Args:
        env_key: Environment variable name.

    Returns:
        Path if the env var is set, otherwise None.
    """
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return None
    return Path(raw)
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

_CONFIG: Optional[Dict[str, Any]] = None
_REASONING_SETTINGS: Optional[Dict[str, Any]] = None


def _get_config() -> Dict[str, Any]:
    """Load and cache Azure OpenAI config.

    Returns:
        Azure OpenAI config dict.
    """
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_azure_openai_config()
    return _CONFIG


def _get_reasoning_settings() -> Optional[Dict[str, Any]]:
    """Load and cache reasoning settings if available.

    Returns:
        Reasoning settings dict or None.
    """
    global _REASONING_SETTINGS
    if _REASONING_SETTINGS is None:
        _REASONING_SETTINGS = get_reasoning_settings()
    return _REASONING_SETTINGS


def _resolve_repo_root() -> Path:
    """Resolve the repository root by walking up to pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[2]


_ROOT = _resolve_repo_root()


def _resolve_existing_path(*candidates: Path) -> Path:
    """Return the first existing path from candidates, or the first candidate."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


KG_PATH = _env_path("DRUGAGENT_KG_PATH") or _ROOT / "data" / "KG+BDB.csv.gz"
RAG_INDEX_PATH = _env_path("DRUGAGENT_RAG_INDEX_PATH") or _resolve_existing_path(
    _ROOT / "data" / "kinase_rag_index.faiss",
    _ROOT / "src" / "drugagent" / "kinase" / "rag_index.faiss",
    _ROOT / "GPCR" / "rag_index.faiss",
    _ROOT / "preprocess" / "drugbank_rag_index.faiss",
    _ROOT / "rag_index.faiss",
)
RAG_META_PATH = _env_path("DRUGAGENT_RAG_META_PATH") or _resolve_existing_path(
    _ROOT / "data" / "kinase_rag_metadata.json",
    _ROOT / "src" / "drugagent" / "kinase" / "rag_metadata.json",
    _ROOT / "GPCR" / "rag_metadata.json",
    _ROOT / "preprocess" / "drugbank_rag_metadata.json",
    _ROOT / "rag_metadata.json",
)
RAG_RESULTS_JSONL = _env_path("DRUGAGENT_RAG_RESULTS_JSONL") or _resolve_existing_path(
    _ROOT / "src" / "drugagent" / "kinase" / "rag_results_v2.jsonl",
    _ROOT / "GPCR" / "rag_results_v2.jsonl",
    _ROOT / "rag_results_v2.jsonl",
)
KG_VERSION = "kg_2026_02_26"
SAVE_VERSION = "v1"

BINARY_MODE = os.getenv("DRUGAGENT_BINARY_MODE", "").strip().lower() in {"1", "true", "yes"}

LABEL_ORDER = {
    "NONE": 0,
    "INACTIVE": 1,
    "WEAK": 1,
    "MODERATE": 2,
    "STRONG": 3,
    "ACTIVE": 3,
    "LOW": 1,
    "INSUFFICIENT": 1,
}

ALL_EVIDENCE_AGENTS = ["ML", "KG", "RAG"]


_tool_client: AzureOpenAI | None = None


def _get_tool_client() -> AzureOpenAI:
    """Create or reuse the AzureOpenAI client for tool calls.

    Returns:
        AzureOpenAI client.
    """
    global _tool_client
    if _tool_client is None:
        cfg = _get_config()
        _tool_client = AzureOpenAI(
            api_key=cfg["api_key"],
            azure_endpoint=cfg["endpoint"],
            api_version=cfg["api_version"],
        )
    return _tool_client


# Configure logging
logging.getLogger("autogen_core").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Termination conditions
text_mention_termination = TextMentionTermination("TERMINATE", sources=["SummaryAgent"])
max_messages_termination = MaxMessageTermination(max_messages=10)
termination = text_mention_termination | max_messages_termination

# Selector prompt
selector_prompt_with_planner = """Please select the most appropriate agent to perform the next task.

{roles}

Current conversation context:
{history}

Carefully analyze the conversation history above. You must ensure that the Planning Agent has initiated the task and assigned subtasks before any other specialized agent begins working.

Do NOT select SummaryAgent until you have received tool results from all enabled evidence agents (MLAgent, KGAgent, RAGAgent). If any evidence is missing, select a missing evidence agent first.

From the available participants in {participants}, select only one agent to proceed with the next step.
"""

selector_prompt_no_planner = """Please select the most appropriate agent to perform the next task.

{roles}

Current conversation context:
{history}

Do NOT select SummaryAgent until you have received tool results from all enabled evidence agents (MLAgent, KGAgent, RAGAgent). If any evidence is missing, select a missing evidence agent first.

From the available participants in {participants}, select only one agent to proceed with the next step.
"""
