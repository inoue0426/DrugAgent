#!/usr/bin/env python
# coding: utf-8
"""Kinase DrugAgent with Autogen multi-agent orchestration."""

from __future__ import annotations

import sys
from pathlib import Path

def _resolve_repo_root() -> Path:
    """Resolve the repository root by walking up to pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[3]


_REPO_ROOT = _resolve_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config_utils import load_azure_openai_config
from drugagent.agents import build_agents, create_model_client
from drugagent.cli import get_active_agents, main
from drugagent.config import (
    ALL_EVIDENCE_AGENTS,
    KG_PATH,
    KG_VERSION,
    LABEL_ORDER,
    RAG_INDEX_PATH,
    RAG_META_PATH,
    RAG_RESULTS_JSONL,
    SAVE_VERSION,
    _get_config,
    _get_reasoning_settings,
    _get_tool_client,
    selector_prompt_no_planner,
    selector_prompt_with_planner,
    termination,
)
from drugagent.csv import (
    check_already_processed,
    ensure_csv_schema,
    sanitize_model_name,
    save_summary_to_csv,
)
from drugagent.evidence import _build_evidence_payload, _gather_evidence_parallel
from drugagent.orchestration import chat_with_agents_and_summarize
from drugagent.resources import (
    _get_kg_df,
    _get_rag_resources,
    _load_rag_result_from_jsonl,
)
from drugagent.summary import (
    TokenAgg,
    _extract_json_block,
    _get_source_label_with_status,
    _get_summary_client,
    _make_azure_client_with_reasoning,
    _run_llm_final_decision,
    _run_summary_with_evidence,
    _split_summary_output,
    _extract_label_from_text,
    apply_final_decision,
    attach_evidence_metadata,
    get_summary_system_message,
)
from drugagent.tools import kg_score, ml_score, rag_score
from drugagent.utils import (
    _is_rate_limit_exception,
    _normalize_fusion_label,
    _normalize_label,
    _retry_after_seconds,
    _sleep_with_backoff,
    config_id_from_enabled,
    generate_ablation_configs,
    normalize_enabled_agents,
)

__all__ = [
    "ALL_EVIDENCE_AGENTS",
    "KG_PATH",
    "KG_VERSION",
    "LABEL_ORDER",
    "RAG_INDEX_PATH",
    "RAG_META_PATH",
    "RAG_RESULTS_JSONL",
    "SAVE_VERSION",
    "TokenAgg",
    "_build_evidence_payload",
    "_extract_json_block",
    "_extract_label_from_text",
    "_gather_evidence_parallel",
    "_get_config",
    "_get_kg_df",
    "_get_rag_resources",
    "_get_reasoning_settings",
    "_get_source_label_with_status",
    "_get_summary_client",
    "_get_tool_client",
    "_is_rate_limit_exception",
    "_load_rag_result_from_jsonl",
    "_make_azure_client_with_reasoning",
    "_normalize_fusion_label",
    "_normalize_label",
    "_retry_after_seconds",
    "_run_llm_final_decision",
    "_run_summary_with_evidence",
    "_sleep_with_backoff",
    "_split_summary_output",
    "apply_final_decision",
    "attach_evidence_metadata",
    "build_agents",
    "chat_with_agents_and_summarize",
    "check_already_processed",
    "config_id_from_enabled",
    "create_model_client",
    "ensure_csv_schema",
    "generate_ablation_configs",
    "get_active_agents",
    "get_summary_system_message",
    "kg_score",
    "load_azure_openai_config",
    "main",
    "ml_score",
    "normalize_enabled_agents",
    "rag_score",
    "sanitize_model_name",
    "save_summary_to_csv",
    "selector_prompt_no_planner",
    "selector_prompt_with_planner",
    "termination",
]


if __name__ == "__main__":
    main()
