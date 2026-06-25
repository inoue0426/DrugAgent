#!/usr/bin/env python
# coding: utf-8
"""Orchestration flow for DrugAgent."""

from __future__ import annotations

import asyncio
from typing import Optional, Tuple

from autogen_agentchat.messages import (
    BaseChatMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_agentchat.teams import SelectorGroupChat

from drugagent.config import (
    selector_prompt_no_planner,
    selector_prompt_with_planner,
    termination,
)
from drugagent.csv import check_already_processed, save_summary_to_csv
from drugagent.evidence import _build_evidence_payload, _gather_evidence_parallel
from drugagent.summary import (
    _extract_json_block,
    _run_summary_with_evidence,
    _split_summary_output,
    apply_final_decision,
    attach_evidence_metadata,
)
from drugagent.utils import (
    _is_rate_limit_exception,
    _sleep_with_backoff,
    normalize_enabled_agents,
)


def _selector_prompt_for_agents(active_agents) -> str:
    """Resolve selector prompt based on presence of PlanningAgent.

    Args:
        active_agents: Active agent list.

    Returns:
        Selector prompt string.
    """
    has_planner = any(getattr(a, "name", "") == "PlanningAgent" for a in active_agents)
    return selector_prompt_with_planner if has_planner else selector_prompt_no_planner


async def chat_with_agents_and_summarize(
    drug: str,
    gene: str,
    verbose: bool = True,
    active_agents=None,
    enabled_agents: Optional[list[str]] = None,
    ablation: str = "full",
    model_client=None,
    model: Optional[str] = None,
    cache_enabled: bool = True,
    reasoning_effort: Optional[str] = None,
    fast_mode: bool = True,
    save_version: Optional[str] = None,
    *,
    binary_mode: Optional[bool] = None,
) -> Tuple[Optional[dict], Optional[dict], Optional[dict]]:
    """Run the DrugAgent pipeline for a single drug-target pair.

    Args:
        drug: Drug name.
        gene: Target gene symbol.
        verbose: Whether to print progress logs.
        active_agents: Agent list for non-fast mode.
        enabled_agents: Enabled evidence sources.
        ablation: Ablation mode.
        model_client: Autogen model client.
        model: Model name.
        cache_enabled: Whether to read/write cache.
        reasoning_effort: Optional reasoning effort override.
        fast_mode: Whether to use fast parallel tool mode.
        save_version: Optional CSV save version.
        binary_mode: Optional override for Active/Inactive mode.

    Returns:
        Tuple of (summary, metadata, audit_log) or (None, None, None) if skipped.
    """
    enabled_agents = normalize_enabled_agents(enabled_agents)

    if fast_mode:
        enabled_sources = enabled_agents

        if cache_enabled and check_already_processed(
            drug,
            gene,
            ablation,
            model,
            enabled_sources,
            reasoning_effort,
            save_version=save_version,
        ):
            if verbose:
                print(
                    f"[SKIP] Already processed: {drug}-{gene} for {ablation} ({model})"
                )
            return None, None, None

        payload = await _gather_evidence_parallel(
            drug, gene, enabled_sources, verbose=verbose, binary_mode=binary_mode
        )
        summary = await _run_summary_with_evidence(
            payload,
            ablation,
            enabled_sources,
            reasoning_effort=reasoning_effort,
            binary_mode=binary_mode,
        )

        await apply_final_decision(
            summary,
            enabled_sources,
            payload,
            reasoning_effort=reasoning_effort,
            binary_mode=binary_mode,
        )
        attach_evidence_metadata(summary, payload)

        if cache_enabled:
            save_summary_to_csv(
                summary,
                ablation=ablation,
                model=model,
                reasoning_effort=reasoning_effort,
                save_version=save_version,
            )
        return _split_summary_output(summary, ablation, fast_mode)

    if active_agents is None:
        raise ValueError("active_agents must be provided")
    if model_client is None:
        raise ValueError("model_client must be provided")

    if cache_enabled and check_already_processed(
        drug,
        gene,
        ablation,
        model,
        enabled_agents,
        reasoning_effort,
        save_version=save_version,
    ):
        if verbose:
            print(f"[SKIP] Already processed: {drug}-{gene} for {ablation} ({model})")
        return None, None, None

    team = SelectorGroupChat(
        active_agents,
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=_selector_prompt_for_agents(active_agents),
        allow_repeated_speaker=False,
    )

    full_conversation_messages = []
    evidence_sources_seen: set[str] = set()
    agent_source_map = {
        "RAGAgent": "RAG",
        "KGAgent": "KG",
        "MLAgent": "ML",
    }
    initial_message = (
        f"Analyze the drug-target interaction between drug: {drug} and Target: {gene}."
    )

    max_attempts = 8
    attempt = 0

    while True:
        attempt += 1
        full_conversation_messages = []
        evidence_sources_seen = set()

        try:
            async for event in team.run_stream(task=initial_message):
                if hasattr(event, "source"):
                    current_status_text = f"Agent working: {event.source}"
                else:
                    current_status_text = "Processing..."

                display_message = ""
                if isinstance(event, BaseChatMessage):
                    display_message = f"{event.source}: {event.content.strip()}"
                    full_conversation_messages.append(event)
                    mapped = agent_source_map.get(event.source)
                    if mapped:
                        evidence_sources_seen.add(mapped)
                elif isinstance(event, ToolCallRequestEvent):
                    display_message = (
                        f"{event.source} is calling a tool: {event.content[0].name}"
                    )
                elif isinstance(event, ToolCallExecutionEvent):
                    display_message = f"{event.source} tool execution complete."
                elif isinstance(event, ToolCallSummaryMessage):
                    display_message = f"{event.source} received tool result."

                if display_message and verbose:
                    print(f"[{current_status_text}] {display_message}")

            break

        except Exception as exc:
            if _is_rate_limit_exception(exc) and attempt < max_attempts:
                if verbose:
                    print(
                        f"[RateLimit] attempt={attempt}/{max_attempts} -> retrying..."
                    )
                await _sleep_with_backoff(exc, attempt)
                continue
            raise

    summary_agent_messages = [
        m.content for m in full_conversation_messages if m.source == "SummaryAgent"
    ]

    enabled_sources = normalize_enabled_agents(enabled_agents)
    missing_sources = [
        source for source in enabled_sources if source not in evidence_sources_seen
    ]

    summary = None
    summary_sources: set[str] = set()
    if summary_agent_messages:
        full_summary_text = summary_agent_messages[-1]
        try:
            summary = _extract_json_block(full_summary_text)
            root = summary.get("root", {})
            summary_sources = {
                child.get("source")
                for child in root.get("children", [])
                if child.get("source")
            }
        except ValueError:
            summary = None

    missing_in_summary = [
        source for source in enabled_sources if source not in summary_sources
    ]

    if summary is None or missing_sources or missing_in_summary:
        if verbose:
            reasons = []
            if summary is None:
                reasons.append("missing summary")
            if missing_sources:
                reasons.append(f"missing evidence: {', '.join(missing_sources)}")
            if missing_in_summary:
                reasons.append(f"summary omitted: {', '.join(missing_in_summary)}")
            print(f"[RECOVER] Rebuilding summary ({'; '.join(reasons)}).")
        payload = _build_evidence_payload(
            drug, gene, enabled_sources, verbose=verbose, binary_mode=binary_mode
        )
        try:
            summary = await _run_summary_with_evidence(
                payload,
                ablation,
                enabled_sources,
                reasoning_effort=reasoning_effort,
                binary_mode=binary_mode,
            )
        except Exception as exc:
            if verbose:
                print(f"[ERROR] Summary rebuild failed: {exc}")
            return None, None, None
    else:
        payload = _build_evidence_payload(
            drug, gene, enabled_sources, verbose=verbose, binary_mode=binary_mode
        )

    try:
        serialized_messages = []
        for m in full_conversation_messages:
            serialized_messages.append(
                {
                    "source": getattr(m, "source", None),
                    "role": getattr(m, "role", None),
                    "content": getattr(m, "content", ""),
                }
            )
        if isinstance(summary, dict):
            if "_input_payload" not in summary:
                summary["_input_payload"] = payload
            if "_input_messages" not in summary:
                summary["_input_messages"] = serialized_messages
        else:
            payload["_input_messages"] = serialized_messages
    except Exception:
        pass

    await apply_final_decision(
        summary,
        enabled_sources,
        payload,
        reasoning_effort=reasoning_effort,
        binary_mode=binary_mode,
    )
    attach_evidence_metadata(summary, payload)

    if verbose:

        def _format_fusion_label(label: str) -> str:
            value = (label or "").strip()
            if value.lower() == "low":
                return "Weak"
            return value

        root = summary.get("root", {})
        print("--------------------------------")
        print(f"[FUSION] {_format_fusion_label(root.get('fusion_label', ''))}")
        reason = root.get("fusion_reason", "")
        if reason:
            print(f"[REASON] {reason}")
        print("--------------------------------")

    if cache_enabled:
        save_summary_to_csv(
            summary,
            ablation=ablation,
            model=model,
            reasoning_effort=reasoning_effort,
            save_version=save_version,
        )
    return _split_summary_output(summary, ablation, fast_mode)
