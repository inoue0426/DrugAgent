#!/usr/bin/env python
# coding: utf-8
"""Evidence collection helpers for DrugAgent."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from drugagent.tools import kg_score, ml_score, rag_score
from drugagent.utils import _is_rate_limit_exception, _sleep_with_backoff


def _build_evidence_payload(
    drug: str,
    gene: str,
    enabled_sources: List[str],
    verbose: bool = True,
    *,
    binary_mode: Optional[bool] = None,
) -> Dict[str, Any]:
    """Collect evidence tool outputs into a payload for SummaryAgent.

    Args:
        drug: Drug name.
        gene: Target gene symbol.
        enabled_sources: Enabled evidence sources.
        binary_mode: Optional override for Active/Inactive mode.

    Returns:
        Payload dict with evidence per source.
    """
    payload: Dict[str, Any] = {}

    if "ML" in enabled_sources:
        try:
            ml_result = ml_score(drug, gene, verbose=verbose, binary_mode=binary_mode)
            payload["ml_evidence"] = {
                "drug": ml_result.get("drug"),
                "target": ml_result.get("target"),
                "reason": ml_result.get("reason", ""),
                "label": ml_result.get("label", ""),
                "pKd": ml_result.get("pKd", None),
            }
        except Exception as exc:
            payload["ml_evidence"] = {
                "drug": drug,
                "target": gene,
                "reason": f"ML evidence collection failed: {exc}",
            }

    if "KG" in enabled_sources:
        try:
            kg_result = kg_score(drug, gene, verbose=verbose, binary_mode=binary_mode)
            payload["kg_evidence"] = {
                "drug": kg_result.get("drug"),
                "target": kg_result.get("target"),
                "reason": kg_result.get("reason", ""),
                "label": kg_result.get("label", ""),
                "kg_direct": kg_result.get("kg_direct", False),
                "token_usage": kg_result.get("token_usage", {}),
            }
        except Exception as exc:
            payload["kg_evidence"] = {
                "drug": drug,
                "target": gene,
                "reason": f"KG evidence collection failed: {exc}",
                "label": "NONE",
                "token_usage": {},
            }

    if "RAG" in enabled_sources:
        try:
            rag_result = rag_score(drug, gene, verbose=verbose, binary_mode=binary_mode)
            payload["rag_evidence"] = {
                "drug": rag_result.get("drug"),
                "target": rag_result.get("target"),
                "reason": rag_result.get("reason", ""),
                "label": rag_result.get("label", ""),
                "pmc_ids": rag_result.get("pmc_ids", []),
                "token_usage": rag_result.get("token_usage", {}),
            }
        except Exception as exc:
            payload["rag_evidence"] = {
                "drug": drug,
                "target": gene,
                "reason": f"RAG evidence collection failed: {exc}",
                "label": "NONE",
                "pmc_ids": [],
                "token_usage": {},
            }

    return payload


async def _gather_evidence_parallel(
    drug: str,
    gene: str,
    enabled_sources: List[str],
    verbose: bool = True,
    *,
    binary_mode: Optional[bool] = None,
) -> Dict[str, Any]:
    """Gather evidence from tools in parallel.

    Args:
        drug: Drug name.
        gene: Target gene symbol.
        enabled_sources: Enabled evidence sources.
        binary_mode: Optional override for Active/Inactive mode.

    Returns:
        Payload dict shaped for SummaryAgent.
    """

    async def run_tool(fn, *args, **kwargs):
        max_attempts = 8
        for attempt in range(1, max_attempts + 1):
            try:
                return await asyncio.to_thread(fn, *args, **kwargs)
            except Exception as exc:
                if _is_rate_limit_exception(exc) and attempt < max_attempts:
                    await _sleep_with_backoff(exc, attempt)
                    continue
                raise

    tasks = {}
    if "ML" in enabled_sources:
        tasks["ml_evidence"] = asyncio.create_task(
            run_tool(ml_score, drug, gene, verbose=verbose, binary_mode=binary_mode)
        )
    if "KG" in enabled_sources:
        tasks["kg_evidence"] = asyncio.create_task(
            run_tool(kg_score, drug, gene, verbose=verbose, binary_mode=binary_mode)
        )
    if "RAG" in enabled_sources:
        tasks["rag_evidence"] = asyncio.create_task(
            run_tool(rag_score, drug, gene, verbose=verbose, binary_mode=binary_mode)
        )

    payload: Dict[str, Any] = {}
    for key, t in tasks.items():
        try:
            res = await t
        except Exception as exc:
            if key == "ml_evidence":
                res = {
                    "drug": drug,
                    "target": gene,
                    "reason": f"ML failed: {exc}",
                    "label": "",
                    "pKd": None,
                }
            elif key == "kg_evidence":
                res = {
                    "drug": drug,
                    "target": gene,
                    "reason": f"KG failed: {exc}",
                    "label": "NONE",
                    "kg_direct": False,
                }
            else:
                res = {
                    "drug": drug,
                    "target": gene,
                    "reason": f"RAG failed: {exc}",
                    "label": "NONE",
                    "pmc_ids": [],
                }

        if key == "ml_evidence":
            payload[key] = {
                "drug": res.get("drug"),
                "target": res.get("target"),
                "reason": res.get("reason", ""),
                "label": res.get("label", ""),
                "pKd": res.get("pKd", None),
            }
        elif key == "kg_evidence":
            payload[key] = {
                "drug": res.get("drug"),
                "target": res.get("target"),
                "reason": res.get("reason", ""),
                "label": res.get("label", ""),
                "kg_direct": bool(res.get("kg_direct", False)),
                "token_usage": res.get("token_usage", {}),
            }
        else:
            payload[key] = {
                "drug": res.get("drug"),
                "target": res.get("target"),
                "reason": res.get("reason", ""),
                "label": res.get("label", ""),
                "pmc_ids": res.get("pmc_ids", []) or [],
                "token_usage": res.get("token_usage", {}),
            }

    return payload
