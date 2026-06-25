#!/usr/bin/env python
# coding: utf-8
"""Tool wrappers for ML/KG/RAG evidence."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

try:
    from drugagent.kinase.kg_utils import predict_dti_strength_full_pipeline
except Exception:
    from kg_utils import predict_dti_strength_full_pipeline

try:
    from drugagent.kinase.ml_utils import get_dti_score as ml_raw_score
except Exception:
    from ml_utils import get_dti_score as ml_raw_score

try:
    from drugagent.kinase.rag_utils import run_dti_rag
except Exception:
    from rag_utils import run_dti_rag

from drugagent.config import BINARY_MODE, KG_VERSION, _get_config, _get_tool_client
from drugagent.resources import (
    _get_kg_df,
    _get_rag_resources,
    _load_rag_result_from_jsonl,
)


def _resolve_binary_mode(binary_mode: Optional[bool]) -> bool:
    """Resolve binary mode preference.

    Args:
        binary_mode: Optional override.

    Returns:
        Boolean indicating binary label mode.
    """
    return BINARY_MODE if binary_mode is None else bool(binary_mode)


def ml_score(
    drug: str,
    target: str,
    verbose: bool = True,
    *,
    binary_mode: Optional[bool] = None,
) -> Dict[str, Any]:
    """Call the ML model to score a drug-target pair.

    Args:
        drug: Drug name.
        target: Target gene symbol.
        binary_mode: Optional override for Active/Inactive mode.

    Returns:
        ML evidence dict.
    """
    binary = _resolve_binary_mode(binary_mode)
    result = ml_raw_score(drug, target, binary_class=binary)
    if verbose:
        print(f"ML raw score for {drug} - {target}: {result}")
    if not result:
        return {
            "drug": drug,
            "target": target,
            "reason": "DeepPurpose returned no result.",
            "label": "",
            "pKd": None,
        }

    row = result[0]
    label_raw = row[3] if len(row) > 3 else ""
    pkd = row[2] if len(row) > 2 else None

    # ---- canonical ML reason string (DO NOT reuse upstream 'reason') ----
    if pkd is None or (isinstance(pkd, str) and not pkd.strip()):
        reason = "DeepPurpose predicted this as pKd: NA."
    else:
        try:
            pkd_val = float(pkd)
            reason = f"DeepPurpose predicted this as pKd: {pkd_val:.3f}."
        except Exception:
            reason = f"DeepPurpose predicted this as pKd: {pkd}."

    return {
        "drug": row[0],
        "target": row[1],
        "reason": reason,
        "label": label_raw,
        "pKd": pkd,
    }


def _is_direct_kg_path(result: Dict[str, Any]) -> bool:
    """Heuristic detection for direct KG evidence.

    Args:
        result: KG pipeline result dict.

    Returns:
        True if a direct one-hop path exists.
    """
    for path in result.get("path_summaries", []) or []:
        if path.get("length") == 1:
            return True
    return False


def kg_score(
    drug: str,
    target: str,
    verbose: bool = True,
    *,
    binary_mode: Optional[bool] = None,
) -> Dict[str, Any]:
    """KG tool wrapper returning normalized label and reasoning.

    Args:
        drug: Drug name.
        target: Target gene symbol.
        binary_mode: Optional override for Active/Inactive mode.

    Returns:
        Dict with KG evidence fields.
    """
    kg_df = _get_kg_df()
    cfg = _get_config()
    client = _get_tool_client()
    binary = _resolve_binary_mode(binary_mode)
    result = predict_dti_strength_full_pipeline(
        kg_df=kg_df,
        drug=drug,
        gene=target,
        client=client,
        deployment_name=cfg["deployment_name"],
        kg_version=KG_VERSION,
        max_hops=5,
        max_paths=20,
        topn_paths_for_judge=3,
        enable_cache=True,
        binary_class=binary,
    )

    if verbose:
        print(f"KG result for {drug} - {target}: {result}")
    judgement = result.get("llm_judgement", {}) or {}
    tok = result.get("token_usage") or {}
    label_raw = str(judgement.get("label", ""))
    return {
        "drug": drug,
        "target": target,
        "reason": json.dumps(judgement, ensure_ascii=True),
        "label": label_raw,
        "kg_direct": _is_direct_kg_path(result),
        "token_usage": tok,
    }


def rag_score(
    drug: str,
    target: str,
    verbose: bool = True,
    *,
    binary_mode: Optional[bool] = None,
) -> Dict[str, Any]:
    """RAG tool wrapper returning normalized label and reasoning.

    Args:
        drug: Drug name.
        target: Target gene symbol.
        binary_mode: Optional override for Active/Inactive mode.

    Returns:
        Dict with RAG evidence fields.
    """
    cached = _load_rag_result_from_jsonl(drug, target)
    if cached is not None:
        obj = cached
    else:
        index, filtered_chunks = _get_rag_resources()
        cfg = _get_config()
        client = _get_tool_client()
        binary = _resolve_binary_mode(binary_mode)
        raw = run_dti_rag(
            drug,
            target,
            client=client,
            config=cfg,
            index=index,
            filtered_chunks=filtered_chunks,
            kg_version=KG_VERSION,
            top_pair_pmc=3,
            chunks_per_pmc=6,
            top_drug_pmc=1,
            top_target_pmc=1,
            min_chunk_words=15,
            binary_class=binary,
        )
        if verbose:
            print(f"RAG raw result for {drug} - {target}: {raw}")
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            obj = {"raw": raw}
    label_raw = ""
    stage2 = obj.get("stage2")
    if isinstance(stage2, dict):
        label_raw = str(stage2.get("label", "")).strip()
    if not label_raw:
        label_raw = str(obj.get("label", "")).strip()
    pmc_ids = []
    pmc_block = obj.get("pmcids")
    if isinstance(pmc_block, dict):
        for key in ("pair", "drug", "target"):
            ids = pmc_block.get(key)
            if isinstance(ids, list):
                pmc_ids.extend(ids)
    else:
        raw_ids = obj.get("pmc_ids") or obj.get("paper_ids") or obj.get("pmcids") or []
        if isinstance(raw_ids, list):
            pmc_ids.extend(raw_ids)
    pmc_ids = [str(x).strip() for x in pmc_ids if str(x).strip()]
    pmc_ids = sorted(set(pmc_ids))

    tok = obj.get("token_usage") or {}
    return {
        "drug": drug,
        "target": target,
        "reason": json.dumps(obj, ensure_ascii=True),
        "label": label_raw,
        "pmc_ids": pmc_ids,
        "token_usage": tok,
    }
