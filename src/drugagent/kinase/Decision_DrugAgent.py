#!/usr/bin/env python
# coding: utf-8

"""Kinase DrugAgent with Autogen multi-agent orchestration."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import re
import tempfile
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import (
    BaseChatMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, _model_info
from openai import AzureOpenAI, RateLimitError

from config_utils import get_reasoning_settings, load_azure_openai_config
from kg_utils import predict_dti_strength_full_pipeline
from ml_utils import get_dti_score as ml_raw_score
from rag_utils import load_faiss_index, load_metadata, run_dti_rag


def _resolve_repo_root() -> Path:
    """Resolve the repository root by walking up to pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[3]


def _env_path(env_key: str) -> Optional[Path]:
    """Resolve a path from an environment variable if set."""
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return None
    return Path(raw)


_REPO_ROOT = _resolve_repo_root()


@dataclass
class TokenAgg:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    # Optional breakdown by component.
    breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add(self, name: str, usage: Any) -> None:
        if usage is None:
            return
        p = int(getattr(usage, "prompt_tokens", 0) or 0)
        c = int(getattr(usage, "completion_tokens", 0) or 0)
        t = int(getattr(usage, "total_tokens", 0) or 0)
        self.prompt_tokens += p
        self.completion_tokens += c
        self.total_tokens += t
        self.calls += 1
        self.breakdown[name] = {
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": t,
        }

    def add_dict(self, name: str, d: Dict[str, Any]) -> None:
        if not d:
            return
        p = int(d.get("prompt_tokens", 0) or 0)
        c = int(d.get("completion_tokens", 0) or 0)
        t = int(d.get("total_tokens", 0) or 0)
        self.prompt_tokens += p
        self.completion_tokens += c
        self.total_tokens += t
        self.calls += int(d.get("calls", 1) or 1)
        self.breakdown[name] = {
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": t,
        }


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

_CONFIG: Optional[Dict[str, Any]] = None
_REASONING_SETTINGS: Optional[Dict[str, Any]] = None


def _get_config() -> Dict[str, Any]:
    """Load and cache Azure OpenAI config."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_azure_openai_config()
    return _CONFIG


def _get_reasoning_settings() -> Optional[Dict[str, Any]]:
    """Load and cache reasoning settings if available."""
    global _REASONING_SETTINGS
    if _REASONING_SETTINGS is None:
        _REASONING_SETTINGS = get_reasoning_settings()
    return _REASONING_SETTINGS


KG_PATH = _env_path("DRUGAGENT_KG_PATH") or _REPO_ROOT / "data" / "KG+BDB.csv.gz"
RAG_INDEX_PATH = _env_path("DRUGAGENT_RAG_INDEX_PATH") or _REPO_ROOT / "rag_index.faiss"
RAG_META_PATH = _env_path("DRUGAGENT_RAG_META_PATH") or _REPO_ROOT / "rag_metadata.json"
KG_VERSION = "kg_2026_02_26"

LABEL_ORDER = {
    "NONE": 0,
    "WEAK": 1,
    "MODERATE": 2,
    "STRONG": 3,
    "LOW": 1,
    "INSUFFICIENT": 1,
}

FUSION_ORD = {"Low": 0, "Moderate": 1, "Strong": 2}
FUSION_INV_ORD = {0: "Low", 1: "Moderate", 2: "Strong"}

ALL_EVIDENCE_AGENTS = ["ML", "KG", "RAG"]


_tool_client: AzureOpenAI | None = None


def _get_tool_client() -> AzureOpenAI:
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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _normalize_label(label: str) -> str:
    """Normalize a label string to a known categorical value.

    Args:
        label: Raw label string.

    Returns:
        Uppercased label or an empty string if unknown.
    """
    value = str(label).strip().upper()
    for dash in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212"):
        value = value.replace(dash, "-")
    value = value.replace(" ", "_").replace("-", "_")
    while "__" in value:
        value = value.replace("__", "_")
    if value in {"MODERATE_HIGH", "MODERATE_LOW"}:
        value = "MODERATE"
    return value if value in LABEL_ORDER else ""


def _normalize_fusion_label(label: str | None) -> Optional[str]:
    """Normalize labels into Low/Moderate/Strong for fusion.

    Args:
        label: Raw label string.

    Returns:
        Normalized fusion label or None if unknown.
    """
    if not label:
        return None
    value = str(label).strip().lower().replace("_", "-").replace(" ", "-")
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
    """Normalize enabled agent list to known evidence agent names."""
    if not enabled_agents:
        return ALL_EVIDENCE_AGENTS[:]
    normalized = []
    for agent in enabled_agents:
        name = str(agent).strip()
        for candidate in ALL_EVIDENCE_AGENTS:
            if name.lower() == candidate.lower():
                if candidate not in normalized:
                    normalized.append(candidate)
    return normalized


def config_id_from_enabled(enabled_agents: List[str]) -> str:
    """Build a stable config id from enabled agents."""
    if not enabled_agents:
        return "none"
    ordered = [a.lower() for a in ALL_EVIDENCE_AGENTS if a in enabled_agents]
    return "_".join(ordered)


def generate_ablation_configs() -> List[List[str]]:
    """Generate all non-empty ablation combinations."""
    combos: List[List[str]] = []
    for size in range(1, len(ALL_EVIDENCE_AGENTS) + 1):
        for subset in combinations(ALL_EVIDENCE_AGENTS, size):
            combos.append(list(subset))
    return combos


def _is_rate_limit_exception(exc: Exception) -> bool:
    # openai.RateLimitError is definitive.
    if isinstance(exc, RateLimitError):
        return True
    # Autogen can wrap errors or only leave a message; handle both.
    msg = str(exc)
    return (
        ("RateLimit" in msg)
        or ("RateLimitReached" in msg)
        or ("Error code: 429" in msg)
    )


def _retry_after_seconds(exc: Exception, default: float = 3.0) -> float:
    # OpenAI SDK may include retry-after in response headers.
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
    # Honor retry-after and use exponential backoff with jitter to avoid collisions.
    base_wait = _retry_after_seconds(exc, default=3.0)
    exp_wait = 1.7**attempt
    wait = max(base_wait, exp_wait) + random.uniform(0, 0.75)
    await asyncio.sleep(wait)


# -----------------------------------------------------------------------------
# Local resource loaders
# -----------------------------------------------------------------------------

_KG_DF: Optional[pd.DataFrame] = None
_RAG_INDEX = None
_RAG_CHUNKS = None


def _get_kg_df() -> pd.DataFrame:
    """Load the KG dataframe once."""
    global _KG_DF
    if _KG_DF is None:
        if not KG_PATH.exists():
            raise FileNotFoundError(f"KG file not found: {KG_PATH}")
        _KG_DF = pd.read_csv(KG_PATH, index_col=0)
    return _KG_DF


def _get_rag_resources() -> Tuple[Any, Any]:
    """Load RAG index and metadata once."""
    global _RAG_INDEX, _RAG_CHUNKS
    if _RAG_INDEX is None:
        _RAG_INDEX = load_faiss_index(str(RAG_INDEX_PATH))
    if _RAG_CHUNKS is None:
        _RAG_CHUNKS = load_metadata(str(RAG_META_PATH))
    return _RAG_INDEX, _RAG_CHUNKS


# -----------------------------------------------------------------------------
# Tool wrappers (local KG/ML/RAG)
# -----------------------------------------------------------------------------


def ml_score(drug: str, target: str) -> Dict[str, Any]:
    result = ml_raw_score(drug, target)
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
        # numeric pretty-print (optional)
        try:
            pkd_val = float(pkd)
            reason = f"DeepPurpose predicted this as pKd: {pkd_val:.3f}."
        except Exception:
            reason = f"DeepPurpose predicted this as pKd: {pkd}."

    # If you want to include the label, replace with the next line.
    # reason = reason[:-1] + f" Label: {label_raw}."

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


def kg_score(drug: str, target: str) -> Dict[str, Any]:
    """KG tool wrapper returning normalized label and reasoning.

    Args:
        drug: Drug name.
        target: Target gene symbol.

    Returns:
        Dict with KG evidence fields.
    """
    kg_df = _get_kg_df()
    cfg = _get_config()
    client = _get_tool_client()
    result = predict_dti_strength_full_pipeline(
        kg_df=kg_df,
        drug=drug,
        gene=target,
        client=client,
        deployment_name=cfg["deployment_name"],
        kg_version=KG_VERSION,
        max_hops=5,
        max_paths=10,
        topn_paths_for_judge=3,
    )

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


def rag_score(drug: str, target: str) -> Dict[str, Any]:
    """RAG tool wrapper returning normalized label and reasoning.

    Args:
        drug: Drug name.
        target: Target gene symbol.

    Returns:
        Dict with RAG evidence fields.
    """
    index, filtered_chunks = _get_rag_resources()
    cfg = _get_config()
    client = _get_tool_client()
    raw = run_dti_rag(
        drug,
        target,
        client=client,
        config=cfg,
        index=index,
        filtered_chunks=filtered_chunks,
        kg_version=KG_VERSION,
    )
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


# -----------------------------------------------------------------------------
# Summary agent system message
# -----------------------------------------------------------------------------


def get_summary_system_message(ablation: str, enabled_sources: List[str]) -> str:
    """Build summary system message for enabled sources.

    Args:
        ablation: Ablation mode.
        enabled_sources: Evidence sources to include.

    Returns:
        Summary system prompt.
    """
    source_lines = []
    if "ML" in enabled_sources:
        source_lines.append("- ML-based prediction scores and reasoning")
    if "KG" in enabled_sources:
        source_lines.append("- Knowledge Graph (KG) evidence and reasoning")
    if "RAG" in enabled_sources:
        source_lines.append("- RAG literature evidence and reasoning")
    sources_text = "\n".join(source_lines)

    input_format = """You will receive a JSON object with some of the following keys (only include keys for available sources):
{
  "ml_evidence": {"drug": string, "target": string, "reason": string, "label": string?, "pKd": number?}
  "kg_evidence": {"drug": string, "target": string, "reason": string, "label": string?, "kg_direct": bool?},
  "rag_evidence": {"drug": string, "target": string, "reason": string, "label": string?, "pmc_ids": [string]}
}"""

    schema_children = []
    for source in enabled_sources:
        if source == "RAG":
            schema_children.append(
                """{
        "type": "evidence_analysis",
        "source": "RAG",
        "thought": string,
        "action": string,
        "observation": string,
        "label": string,
        "pmc_ids": [string]
      }"""
            )
        else:
            schema_children.append(
                f"""{{
        "type": "evidence_analysis",
        "source": "{source}",
        "thought": string,
        "action": string,
        "observation": string,
        "label": string
      }}"""
            )

    children_text = ",\n      ".join(schema_children)
    schema_text = (
        "```json\n"
        "{\n"
        '  "type": "reasoning_tree",\n'
        '  "drug": string,\n'
        '  "target": string,\n'
        '  "root": {\n'
        '    "type": "comparison",\n'
        '    "children": [\n'
        f"      {children_text}\n"
        "    ],\n"
        '    "summary_reasoning": string\n'
        "  }\n"
        "}\n"
        "```\n"
    )

    return f"""
Task:
You will synthesize evidence from the provided sources regarding drug-target interactions:
{sources_text}

Your role is to:
1. Analyze each evidence source independently and summarize its reasoning using a structured format.
2. Compare and contrast the results, identifying agreements and conflicts.
3. Assess the biological plausibility and reliability of each source's reasoning.
4. Use provided categorical labels from KG and RAG evidence when available.
5. Do NOT compute the final decision; it is computed deterministically downstream.

You must output a symbolic reasoning tree in valid JSON format.

### Output requirements (MUST follow strictly):

- Output **ONLY a valid JSON object**, nothing before or after.
- The JSON must exactly match the schema below. All property names must be **enclosed in double quotes**.
- No markdown, no explanation, no comments, no additional text.
- If you include numeric values, they must be JSON numbers (not strings or formulas).
- Do NOT include weights in the output.
- "pmc_ids" must be an array of strings, even if only one.
- After the JSON object, output the word `TERMINATE` **on a new line by itself**.

### Label handling (MUST follow strictly):

- For KG and RAG, if a `label` is provided in the evidence, use it directly.
- If no label is provided, you may judge evidence strength from the reasoning text.
- In each `observation` field, explicitly include the chosen label.
- For RAG, always include **all** received PMC IDs in `pmc_ids` with no omissions.
- If a source is missing from the input, omit that source from `children`.

Input format:
{input_format}

Schema:
{schema_text}
"""


# -----------------------------------------------------------------------------
# Evidence payload and summary parsing
# -----------------------------------------------------------------------------


def _extract_json_block(text: str) -> Dict[str, Any]:
    content = (text or "").strip()

    # 1) Strip code fences (model sometimes emits ```json blocks).
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    # 2) Remove TERMINATE wherever it appears.
    content = re.sub(r"\bTERMINATE\b", "", content).strip()

    # 3) Drop leading text before the first '{'.
    start = content.find("{")
    if start == -1:
        raise ValueError("No JSON object start '{' found in model output.")
    content2 = content[start:].lstrip()

    # 4) raw_decode to read the first JSON object and ignore trailing text.
    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(content2)
        return obj
    except json.JSONDecodeError:
        # 5) As a last resort, slice to the last '}' and retry.
        end_brace = content2.rfind("}")
        if end_brace != -1:
            return json.loads(content2[: end_brace + 1])
        raise ValueError("Failed to parse summary JSON from model output.")


def _build_evidence_payload(
    drug: str, gene: str, enabled_sources: List[str]
) -> Dict[str, Any]:
    """Collect evidence tool outputs into a payload for SummaryAgent.

    Args:
        drug: Drug name.
        gene: Target gene symbol.
        enabled_sources: Enabled evidence sources.

    Returns:
        Payload dict with evidence per source.
    """
    payload: Dict[str, Any] = {}

    if "ML" in enabled_sources:
        try:
            ml_result = ml_score(drug, gene)
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
            kg_result = kg_score(drug, gene)
            payload["kg_evidence"] = {
                "drug": kg_result.get("drug"),
                "target": kg_result.get("target"),
                "reason": kg_result.get("reason", ""),
                "label": kg_result.get("label", ""),
                "kg_direct": kg_result.get("kg_direct", False),
                "token_usage": kg_result.get("token_usage", {}),  # Added.
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
            rag_result = rag_score(drug, gene)
            payload["rag_evidence"] = {
                "drug": rag_result.get("drug"),
                "target": rag_result.get("target"),
                "reason": rag_result.get("reason", ""),
                "label": rag_result.get("label", ""),
                "pmc_ids": rag_result.get("pmc_ids", []),
                "token_usage": rag_result.get("token_usage", {}),  # Added.
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


_summary_clients: Dict[Optional[str], AzureOpenAI] = {}


def _make_azure_client_with_reasoning(reasoning_effort: Optional[str]) -> AzureOpenAI:
    # NOTE: AzureOpenAI.__init__ does NOT accept a 'reasoning' kwarg.
    # Pass reasoning only as a request-time param (request_kwargs), not into constructor.
    cfg = _get_config()
    return AzureOpenAI(
        api_key=cfg["api_key"],
        azure_endpoint=cfg["endpoint"],
        api_version=cfg["api_version"],
    )


def _get_summary_client(reasoning_effort: Optional[str] = None) -> AzureOpenAI:
    """
    Create or reuse a summary LLM client keyed by reasoning_effort.
    - reasoning_effort: None | "low" | "medium" | "high"
    Returns an AzureOpenAI client that will (when possible) include the reasoning param.
    """
    key = reasoning_effort if reasoning_effort is not None else "__NONE__"
    client = _summary_clients.get(key)
    if client is None:
        client = _make_azure_client_with_reasoning(reasoning_effort)
        _summary_clients[key] = client
    return client


async def _run_summary_with_evidence(
    payload: Dict[str, Any],
    ablation: str,
    enabled_sources: List[str],
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Run SummaryAgent deterministically with explicit evidence payload.

    Normalize reasoning_effort and ensure it's passed in a stable, backend-friendly form.
    """
    # --- normalize reasoning_effort (allow "High", "high", " HIGH ") ---
    if reasoning_effort is not None:
        reasoning_effort = str(reasoning_effort).strip()
        if reasoning_effort == "":
            reasoning_effort = None
        else:
            # treat the explicit string "none" (any case) as None
            if reasoning_effort.lower() == "none":
                reasoning_effort = None
            else:
                # normalize to a canonical lower-case token expected by service
                reasoning_effort = reasoning_effort.lower()

    cfg = _get_config()
    reasoning_settings = _get_reasoning_settings()
    system_message = get_summary_system_message(
        ablation=ablation, enabled_sources=enabled_sources
    )

    # Create/lookup client keyed by normalized reasoning_effort
    client = _get_summary_client(reasoning_effort)

    # Build request kwargs; keep 'reasoning' only as a Responses-API style param.
    request_kwargs = {"temperature": 0, "seed": 42}
    if reasoning_effort is not None:
        # pass a canonical structure
        request_kwargs["reasoning"] = {"effort": reasoning_effort}
    elif reasoning_settings is not None:
        request_kwargs["reasoning"] = reasoning_settings

    # --- retry for summary call using Responses API / fallback to chat.completions ---
    max_attempts = 8
    response = None
    for attempt in range(1, max_attempts + 1):
        try:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": json.dumps(payload)},
            ]

            def _call_llm():
                # Try Responses-style first (may accept 'reasoning')
                try:
                    return client.responses.create(
                        model=cfg["deployment_name"],
                        messages=messages,
                        **request_kwargs,
                    )
                except TypeError:
                    # param mismatch -> try alternative param names
                    pass
                except Exception:
                    raise

                try:
                    return client.responses.create(
                        deployment_id=cfg["deployment_name"],
                        input=messages,
                        **request_kwargs,
                    )
                except TypeError:
                    pass
                except Exception:
                    raise

                # Fallback to legacy chat.completions.create
                # -> chat API may not accept 'reasoning' so remove it
                chat_kwargs = dict(request_kwargs)
                if "reasoning" in chat_kwargs:
                    chat_kwargs.pop("reasoning")

                return client.chat.completions.create(
                    model=cfg["deployment_name"],
                    messages=messages,
                    **chat_kwargs,
                )

            response = await asyncio.to_thread(_call_llm)
            break
        except Exception as exc:
            if _is_rate_limit_exception(exc) and attempt < max_attempts:
                await _sleep_with_backoff(exc, attempt)
                continue
            raise

    if response is None:
        raise RuntimeError("No response received from LLM after retries.")

    # --- robust extraction of textual content from various response shapes ---
    content = ""
    try:
        # chat.completions style
        if hasattr(response, "choices") and response.choices:
            choice0 = response.choices[0]
            if (
                hasattr(choice0, "message")
                and getattr(choice0.message, "content", None) is not None
            ):
                content = choice0.message.content or ""
            else:
                try:
                    content = choice0["message"]["content"]
                except Exception:
                    content = ""
        else:
            # Responses API style
            out = getattr(response, "output", None) or getattr(
                response, "outputs", None
            )
            if out and isinstance(out, (list, tuple)) and len(out) > 0:
                first = out[0]
                if isinstance(first, dict):
                    cont = (
                        first.get("content") or first.get("body") or first.get("text")
                    )
                    if isinstance(cont, list) and cont:
                        if isinstance(cont[0], dict):
                            text = (
                                cont[0].get("text")
                                or cont[0].get("value")
                                or cont[0].get("content")
                            )
                            if text:
                                content = text
                        elif isinstance(cont[0], str):
                            content = cont[0]
                    elif isinstance(cont, str):
                        content = cont
                else:
                    text_attr = getattr(first, "text", None) or getattr(
                        first, "content", None
                    )
                    if isinstance(text_attr, str):
                        content = text_attr
            if not content and hasattr(response, "output_text"):
                content = getattr(response, "output_text") or ""
            if not content and hasattr(response, "text"):
                content = getattr(response, "text") or ""
    except Exception:
        content = ""

    if not content:
        try:
            content = str(response)
        except Exception:
            content = ""

    # --- parse JSON block as before ---
    obj = _extract_json_block(content)

    # --- BEGIN PATCH: attach input payload/messages for auditability ---
    # Attach the input payload and the exact messages sent to the LLM for reproducibility.
    # We keep the full payload in the JSON object (raw.jsonl), but for CSV we will truncate later.
    try:
        # Raw payload (tool outputs) is safe to attach (already serializable).
        obj["_input_payload"] = payload

        # messages that were sent to the model (system + user)
        # 'messages' exists in this scope above where we built the LLM call
        try:
            obj["_input_messages"] = messages
        except Exception:
            # If messages unavailable (edge-case), at least record a minimal representation
            obj["_input_messages"] = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": json.dumps(payload)},
            ]
    except Exception:
        # attach best-effort; never raise here
        pass
    # --- END PATCH

    # --- BEGIN PATCH: ensure requested reasoning is recorded in returned summary object
    # Guarantee that the summary JSON contains a root dict and an explicit record
    # of the reasoning_effort we used (so downstream processing / CSV / parsing sees it).
    if not isinstance(obj, dict):
        obj = {"root": {}}
    if "root" not in obj or not isinstance(obj["root"], dict):
        obj["root"] = {}

    # record both the canonical token (e.g., "high"/"medium"/"low" or None) the call used
    # and a human-friendly requested representation. This does not overwrite LLM text
    # but guarantees the field exists for auditing and CSV output.
    obj["root"]["reasoning_effort"] = reasoning_effort or ""
    # Keep an audit field indicating whether defaults were used as fallback.
    obj["root"]["requested_reasoning"] = {
        "effort_param": reasoning_effort or None,
        "used_defaults": (reasoning_effort is None and reasoning_settings is not None),
    }
    # --- END PATCH

    # Added: summary token usage if available.
    usage = getattr(response, "usage", None)
    if usage is None:
        usage = (
            getattr(response, "meta", None)
            or getattr(response, "output_meta", None)
            or getattr(response, "raw", None)
        )

    obj["_token_usage_summary"] = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "calls": 1,
    }
    return obj


# -----------------------------------------------------------------------------
# Fusion rules (from 9_DrugAgent.ipynb)
# -----------------------------------------------------------------------------


def _median_label(labels: Tuple[str, str, str]) -> str:
    """Deterministic ordinal median of three labels."""
    vals = sorted(FUSION_ORD[label] for label in labels)
    return FUSION_INV_ORD[vals[1]]


def _count(label: str, labels: Tuple[str, str, str]) -> int:
    return sum(item == label for item in labels)


def _any_in(values: set, labels: Tuple[str, str, str]) -> bool:
    return any(item in values for item in labels)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    description: str
    predicate: Callable[[Dict[str, Any]], bool]
    output_label: Callable[[Dict[str, Any]], str]
    confidence: str


def make_reviewer_friendly_rules_v2() -> Tuple[Rule, ...]:
    """
    Deterministic, non-tuned rules, designed to be explainable + auditable.

    Assumes row has: ml, kg, rag (labels), optional kg_direct (bool), optional n_papers (int).
    """

    def out_fixed(lbl: str) -> Callable[[Dict[str, Any]], str]:
        return lambda _: lbl

    def out_median() -> Callable[[Dict[str, Any]], str]:
        return lambda r: _median_label((r["ml"], r["kg"], r["rag"]))

    return (
        Rule(
            "S0",
            "All three modules predict Strong (ML=KG=RAG=Strong).",
            lambda r: r["ml"] == "Strong"
            and r["kg"] == "Strong"
            and r["rag"] == "Strong",
            out_fixed("Strong"),
            "HIGH",
        ),
        Rule(
            "S1",
            "ML and RAG both predict Strong (conservative literature confirmation).",
            lambda r: r["ml"] == "Strong" and r["rag"] == "Strong",
            out_fixed("Strong"),
            "HIGH",
        ),
        Rule(
            "S2",
            "KG Strong is trusted when a direct structural edge exists and at least one other source supports (ML>=Moderate or RAG=Strong).",
            lambda r: (
                r["kg"] == "Strong"
                and r.get("kg_direct", False) is True
                and (r["ml"] in {"Moderate", "Strong"} or r["rag"] == "Strong")
            ),
            out_fixed("Strong"),
            "HIGH",
        ),
        Rule(
            "S3p",
            "ML Strong accepted only when KG is also Strong.",
            lambda r: r["ml"] == "Strong" and r["kg"] == "Strong",
            out_fixed("Strong"),
            "MEDIUM",
        ),
        Rule(
            "L1",
            "All three modules predict Low.",
            lambda r: r["ml"] == "Low" and r["kg"] == "Low" and r["rag"] == "Low",
            out_fixed("Low"),
            "HIGH",
        ),
        Rule(
            "L0",
            "At least two modules predict Low (2-of-3 Low vote).",
            lambda r: _count("Low", (r["ml"], r["kg"], r["rag"])) >= 2,
            out_fixed("Low"),
            "MEDIUM",
        ),
        Rule(
            "M0",
            "Strong/Low conflict across modules -> return Moderate (uncertainty buffer).",
            lambda r: _any_in({"Strong"}, (r["ml"], r["kg"], r["rag"]))
            and _any_in({"Low"}, (r["ml"], r["kg"], r["rag"])),
            out_fixed("Moderate"),
            "LOW",
        ),
        Rule(
            "M1m",
            "Fallback to ordinal median of (ML, KG, RAG).",
            lambda _: True,
            out_median(),
            "LOW",
        ),
    )


def _median_label2(a: str, b: str) -> str:
    """
    Deterministic ordinal median of two labels.
    If same -> that label; else -> Moderate.
    """
    return a if a == b else "Moderate"


def make_reviewer_friendly_rules_2src() -> Tuple[Rule, ...]:
    """
    2-source deterministic rules.
    Input row has:
      - a_src, b_src : str in {"ML","KG","RAG"}
      - a, b         : labels in {"Low","Moderate","Strong"}
      - kg_direct    : bool (only meaningful if KG is present)
    """

    def out_fixed(lbl: str) -> Callable[[Dict[str, Any]], str]:
        return lambda _: lbl

    def out_agreement_or_moderate() -> Callable[[Dict[str, Any]], str]:
        return lambda r: _median_label2(r["a"], r["b"])

    def has_pair(r, x, y) -> bool:
        return {r["a"], r["b"]} == {x, y}

    def includes_src(r, s: str) -> bool:
        return (r["a_src"] == s) or (r["b_src"] == s)

    def label_of(r, s: str) -> Optional[str]:
        if r["a_src"] == s:
            return r["a"]
        if r["b_src"] == s:
            return r["b"]
        return None

    return (
        Rule(
            "2S0",
            "Two-source agreement (a=b).",
            lambda r: r["a"] == r["b"],
            lambda r: r["a"],
            "HIGH",
        ),
        Rule(
            "2S2",
            "KG Strong trusted when direct 1-hop edge exists and the other source is >=Moderate.",
            lambda r: (
                includes_src(r, "KG")
                and (label_of(r, "KG") == "Strong")
                and (r.get("kg_direct", False) is True)
                and (
                    # other source >= Moderate
                    (r["a_src"] != "KG" and r["a"] in {"Moderate", "Strong"})
                    or (r["b_src"] != "KG" and r["b"] in {"Moderate", "Strong"})
                )
            ),
            out_fixed("Strong"),
            "HIGH",
        ),
        Rule(
            "2M0",
            "Strong/Low conflict across two sources -> Moderate (uncertainty buffer).",
            lambda r: has_pair(r, "Strong", "Low"),
            out_fixed("Moderate"),
            "LOW",
        ),
        Rule(
            "2M1m",
            "Fallback: if disagreement remains, return Moderate (2-point ordinal median).",
            lambda _: True,
            out_agreement_or_moderate(),
            "LOW",
        ),
    )


def _get_source_label_with_status(
    root: dict, source: str, enabled_agents: List[str] | None
) -> Tuple[Optional[str], str]:
    """Find label and status for a given evidence source.

    Args:
        root: Root node of the reasoning tree.
        source: Evidence source name (ML, KG, RAG).
        enabled_agents: Enabled evidence sources.

    Returns:
        Tuple of (label, status) where status is present/missing/disabled.
    """
    if enabled_agents is not None and source not in enabled_agents:
        return None, "disabled"
    for child in root.get("children", []):
        if child.get("source") != source:
            continue
        status = str(child.get("status", "present")).strip().lower()
        if status not in {"present", "missing", "disabled"}:
            status = "present"
        label = _normalize_label(str(child.get("label", "")))
        if label:
            return label, status

        observation = child.get("observation", "")
        if isinstance(observation, dict):
            label = _normalize_label(str(observation.get("label", "")))
            if label:
                return label, status
        elif isinstance(observation, str):
            # Try to parse JSON observation if present.
            try:
                obs_obj = json.loads(observation)
            except (json.JSONDecodeError, TypeError):
                obs_obj = None
            if isinstance(obs_obj, dict):
                label = _normalize_label(str(obs_obj.get("label", "")))
                if label:
                    return label, status
            label = _normalize_label(_extract_label_from_text(observation))
            if label:
                return label, status
        return "NONE", status
    return "NONE", "missing"


def _extract_label_from_text(text: str) -> str:
    match = re.search(
        "Label\\s*[:=]\\s*([A-Za-z][A-Za-z\\s\\-\\u2013\\u2014]+)",
        text or "",
    )
    if not match:
        return ""
    value = match.group(1).strip()
    value = re.split(r"[\.;,\)\]]", value, maxsplit=1)[0].strip()
    return value


def attach_evidence_metadata(summary: dict, payload: dict) -> None:
    """Attach tool-derived evidence metadata into summary.root for deterministic explanation."""
    if not isinstance(summary, dict):
        return
    root = summary.setdefault("root", {})
    root["_evidence"] = root.get("_evidence", {})

    ml = payload.get("ml_evidence") or {}
    kg = payload.get("kg_evidence") or {}
    rag = payload.get("rag_evidence") or {}

    # these are tool-grounded (not LLM hallucination-prone)
    root["kg_direct"] = bool(kg.get("kg_direct", False))
    root["_evidence"]["ml_reason"] = str(ml.get("reason", ""))[:2000]
    root["_evidence"]["kg_reason"] = str(kg.get("reason", ""))[:2000]
    root["_evidence"]["rag_reason"] = str(rag.get("reason", ""))[:4000]

    root["_evidence"]["pKd"] = ml.get("pKd", None)
    root["_evidence"]["pmc_ids"] = rag.get("pmc_ids", []) or []

    agg = TokenAgg()
    agg.add_dict("kg", (kg.get("token_usage") or {}))
    agg.add_dict("rag", (rag.get("token_usage") or {}))
    agg.add_dict("ml", (ml.get("token_usage") or {}))  # Added.
    agg.add_dict("summary", (summary.get("_token_usage_summary") or {}))
    agg.add_dict("addendum", (root.get("_token_usage_addendum") or {}))

    root["token_usage_total"] = {
        "prompt_tokens": agg.prompt_tokens,
        "completion_tokens": agg.completion_tokens,
        "total_tokens": agg.total_tokens,
        "calls": agg.calls,
        "breakdown": agg.breakdown,
    }


def _extract_label_from_reasoning_text(text: str) -> Optional[str]:
    """Extract an explicit label from summary_reasoning, if present."""
    if not text:
        return None
    # Look for common label tokens (Strong/Moderate/Weak/Low, etc.).
    m = re.search(
        r"\b(Strong|Moderate|Weak|Low|Insufficient|NONE|None)\b",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        lbl = m.group(1)
        # normalize to fusion labels
        norm = _normalize_fusion_label(lbl)
        return norm
    return None


def _compute_alignment_status(summary: dict) -> str:
    """
    Compare LLM's free-text label (if any extracted) with deterministic fusion_label.
    Returns: "aligned" | "near_miss" | "conflict" | "unknown"
    """
    root = summary.get("root", {}) or {}
    fusion_label = root.get("fusion_label", "")
    if not fusion_label:
        return "unknown"
    llm_text = (root.get("summary_reasoning") or "").strip()
    llm_label = _extract_label_from_reasoning_text(llm_text)
    if llm_label is None:
        return "unknown"
    if llm_label == fusion_label:
        return "aligned"
    # near miss: e.g., LLM said "Weak" but mapping to "Low" etc.
    if llm_label in {"Low", "Moderate", "Strong"} and fusion_label in {
        "Low",
        "Moderate",
        "Strong",
    }:
        # ordinal distance check (Low<->Strong is bigger than Low<->Moderate)
        ord_map = {"Low": 0, "Moderate": 1, "Strong": 2}
        try:
            d = abs(ord_map[llm_label] - ord_map[fusion_label])
            if d == 1:
                return "near_miss"
            else:
                return "conflict"
        except Exception:
            return "conflict"
    return "conflict"


def apply_rule_based_explanation(summary: dict) -> None:
    """Generate deterministic explanation text that strictly follows the fired fusion rule.
    Preserve LLM summary_reasoning if present; append deterministic fusion explanation.
    Also compute alignment_status and lower fusion_conf for conflicts.
    """
    if not isinstance(summary, dict):
        return
    root = summary.get("root", {}) or {}

    rule_id = str(root.get("fusion_rule", "NA"))
    fusion_label = str(root.get("fusion_label", "NA"))
    conf = str(root.get("fusion_conf", "LOW"))

    ml = root.get("ml_label", "")
    kg = root.get("kg_label", "")
    rag = root.get("rag_label", "")
    kg_direct = bool(root.get("kg_direct", False))

    ev = root.get("_evidence", {}) or {}
    pkd = ev.get("pKd", None)
    pmc_ids = ev.get("pmc_ids", []) or []

    # ---- rule templates (deterministic) ----
    templates = {
        "S0": lambda: (
            f"Rule S0 fired: all modules agree on Strong (ML={ml}, KG={kg}, RAG={rag}). "
            f"Therefore final label is Strong with HIGH confidence."
        ),
        "S1": lambda: (
            f"Rule S1 fired: ML and RAG both support Strong (ML={ml}, RAG={rag}). "
            f"Literature confirmation makes the decision conservative -> Strong (HIGH)."
        ),
        "S2": lambda: (
            f"Rule S2 fired: KG predicts Strong with a direct structural edge (kg_direct={kg_direct}), "
            f"and at least one other source supports (ML or RAG). -> Strong (HIGH)."
        ),
        "S3p": lambda: (
            f"Rule S3p fired: ML Strong is accepted only when KG is also Strong (ML={ml}, KG={kg}). "
            f"-> Strong (MEDIUM)."
        ),
        "L1": lambda: (
            f"Rule L1 fired: all modules predict Low (ML={ml}, KG={kg}, RAG={rag}). "
            f"-> Low (HIGH)."
        ),
        "L0": lambda: (
            f"Rule L0 fired: at least two modules vote Low (ML={ml}, KG={kg}, RAG={rag}). "
            f"-> Low (MEDIUM)."
        ),
        "M0": lambda: (
            f"Rule M0 fired: Strong/Low conflict detected across modules (ML={ml}, KG={kg}, RAG={rag}). "
            f"Return Moderate as uncertainty buffer -> Moderate (LOW)."
        ),
        "M1m": lambda: (
            f"Rule M1m fired: fallback to ordinal median of (ML, KG, RAG) = {fusion_label} "
            f"(ML={ml}, KG={kg}, RAG={rag}). -> {fusion_label} (LOW)."
        ),
        "1S0": lambda: (
            f"Rule 1S0 fired: only one evidence source available ({root.get('fusion_sources', [])}). "
            f"Pass-through label -> {fusion_label} (LOW)."
        ),
        "2S0": lambda: (
            f"Rule 2S0 fired: two-source agreement -> {fusion_label} (HIGH). "
            f"Sources={root.get('fusion_sources', [])}."
        ),
        "2S2": lambda: (
            f"Rule 2S2 fired: KG Strong with direct edge (kg_direct={kg_direct}) and the other source supports (>=Moderate). "
            f"-> Strong (HIGH). Sources={root.get('fusion_sources', [])}."
        ),
        "2M0": lambda: (
            f"Rule 2M0 fired: Strong/Low conflict across two sources. "
            f"Return Moderate as uncertainty buffer -> Moderate (LOW). Sources={root.get('fusion_sources', [])}."
        ),
        "2M1m": lambda: (
            f"Rule 2M1m fired: fallback for two-source disagreement -> {fusion_label} (LOW). "
            f"Sources={root.get('fusion_sources', [])}."
        ),
    }

    core = templates.get(rule_id, lambda: f"No matching rule template for {rule_id}.")()

    # ---- add tool-grounded anchors ----
    anchors = []
    if pkd is not None and str(pkd).strip() != "":
        anchors.append(f"ML pKd={pkd}")
    if pmc_ids:
        anchors.append(
            f"RAG PMCIDs={','.join(map(str, pmc_ids[:20]))}"
            + ("..." if len(pmc_ids) > 20 else "")
        )

    if anchors:
        core = core + " Evidence anchors: " + " | ".join(anchors) + "."

    # --- preserve original LLM summary_reasoning if present ---
    orig = (root.get("summary_reasoning") or "").strip()
    if orig:
        # keep original LLM reasoning first, then append deterministic rule explanation
        combined = orig + "\n\n" + core
    else:
        combined = core
    root["fusion_explanation"] = core
    root["summary_reasoning"] = combined

    # --- compute alignment ---
    alignment = _compute_alignment_status(summary)
    root["reasoning_alignment"] = alignment

    # if clear conflict, lower declared fusion confidence and add flag in reason
    if alignment == "conflict":
        # weaken confidence if currently HIGH/MEDIUM
        if root.get("fusion_conf", "") in {"HIGH", "MEDIUM"}:
            root["fusion_conf"] = "MEDIUM"
        # annotate reason
        root["fusion_reason"] = (
            str(root.get("fusion_reason", ""))
            + f" NOTE: LLM summary conflicts with fusion result (alignment={alignment})."
        ).strip()
    elif alignment == "near_miss":
        root["fusion_reason"] = (
            str(root.get("fusion_reason", ""))
            + f" NOTE: LLM and fusion nearly agree (alignment={alignment})."
        ).strip()

    summary["root"] = root


def _get_explainer_client() -> AzureOpenAI:
    # Reuse the summary client.
    return _get_summary_client()


async def _generate_rule_aligned_addendum(summary: dict) -> str:
    """
    Generate a short addendum that does not contradict the deterministic rule.
    The addendum must not change the decision reasoning.
    """
    root = (summary or {}).get("root", {}) or {}

    payload = {
        "drug": summary.get("drug", ""),
        "target": summary.get("target", ""),
        "fusion_label": root.get("fusion_label", ""),
        "fusion_conf": root.get("fusion_conf", ""),
        "fusion_rule": root.get("fusion_rule", ""),
        "ml_label": root.get("ml_label", ""),
        "kg_label": root.get("kg_label", ""),
        "rag_label": root.get("rag_label", ""),
        "kg_direct": bool(root.get("kg_direct", False)),
        "anchors": {
            "pKd": (root.get("_evidence", {}) or {}).get("pKd", None),
            "pmc_ids": (root.get("_evidence", {}) or {}).get("pmc_ids", []) or [],
            "ml_reason": (root.get("_evidence", {}) or {}).get("ml_reason", ""),
            "kg_reason": (root.get("_evidence", {}) or {}).get("kg_reason", ""),
            "rag_reason": (root.get("_evidence", {}) or {}).get("rag_reason", ""),
        },
    }

    system = """
You write a short addendum to a deterministic rule-based decision.
Hard constraints:
- Do NOT change or contradict fusion_label, fusion_conf, fusion_rule.
- Do NOT argue for a different final label.
- Do NOT introduce new evidence not present in the payload.
- Keep it to 1-2 sentences.
- Focus on non-decision context: e.g., what evidence is missing, what would strengthen confidence, limitations.
Output ONLY plain text (no JSON, no markdown).
""".strip()

    client = _get_explainer_client()
    cfg = _get_config()
    resp = client.chat.completions.create(
        model=cfg["deployment_name"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload)},
        ],
        temperature=0,
        seed=42,
    )
    usage = getattr(resp, "usage", None)

    addendum = (resp.choices[0].message.content or "").strip()
    tok = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "calls": 1,
    }
    return addendum, tok


def apply_final_decision(summary: dict, enabled_agents: List[str]) -> None:
    """Apply deterministic fusion logic to the reasoning tree (supports 1/2/3 sources)."""
    if not isinstance(summary, dict):
        return
    root = summary.get("root", {}) or {}

    # --- extract raw labels/status from summary tree ---
    ml_label_raw, ml_status = _get_source_label_with_status(root, "ML", enabled_agents)
    kg_label_raw, kg_status = _get_source_label_with_status(root, "KG", enabled_agents)
    rag_label_raw, rag_status = _get_source_label_with_status(
        root, "RAG", enabled_agents
    )

    def to_fusion(label: Optional[str], status: str) -> Optional[str]:
        if status == "disabled":
            return None
        return _normalize_fusion_label(label)

    ml_label = to_fusion(ml_label_raw, ml_status)
    kg_label = to_fusion(kg_label_raw, kg_status)
    rag_label = to_fusion(rag_label_raw, rag_status)

    # store per-source labels for logging/explanations
    root["ml_label"] = ml_label or ""
    root["kg_label"] = kg_label or ""
    root["rag_label"] = rag_label or ""

    # record enabled/disabled
    enabled_agents = normalize_enabled_agents(enabled_agents)
    root["enabled_agents"] = enabled_agents
    root["disabled_agents"] = [
        a for a in ALL_EVIDENCE_AGENTS if a not in enabled_agents
    ]

    # build available sources list in canonical order
    src_to_label = {"ML": ml_label, "KG": kg_label, "RAG": rag_label}
    available = [
        (s, src_to_label[s])
        for s in ALL_EVIDENCE_AGENTS
        if s in enabled_agents and src_to_label[s]
    ]

    root["fusion_sources"] = [s for s, _ in available]  # <- helpful for review

    # --- no valid labels ---
    if len(available) == 0:
        root["fusion_label"] = "NA"
        root["fusion_conf"] = "LOW"
        root["fusion_rule"] = "NA"
        root["fusion_reason"] = "No valid input labels."
        summary["root"] = root
        return

    # --- 1-source: pass-through (LOW) ---
    if len(available) == 1:
        s, lbl = available[0]
        root["fusion_label"] = lbl
        root["fusion_conf"] = "LOW"
        root["fusion_rule"] = "1S0"
        root["fusion_reason"] = f"Single-source decision using {s} only (label={lbl})."
        summary["root"] = root
        return

    # --- 2-source rules ---
    if len(available) == 2:
        (a_src, a_lbl), (b_src, b_lbl) = available

        rules2 = make_reviewer_friendly_rules_2src()
        fusion_input2 = {
            "a_src": a_src,
            "b_src": b_src,
            "a": a_lbl,
            "b": b_lbl,
            "kg_direct": bool(root.get("kg_direct", False)),
        }

        fired = None
        for rule in rules2:
            if rule.predicate(fusion_input2):
                fired = rule
                break

        if fired is None:
            root["fusion_label"] = "NA"
            root["fusion_conf"] = "LOW"
            root["fusion_rule"] = "NA"
            root["fusion_reason"] = "No 2-source fusion rule matched."
        else:
            pred = fired.output_label(fusion_input2)
            root["fusion_label"] = pred
            root["fusion_conf"] = fired.confidence
            root["fusion_rule"] = fired.rule_id
            root["fusion_reason"] = (
                f"{fired.description} ({a_src}={a_lbl}, {b_src}={b_lbl}, kg_direct={fusion_input2['kg_direct']})"
            )

        summary["root"] = root
        return

    # --- 3-source existing rules ---
    rules = make_reviewer_friendly_rules_v2()
    fusion_input = {
        "ml": ml_label,
        "kg": kg_label,
        "rag": rag_label,
        "kg_direct": bool(root.get("kg_direct", False)),
    }

    fired = None
    for rule in rules:
        if rule.predicate(fusion_input):
            fired = rule
            break

    if fired is None:
        root["fusion_label"] = "NA"
        root["fusion_conf"] = "LOW"
        root["fusion_rule"] = "NA"
        root["fusion_reason"] = "No fusion rule matched."
    else:
        prediction = fired.output_label(fusion_input)
        reason = (
            f"{fired.description} (ML={fusion_input['ml']}, KG={fusion_input['kg']}, "
            f"RAG={fusion_input['rag']}, kg_direct={fusion_input['kg_direct']})"
        )
        root["fusion_label"] = prediction
        root["fusion_conf"] = fired.confidence
        root["fusion_rule"] = fired.rule_id
        root["fusion_reason"] = reason

    summary["root"] = root


# -----------------------------------------------------------------------------
# CSV utilities
# -----------------------------------------------------------------------------


def ensure_csv_schema(filename: str, fields: List[str]) -> None:
    """Ensure a CSV file matches the desired schema, rewriting if needed."""
    if not os.path.exists(filename):
        return
    with open(filename, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header = rows[0]
    if header == fields:
        return
    normalized_rows = []
    for row in rows[1:]:
        if not row:
            continue
        if len(row) == len(fields):
            row_map = dict(zip(fields, row))
        else:
            row_map = {key: row[i] for i, key in enumerate(header) if i < len(row)}
        normalized_rows.append(row_map)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row_map in normalized_rows:
            writer.writerow({key: row_map.get(key, "") for key in fields})


def sanitize_model_name(name: str) -> str:
    return name.replace(":", "_")


def _normalize_reasoning_token(reasoning_effort: Optional[str]) -> str:
    """Normalize reasoning_effort into a filename-safe token."""
    if reasoning_effort is None:
        return "none"
    token = str(reasoning_effort).strip().lower()
    if token == "":
        return "none"
    # common aliases
    if token in {"h", "high"}:
        return "high"
    if token in {"m", "med", "medium"}:
        return "medium"
    if token in {"l", "low"}:
        return "low"
    # fallback: remove problematic chars
    return re.sub(r"[^0-9a-zA-Z_-]", "_", token)


def _summary_filename_for_config(
    enabled_agents: List[str], model: Optional[str], reasoning_effort: Optional[str]
) -> str:
    enabled = normalize_enabled_agents(enabled_agents)
    config_id = config_id_from_enabled(enabled)
    model_part = "nomodel" if not model else sanitize_model_name(model)
    reason_token = _normalize_reasoning_token(reasoning_effort)
    os.makedirs("output", exist_ok=True)
    return f"output/summary_{config_id}_{model_part}_{reason_token}.csv"


def save_summary_to_csv(
    summary: dict, ablation: str, model: Optional[str], reasoning_effort: Optional[str]
) -> None:
    """Persist summary results to CSV (separate per enabled-agent config and reasoning)."""
    root = summary.get("root", {}) or {}
    enabled_agents = root.get("enabled_agents") or ALL_EVIDENCE_AGENTS
    enabled_agents = normalize_enabled_agents(enabled_agents)

    filename = _summary_filename_for_config(enabled_agents, model, reasoning_effort)

    fields = [
        "drug",
        "target",
        "fusion_label",
        "fusion_conf",
        "fusion_rule",
        "fusion_reason",
        "ml_label",
        "kg_label",
        "rag_label",
        "enabled_agents",
        "config",
        "reasoning_effort",
        "summary_reasoning",
        "reasoning_alignment",
        "token_usage",
        "fusion_sources",
        "input_payload",
    ]

    ensure_csv_schema(filename, fields)

    row = {
        "drug": summary.get("drug", ""),
        "target": summary.get("target", ""),
        "fusion_label": root.get("fusion_label", ""),
        "fusion_conf": root.get("fusion_conf", ""),
        "fusion_rule": root.get("fusion_rule", ""),
        "fusion_reason": root.get("fusion_reason", ""),
        "ml_label": root.get("ml_label", ""),
        "kg_label": root.get("kg_label", ""),
        "rag_label": root.get("rag_label", ""),
        "enabled_agents": ",".join(enabled_agents),
        "config": config_id_from_enabled(enabled_agents),
        "reasoning_effort": reasoning_effort or "",
        "summary_reasoning": root.get("summary_reasoning", ""),
        "fusion_sources": ",".join(root.get("fusion_sources", []) or []),
    }

    # truncate input payload for CSV
    input_payload_obj = summary.get("_input_payload") or {}
    try:
        input_payload_json = json.dumps(input_payload_obj, ensure_ascii=True)
        if len(input_payload_json) > 2000:
            input_payload_json = input_payload_json[:2000] + "...(truncated)"
    except Exception:
        input_payload_json = ""
    row["input_payload"] = input_payload_json

    token_total = root.get("token_usage_total") or {}
    row["token_usage"] = json.dumps(token_total, ensure_ascii=True)

    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0

    # atomic create when file absent
    if not file_exists:
        with tempfile.NamedTemporaryFile(
            "w", delete=False, encoding="utf-8", newline=""
        ) as tmp:
            writer = csv.DictWriter(tmp, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
            tmp_name = tmp.name
        os.replace(tmp_name, filename)
    else:
        # append row (keeping existing header)
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow(row)


def check_already_processed(
    drug: str,
    gene: str,
    ablation: str,
    model: Optional[str],
    enabled_agents: List[str],
    reasoning_effort: Optional[str],
) -> bool:
    """Check per-config+reasoning CSV for an existing processed row."""
    enabled_agents = normalize_enabled_agents(enabled_agents)
    filename = _summary_filename_for_config(enabled_agents, model, reasoning_effort)

    if not os.path.exists(filename):
        return False

    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("drug") != drug:
                continue
            if row.get("target") != gene:
                continue
            if row.get("config") != config_id_from_enabled(enabled_agents):
                continue
            # reasoning_effort is now implied by filename, but keep this row-level check as extra guard
            if (row.get("reasoning_effort") or "") != (reasoning_effort or ""):
                continue
            return True
    return False


# -----------------------------------------------------------------------------
# Autogen agents
# -----------------------------------------------------------------------------


def build_agents(
    model_client, ablation: str = "full", use_planning_agent: bool = False
) -> Dict[str, AssistantAgent]:
    """Build Autogen agents wired to local tool wrappers."""
    enabled_sources = normalize_enabled_agents(_resolve_enabled_sources(ablation))

    agents: Dict[str, AssistantAgent] = {}

    if use_planning_agent:
        planning_agent = AssistantAgent(
            "PlanningAgent",
            description="Delegates tasks to specialist agents. First to act on any new user request.",
            model_client=model_client,
            system_message=_build_planning_system_message(enabled_sources),
        )
        agents["planning_agent"] = planning_agent

    if "RAG" in enabled_sources:
        agents["rag_agent"] = AssistantAgent(
            "RAGAgent",
            description="Searches for relevant evidence from literature.",
            tools=[rag_score],
            model_client=model_client,
            system_message="""
        You are the RAGAgent.

        Role:
        - Use the `rag_score` tool to retrieve literature evidence for drug-target interactions.
        - Make only one RAG call per task.
        - Return raw or summarized information found.
        """,
        )

    if "ML" in enabled_sources:
        agents["ml_agent"] = AssistantAgent(
            "MLAgent",
            description="Performs machine learning-based DTI predictions.",
            model_client=model_client,
            tools=[ml_score],
            system_message="""
          You are the MLAgent.

          Role:
          - Use the `ml_score` tool to predict drug-target interaction (DTI) scores.
          - Make only one prediction call per task.
          - Return the predicted score and reasoning.
          """,
        )

    if "KG" in enabled_sources:
        agents["kg_agent"] = AssistantAgent(
            "KGAgent",
            description="Gathers DTI information using knowledge graph data.",
            model_client=model_client,
            tools=[kg_score],
            system_message="""
You are the KGAgent.

Role:
- Use the `kg_score` tool to predict drug-target interaction (DTI) scores from knowledge graph data.
- Make only one prediction call per task.
- Return the score and any reasoning from the tool.
""",
        )

    summary_agent = AssistantAgent(
        "SummaryAgent",
        description="Synthesizes evidence and produces structured reasoning.",
        model_client=model_client,
        system_message=get_summary_system_message(ablation, enabled_sources),
    )

    agents["summary_agent"] = summary_agent
    return agents


def _resolve_enabled_sources(ablation: str) -> List[str]:
    """Resolve enabled evidence sources from ablation flag."""
    mode = str(ablation).strip().lower()
    sources = ALL_EVIDENCE_AGENTS[:]
    if mode == "minimal":
        return []
    if mode == "no_ml" and "ML" in sources:
        sources.remove("ML")
    if mode == "no_kg" and "KG" in sources:
        sources.remove("KG")
    if mode in {"no_rag"} and "RAG" in sources:
        sources.remove("RAG")
    return sources


def _build_planning_system_message(enabled_sources: List[str]) -> str:
    agent_lines = []
    if "RAG" in enabled_sources:
        agent_lines.append("- RAGAgent: Searches for literature evidence.")
    if "ML" in enabled_sources:
        agent_lines.append("- MLAgent: Performs machine learning predictions.")
    if "KG" in enabled_sources:
        agent_lines.append("- KGAgent: Calculates scores from knowledge graph data.")
    agent_lines.append(
        "- SummaryAgent: Synthesizes findings and produces the final report."
    )
    team_agents = "\n".join(agent_lines)

    return f"""
You are the PlanningAgent.

Role:
- Decompose complex user requests into clear, manageable subtasks.
- Assign those subtasks to appropriate team agents.
- You DO NOT perform any tasks yourself.

Team Agents:
{team_agents}

Execution Rules:
- ALWAYS delegate tasks to ALL relevant agents.
- Use this format for assignments:
    <agent>: <task description>

Workflow:
1. Upon receiving a new task, decompose it and assign subtasks.
2. Wait for all agents to complete their work.
3. Do NOT output the word "TERMINATE". Only SummaryAgent outputs TERMINATE.
"""


def create_model_client(model_type: str, model_name: Optional[str] = None):
    def resolve_model_name(
        preferred_name: Optional[str], deployment: Optional[str]
    ) -> Optional[str]:
        if preferred_name:
            return preferred_name
        return deployment

    def resolve_model_info(model: str) -> Dict[str, Any] | None:
        if not model:
            return None
        try:
            return _model_info.get_info(model)
        except Exception:
            if _model_info._MODEL_INFO:
                return next(iter(_model_info._MODEL_INFO.values()))
            return None

    if model_type == "azure":
        cfg = _get_config()
        model = resolve_model_name(model_name, cfg["deployment_name"])
        client_kwargs = {
            "azure_deployment": cfg["deployment_name"],
            "model": model,
            "api_version": cfg["api_version"],
            "azure_endpoint": cfg["endpoint"],
            "api_key": cfg["api_key"],
            "temperature": 0,
            "seed": 42,
        }
        model_info = resolve_model_info(model)
        if model_info is not None:
            client_kwargs["model_info"] = model_info
        return AzureOpenAIChatCompletionClient(**client_kwargs)
    raise ValueError(f"Unsupported model_type: {model_type}")


# ------
# Fast version
# ------


async def _gather_evidence_parallel(
    drug: str, gene: str, enabled_sources: List[str]
) -> Dict[str, Any]:
    async def run_tool(fn, *args):
        max_attempts = 8
        for attempt in range(1, max_attempts + 1):
            try:
                return await asyncio.to_thread(fn, *args)
            except Exception as exc:
                if _is_rate_limit_exception(exc) and attempt < max_attempts:
                    await _sleep_with_backoff(exc, attempt)
                    continue
                raise

    tasks = {}
    if "ML" in enabled_sources:
        tasks["ml_evidence"] = asyncio.create_task(run_tool(ml_score, drug, gene))
    if "KG" in enabled_sources:
        tasks["kg_evidence"] = asyncio.create_task(run_tool(kg_score, drug, gene))
    if "RAG" in enabled_sources:
        tasks["rag_evidence"] = asyncio.create_task(run_tool(rag_score, drug, gene))

    payload: Dict[str, Any] = {}
    for key, t in tasks.items():
        try:
            res = await t
        except Exception as exc:
            # Tool failure fallback (aligned with _build_evidence_payload behavior).
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

        # Shape payload to match the summary input schema.
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
                "token_usage": res.get("token_usage", {}),  # Added.
            }
        else:
            payload[key] = {
                "drug": res.get("drug"),
                "target": res.get("target"),
                "reason": res.get("reason", ""),
                "label": res.get("label", ""),
                "pmc_ids": res.get("pmc_ids", []) or [],
                "token_usage": res.get("token_usage", {}),  # Added.
            }

    return payload


# -----------------------------------------------------------------------------
# Agent execution flow
# -----------------------------------------------------------------------------


async def chat_with_agents_and_summarize(
    drug: str,
    gene: str,
    verbose: bool = True,
    active_agents=None,
    enabled_agents: Optional[List[str]] = None,
    ablation: str = "full",
    model_client=None,
    model: Optional[str] = None,
    cache_enabled: bool = True,
    reasoning_effort: Optional[str] = None,
    fast_mode: bool = True,
):

    enabled_agents = normalize_enabled_agents(enabled_agents)

    if fast_mode:
        enabled_sources = enabled_agents

        if cache_enabled and check_already_processed(
            drug, gene, ablation, model, enabled_sources, reasoning_effort
        ):
            if verbose:
                print(
                    f"[SKIP] Already processed: {drug}-{gene} for {ablation} ({model})"
                )
            return None

        payload = await _gather_evidence_parallel(drug, gene, enabled_sources)
        summary = await _run_summary_with_evidence(
            payload, ablation, enabled_sources, reasoning_effort=reasoning_effort
        )

        attach_evidence_metadata(summary, payload)
        apply_final_decision(summary, enabled_sources)
        apply_rule_based_explanation(summary)

        # Added: rule-based explanation plus non-conflicting LLM addendum.
        try:
            addendum, add_tok = await _generate_rule_aligned_addendum(summary)
            if addendum:
                root = summary.get("root", {}) or {}
                root["summary_reasoning"] = (
                    root.get("fusion_explanation", "") + " " + addendum
                ).strip()
                root["_token_usage_addendum"] = add_tok
                summary["root"] = root

            attach_evidence_metadata(summary, payload)
        except Exception:
            # Safe to ignore failures; fusion_explanation already exists.
            pass

        if cache_enabled:
            save_summary_to_csv(
                summary,
                ablation=ablation,
                model=model,
                reasoning_effort=reasoning_effort,
            )
        return summary
    else:
        if active_agents is None:
            raise ValueError("active_agents must be provided")
        if model_client is None:
            raise ValueError("model_client must be provided")

    if cache_enabled and check_already_processed(
        drug, gene, ablation, model, enabled_agents, reasoning_effort
    ):
        if verbose:
            print(f"[SKIP] Already processed: {drug}-{gene} for {ablation} ({model})")
        return None

    team = SelectorGroupChat(
        active_agents,
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=(
            selector_prompt_with_planner
            if any(getattr(a, "name", "") == "PlanningAgent" for a in active_agents)
            else selector_prompt_no_planner
        ),
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

    # ---- retry wrapper for team.run_stream (handles Azure 429) ----
    max_attempts = 8
    attempt = 0

    while True:
        attempt += 1

        # Clear transient state on retries to avoid mixing partial messages.
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

            # If we reached here, the run completed successfully.
            break

        except Exception as exc:
            # Retry only for 429-like errors; otherwise raise immediately.
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
        payload = _build_evidence_payload(drug, gene, enabled_sources)
        try:
            summary = await _run_summary_with_evidence(
                payload,
                ablation,
                enabled_sources,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:
            if verbose:
                print(f"[ERROR] Summary rebuild failed: {exc}")
            return None
    else:
        payload = _build_evidence_payload(drug, gene, enabled_sources)

    # --- Serialize conversation messages into a JSON-serializable form and attach for auditability ---
    try:
        serialized_messages = []
        for m in full_conversation_messages:
            # BaseChatMessage like objects: keep source and content; include role if available
            serialized_messages.append(
                {
                    "source": getattr(m, "source", None),
                    "role": getattr(m, "role", None),
                    "content": getattr(m, "content", ""),
                }
            )
        # If summary exists, prefer adding to it; otherwise attach to the payload so downstream save sees it.
        if isinstance(summary, dict):
            # do not overwrite if already present
            if "_input_messages" not in summary:
                summary["_input_messages"] = serialized_messages
        else:
            # ensure payload also carries messages for the _run_summary_with_evidence path
            payload["_input_messages"] = serialized_messages
    except Exception:
        # best-effort: failures here must not break main flow
        pass

    # Added: inject evidence metadata into root.
    attach_evidence_metadata(summary, payload)

    # Existing: fusion.
    apply_final_decision(summary, enabled_sources)

    # Added: rule-aligned explanation.
    apply_rule_based_explanation(summary)

    # Added: rule-based explanation plus non-conflicting LLM addendum.
    try:
        addendum, add_tok = await _generate_rule_aligned_addendum(summary)
        if addendum:
            root = summary.get("root", {}) or {}
            root["summary_reasoning"] = (
                root.get("fusion_explanation", "") + " " + addendum
            ).strip()
            root["_token_usage_addendum"] = add_tok
            summary["root"] = root

        attach_evidence_metadata(summary, payload)
    except Exception:
        # Safe to ignore failures; fusion_explanation already exists.
        pass

    if verbose:
        root = summary.get("root", {})
        print("--------------------------------")
        print(f"[FUSION] {root.get('fusion_label', '')}")
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
        )
    return summary


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------


def get_active_agents(
    ablation: str = "full",
    model_client=None,
    enabled_agents: List[str] | None = None,
    return_enabled: bool = False,
    use_planning_agent: bool = False,  # Added.
):
    agents = build_agents(
        model_client, ablation=ablation, use_planning_agent=use_planning_agent
    )
    planning_agent = agents.get("planning_agent")
    rag_agent = agents.get("rag_agent")
    ml_agent = agents.get("ml_agent")
    kg_agent = agents.get("kg_agent")
    summary_agent = agents["summary_agent"]

    if enabled_agents is None:
        enabled_agents = ALL_EVIDENCE_AGENTS[:]

    mode = str(ablation).strip().lower()
    if mode == "no_rag":
        enabled_agents = [a for a in enabled_agents if a != "RAG"]
    elif mode == "no_ml":
        enabled_agents = [a for a in enabled_agents if a != "ML"]
    elif mode == "no_kg":
        enabled_agents = [a for a in enabled_agents if a != "KG"]
    elif mode == "full":
        pass
    else:
        raise ValueError(f"Unknown ablation type: {ablation}")

    enabled_agents = normalize_enabled_agents(enabled_agents)
    agent_map = {"RAG": rag_agent, "ML": ml_agent, "KG": kg_agent}
    participants = [
        *([planning_agent] if planning_agent is not None else []),
        *[agent_map[a] for a in enabled_agents if agent_map.get(a)],
        summary_agent,
    ]

    if return_enabled:
        return participants, enabled_agents
    return participants


def main() -> None:
    parser = argparse.ArgumentParser(description="Drug-Target Interaction CLI Agent")
    parser.add_argument(
        "--drug", type=str, required=True, help="Drug name (e.g., Gefitinib)"
    )
    parser.add_argument(
        "--gene", type=str, required=True, help="Target gene name (e.g., TOP1)"
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        help="Ablation type (e.g., full, no_ml, no_kg, no_rag)",
    )
    parser.add_argument(
        "--enabled_agents",
        type=str,
        default=None,
        help="Comma-separated list of enabled agents (ML,KG,RAG).",
    )
    parser.add_argument(
        "--run_all_ablations",
        action="store_true",
        help="Run all non-empty enabled agent combinations.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="azure",
        choices=["azure"],
        help="Which LLM provider to use",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name to use for the selected LLM provider",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        help="Override reasoning effort (low, medium, high, none).",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable cache usage (skip CSV cache checks and writes).",
    )
    parser.add_argument(
        "--use_planning_agent",
        action="store_true",
        help="Enable PlanningAgent in the multi-agent orchestration (default: disabled).",
    )

    parser.add_argument(
        "--no_fast_mode",
        action="store_true",
        help="Disable fast mode; use SelectorGroupChat orchestration.",
    )

    args = parser.parse_args()
    model_client = create_model_client(args.model_type, args.model_name)
    enabled_agents = None
    if args.enabled_agents:
        enabled_agents = [
            part.strip() for part in args.enabled_agents.split(",") if part.strip()
        ]

    reasoning_effort = args.reasoning_effort
    if reasoning_effort and reasoning_effort.strip().lower() == "none":
        reasoning_effort = None

    if args.run_all_ablations:
        for combo in generate_ablation_configs():
            try:
                active_agents, normalized_agents = get_active_agents(
                    ablation=args.ablation,
                    model_client=model_client,
                    enabled_agents=combo,
                    return_enabled=True,
                    use_planning_agent=args.use_planning_agent,
                )
            except ValueError as exc:
                print(str(exc))
                exit(1)

            if not active_agents:
                print("Error: No agents enabled after ablation.")
                exit(1)

            asyncio.run(
                chat_with_agents_and_summarize(
                    args.drug,
                    args.gene,
                    active_agents=active_agents,
                    enabled_agents=normalized_agents,
                    ablation=args.ablation,
                    model_client=model_client,
                    model=None,
                    cache_enabled=not args.no_cache,
                    reasoning_effort=reasoning_effort,
                    fast_mode=(not args.no_fast_mode),
                )
            )
        return

    try:
        active_agents, normalized_agents = get_active_agents(
            ablation=args.ablation,
            model_client=model_client,
            enabled_agents=enabled_agents,  # Not a combo in this path.
            return_enabled=True,
            use_planning_agent=args.use_planning_agent,
        )
    except ValueError as exc:
        print(str(exc))
        exit(1)

    if not active_agents:
        print("Error: No agents enabled after ablation.")
        exit(1)

    asyncio.run(
        chat_with_agents_and_summarize(
            args.drug,
            args.gene,
            active_agents=active_agents,
            enabled_agents=normalized_agents,  # Pass normalized_agents.
            ablation=args.ablation,
            model_client=model_client,
            model=None,
            cache_enabled=not args.no_cache,
            reasoning_effort=reasoning_effort,
            fast_mode=(not args.no_fast_mode),
        )
    )


if __name__ == "__main__":
    main()
