#!/usr/bin/env python
# coding: utf-8
"""Summary generation and decision logic for DrugAgent."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai import AzureOpenAI

from drugagent import config as dag_config
from drugagent.config import ALL_EVIDENCE_AGENTS, _get_config, _get_reasoning_settings
from drugagent.utils import (
    _is_rate_limit_exception,
    _normalize_fusion_label,
    _normalize_label,
    _sleep_with_backoff,
    normalize_enabled_agents,
)


@dataclass
class TokenAgg:
    """Aggregate token usage across multiple calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add(self, name: str, usage: Any) -> None:
        """Add a token usage object.

        Args:
            name: Component name.
            usage: Usage object with token attributes.
        """
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
        """Add token usage from a dict.

        Args:
            name: Component name.
            d: Dict with token usage fields.
        """
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






def _is_content_filter_error(exc: Exception) -> bool:
    """Return True if exception looks like a content filter error.

    Args:
        exc: Exception instance.

    Returns:
        True if content filter error is detected.
    """
    msg = str(exc).lower()
    return "content_filter" in msg or "responsibleaipolicyviolation" in msg


def _minimal_payload_for_llm(payload: Any) -> Any:
    """Strip long/free-text fields to avoid content filters.

    Args:
        payload: Any JSON-serializable object.

    Returns:
        Minimized payload with reasons redacted.
    """
    if isinstance(payload, dict):
        out = {}
        for k, v in payload.items():
            if k in {"reason", "observation", "summary_reasoning"}:
                out[k] = "[REDACTED]"
            else:
                out[k] = _minimal_payload_for_llm(v)
        return out
    if isinstance(payload, list):
        return [_minimal_payload_for_llm(v) for v in payload]
    return payload


SELF_HARM_PATTERNS = [
    r"suicid\w*",
    r"self[- ]?harm",
    r"self[- ]?injur\w*",
    r"self[- ]?mutilat\w*",
    r"kill myself",
    r"kill herself",
    r"kill himself",
    r"attempted suicide",
    r"overdose",
    r"cutting",
]

def _redact_self_harm(text: str) -> str:
    """Redact self-harm related phrases to avoid content filters.

    Args:
        text: Input string.

    Returns:
        Redacted string.
    """
    if not text:
        return text
    out = text
    for pat in SELF_HARM_PATTERNS:
        out = re.sub(pat, "[REDACTED]", out, flags=re.IGNORECASE)
    return out


def _sanitize_payload_for_llm(payload: Any) -> Any:
    """Recursively redact self-harm phrases in payload strings.

    Args:
        payload: Any JSON-serializable object.

    Returns:
        Sanitized payload.
    """
    if isinstance(payload, dict):
        return {k: _sanitize_payload_for_llm(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_sanitize_payload_for_llm(v) for v in payload]
    if isinstance(payload, str):
        return _redact_self_harm(payload)
    return payload


# -----------------------------------------------------------------------------
# Summary agent system message
# -----------------------------------------------------------------------------


def get_summary_system_message(
    ablation: str,
    enabled_sources: List[str],
    *,
    binary_mode: Optional[bool] = None,
) -> str:
    """Build summary system message for enabled sources.

    Args:
        ablation: Ablation mode.
        enabled_sources: Evidence sources to include.
        binary_mode: Optional override for binary label mode.

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

    use_binary = False
    label_hint = "STRONG/MODERATE/WEAK/NONE"

    header = f"""Task:
Synthesize evidence from the provided sources about a drug-target interaction.

{sources_text}

Primary objective: produce a compact, biologically-plausible reasoning tree that captures each source's stance and
the overall agreement/uncertainty. Do NOT make the final fusion decision.

Your role (ordered):
1. For each available source, write a single evidence_analysis node with a brief observation.
2. Use the provided label if present; otherwise infer a concise label from evidence ({label_hint}).
3. Summarize cross-source agreement/conflict and key uncertainty in summary_reasoning.

Coherence rules:
1. Use a consistent, neutral scientific tone.
2. Use the same evidence order as enabled_sources: ML -> KG -> RAG.
3. In each observation, use this fixed structure:
   - Claim: <one sentence describing the source's conclusion>.
   - Evidence: <one sentence citing the strongest support from that source>.
   - Caveat: <one short sentence noting uncertainty or limitation>.
4. In summary_reasoning, write 2-3 sentences in this order:
   - Agreement on convergence/divergence across sources.
   - Strongest support and why.
   - Key uncertainty or gap.
   Do not prefix sentences with labels like "Agreement:".
5. Biological plausibility checks:
   - Prioritize describing observed functional/phenotypic effects when binding evidence is weak or absent.
   - Only emphasize direct binding when there is explicit biochemical or target-engagement evidence.
   - Name the biological level of the strongest evidence (biochemical, cellular, in vivo, clinical).
   - Provide one plausible alternative explanation for the association when evidence is indirect.
"""

    body = f"""Input:
A JSON payload that may contain keys:
{input_format}

Output format (MUST follow exactly):
- Output ONLY a single valid JSON object that matches the schema below.
- After the JSON object, output the word `TERMINATE` on its own line.

Schema (children in "root.children" correspond to available sources only):
{schema_text}

Requirements:
- For each evidence_analysis node:
  - If the source provided a label, include it verbatim in "label".
  - If no label is provided, infer a concise label (ACTIVE/INACTIVE in binary mode; STRONG/MODERATE/WEAK/NONE otherwise) and justify it in "observation".
- The "summary_reasoning" must:
  - State whether evidence indicates direct binding, functional/phenotypic association, or insufficient evidence.
  - Give the top 1-2 reasons and the largest uncertainty.
- Do NOT invent new primary-evidence quotes or numbers.
- If a source is missing from input, omit that source's child entirely.

Style and constraints:
- Be concise and conservative about uncertainty.
- Output must be machine-parseable JSON matching the schema above exactly (double quotes, valid types).
- No markdown or extra commentary. ONLY the JSON and the final `TERMINATE` token.

Failure mode:
- If you cannot parse any evidence, return the schema with empty children and "summary_reasoning" set to "No parsable evidence".

End.
"""

    return header + body


# -----------------------------------------------------------------------------
# Evidence payload and summary parsing
# -----------------------------------------------------------------------------


def _extract_json_block(text: str) -> Dict[str, Any]:
    """Extract a JSON object from a possibly noisy model response.

    Args:
        text: Raw model output text.

    Returns:
        Parsed JSON object.
    """
    content = (text or "").strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    content = re.sub(r"TERMINATE", "", content).strip()

    start = content.find("{")
    if start == -1:
        raise ValueError("No JSON object start '{' found in model output.")
    content2 = content[start:].lstrip()

    decoder = json.JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(content2)
        return obj
    except json.JSONDecodeError:
        end_brace = content2.rfind("}")
        if end_brace != -1:
            return json.loads(content2[: end_brace + 1])
        raise ValueError("Failed to parse summary JSON from model output.")


_summary_clients: Dict[Optional[str], AzureOpenAI] = {}


def _make_azure_client_with_reasoning(reasoning_effort: Optional[str]) -> AzureOpenAI:
    """Create a basic AzureOpenAI client for summary calls.

    Args:
        reasoning_effort: Requested reasoning effort.

    Returns:
        AzureOpenAI client.
    """
    cfg = _get_config()
    return AzureOpenAI(
        api_key=cfg["api_key"],
        azure_endpoint=cfg["endpoint"],
        api_version=cfg["api_version"],
    )


def _get_summary_client(reasoning_effort: Optional[str] = None) -> AzureOpenAI:
    """Create or reuse a summary LLM client keyed by reasoning_effort.

    Args:
        reasoning_effort: None | "low" | "medium" | "high".

    Returns:
        AzureOpenAI client.
    """
    key = reasoning_effort if reasoning_effort is not None else "__NONE__"
    client = _summary_clients.get(key)
    if client is None:
        client = _make_azure_client_with_reasoning(reasoning_effort)
        _summary_clients[key] = client
    return client




def _build_summary_messages(system_message: str, payload: Any, use_minimal: bool) -> list[dict]:
    """Build messages for summary LLM calls.

    Args:
        system_message: System prompt.
        payload: Evidence payload.
        use_minimal: Whether to minimize payload.

    Returns:
        Messages list for LLM.
    """
    safe_payload = _sanitize_payload_for_llm(payload)
    if use_minimal:
        safe_payload = _minimal_payload_for_llm(safe_payload)
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": json.dumps(safe_payload)},
    ]


async def _run_summary_with_evidence(
    payload: Dict[str, Any],
    ablation: str,
    enabled_sources: List[str],
    reasoning_effort: Optional[str] = None,
    *,
    binary_mode: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run SummaryAgent deterministically with explicit evidence payload.

    Args:
        payload: Evidence payload for the summary agent.
        ablation: Ablation mode.
        enabled_sources: Enabled evidence sources.
        reasoning_effort: Optional reasoning effort override.
        binary_mode: Optional override for Active/Inactive mode.

    Returns:
        Parsed summary JSON.
    """
    if reasoning_effort is not None:
        reasoning_effort = str(reasoning_effort).strip()
        if reasoning_effort == "":
            reasoning_effort = None
        else:
            if reasoning_effort.lower() == "none":
                reasoning_effort = None
            else:
                reasoning_effort = reasoning_effort.lower()

    cfg = _get_config()
    reasoning_settings = _get_reasoning_settings()
    system_message = get_summary_system_message(
        ablation=ablation, enabled_sources=enabled_sources, binary_mode=binary_mode
    )

    client = _get_summary_client(reasoning_effort)

    request_kwargs = {"temperature": 0, "seed": 42}
    if reasoning_effort is not None:
        request_kwargs["reasoning"] = {"effort": reasoning_effort}
    elif reasoning_settings is not None:
        request_kwargs["reasoning"] = reasoning_settings

    max_attempts = 8
    response = None
    messages = []
    use_minimal_payload = False
    for attempt in range(1, max_attempts + 1):
        try:
            messages = _build_summary_messages(system_message, payload, use_minimal_payload)

            def _call_llm():
                try:
                    return client.responses.create(
                        model=cfg["deployment_name"],
                        messages=messages,
                        timeout=120,
                        **request_kwargs,
                    )
                except TypeError:
                    pass
                except Exception:
                    raise

                try:
                    return client.responses.create(
                        deployment_id=cfg["deployment_name"],
                        input=messages,
                        timeout=120,
                        **request_kwargs,
                    )
                except TypeError:
                    pass
                except Exception:
                    raise

                chat_kwargs = dict(request_kwargs)
                if "reasoning" in chat_kwargs:
                    chat_kwargs.pop("reasoning")

                return client.chat.completions.create(
                    model=cfg["deployment_name"],
                    messages=messages,
                    timeout=120,
                    **chat_kwargs,
                )

            response = await asyncio.to_thread(_call_llm)
            break
        except Exception as exc:
            if _is_content_filter_error(exc) and not use_minimal_payload:
                use_minimal_payload = True
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": json.dumps(_minimal_payload_for_llm(_sanitize_payload_for_llm(payload)))},
                ]
                continue
            if _is_rate_limit_exception(exc) and attempt < max_attempts:
                await _sleep_with_backoff(exc, attempt)
                continue
            raise

    if response is None:
        raise RuntimeError("No response received from LLM after retries.")

    content = ""
    try:
        if hasattr(response, "choices") and response.choices:
            choice0 = response.choices[0]
            if hasattr(choice0, "message") and getattr(choice0.message, "content", None) is not None:
                content = choice0.message.content or ""
            else:
                try:
                    content = choice0["message"]["content"]
                except Exception:
                    content = ""
        else:
            out = getattr(response, "output", None) or getattr(response, "outputs", None)
            if out and isinstance(out, (list, tuple)) and len(out) > 0:
                first = out[0]
                if isinstance(first, dict):
                    cont = first.get("content") or first.get("body") or first.get("text")
                    if isinstance(cont, list) and cont:
                        if isinstance(cont[0], dict):
                            text = cont[0].get("text") or cont[0].get("value") or cont[0].get("content")
                            if text:
                                content = text
                        elif isinstance(cont[0], str):
                            content = cont[0]
                    elif isinstance(cont, str):
                        content = cont
                else:
                    text_attr = getattr(first, "text", None) or getattr(first, "content", None)
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

    obj = _extract_json_block(content)

    try:
        obj["_input_payload"] = payload
        try:
            obj["_input_messages"] = messages
        except Exception:
            obj["_input_messages"] = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": json.dumps(_sanitize_payload_for_llm(payload))},
            ]
    except Exception:
        pass

    if not isinstance(obj, dict):
        obj = {"root": {}}
    if "root" not in obj or not isinstance(obj["root"], dict):
        obj["root"] = {}

    obj["root"]["reasoning_effort"] = reasoning_effort or ""
    obj["root"]["requested_reasoning"] = {
        "effort_param": reasoning_effort or None,
        "used_defaults": (reasoning_effort is None and reasoning_settings is not None),
    }

    usage = getattr(response, "usage", None)
    if usage is None:
        usage = getattr(response, "meta", None) or getattr(response, "output_meta", None) or getattr(response, "raw", None)

    obj["_token_usage_summary"] = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "calls": 1,
    }
    return obj


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
        if status == "present":
            return None, "missing"
        return None, status
    return None, "missing"


def _extract_label_from_text(text: str) -> str:
    """Extract label from free-text observation.

    Args:
        text: Observation text.

    Returns:
        Extracted label string.
    """
    match = re.search(
        r"Label\s*[:=]\s*([A-Za-z][A-Za-z\s-]+)",
        text or "",
    )
    if not match:
        return ""
    value = match.group(1).strip()
    value = re.split(r"[\.;,\)\]]", value, maxsplit=1)[0].strip()
    return value


def attach_evidence_metadata(summary: dict, payload: dict) -> None:
    """Attach tool-derived evidence metadata into summary.root.

    Args:
        summary: Summary JSON object.
        payload: Evidence payload.
    """
    if not isinstance(summary, dict):
        return
    root = summary.setdefault("root", {})
    root["_evidence"] = root.get("_evidence", {})

    ml = payload.get("ml_evidence") or {}
    kg = payload.get("kg_evidence") or {}
    rag = payload.get("rag_evidence") or {}

    root["kg_direct"] = bool(kg.get("kg_direct", False))
    root["_evidence"]["ml_reason"] = str(ml.get("reason", ""))[:2000]
    root["_evidence"]["kg_reason"] = str(kg.get("reason", ""))[:2000]
    root["_evidence"]["rag_reason"] = str(rag.get("reason", ""))[:4000]

    root["_evidence"]["pKd"] = ml.get("pKd", None)
    root["_evidence"]["pmc_ids"] = rag.get("pmc_ids", []) or []

    agg = TokenAgg()
    agg.add_dict("kg", (kg.get("token_usage") or {}))
    agg.add_dict("rag", (rag.get("token_usage") or {}))
    agg.add_dict("ml", (ml.get("token_usage") or {}))
    agg.add_dict("summary", (summary.get("_token_usage_summary") or {}))
    agg.add_dict("decision", (root.get("_token_usage_decision") or {}))

    root["token_usage_total"] = {
        "prompt_tokens": agg.prompt_tokens,
        "completion_tokens": agg.completion_tokens,
        "total_tokens": agg.total_tokens,
        "calls": agg.calls,
        "breakdown": agg.breakdown,
    }


async def _run_llm_final_decision(
    payload: Dict[str, Any],
    reasoning_effort: Optional[str] = None,
    *,
    binary_mode: Optional[bool] = None,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Run an LLM-only decision step that outputs the final label.

    Args:
        payload: Decision payload.
        reasoning_effort: Optional reasoning effort override.
        binary_mode: Optional override for Active/Inactive mode.

    Returns:
        Tuple of (decision_json, token_usage).
    """
    if reasoning_effort is not None:
        reasoning_effort = str(reasoning_effort).strip()
        if reasoning_effort == "":
            reasoning_effort = None
        elif reasoning_effort.lower() == "none":
            reasoning_effort = None
        else:
            reasoning_effort = reasoning_effort.lower()

    cfg = _get_config()
    reasoning_settings = _get_reasoning_settings()
    client = _get_summary_client(reasoning_effort)

    request_kwargs = {"temperature": 0, "seed": 42}
    if reasoning_effort is not None:
        request_kwargs["reasoning"] = {"effort": reasoning_effort}
    elif reasoning_settings is not None:
        request_kwargs["reasoning"] = reasoning_settings

    binary_system = """You are DecisionAgent. Input: the SummaryAgent output (the reasoning_tree JSON) and helper fields in the payload (source_labels/source_statuses/evidence_flags). Your task: produce a single JSON object with the final fusion label and a transparent, rules-based justification. Do NOT invent new primary-evidence quotes -- only use the reasoning_tree and the chain_of_evidence provided.

Allowed final labels: "Active", "Inactive".
Also provide a confidence: one of "LOW","MEDIUM","HIGH".

Rules (apply deterministically in this priority order):
1. Use only PRESENT sources. Ignore sources that are missing or disabled (see source_statuses if provided).
2. Majority vote among present sources: if two or more present sources agree -> choose that label, confidence HIGH.
3. If no majority:
   - If any present source label is Active -> choose Active, confidence MEDIUM.
   - Otherwise choose Inactive, confidence LOW.
4. If only one present source -> adopt that label, confidence MEDIUM.
5. If a label is Active but evidence_flags indicate low confidence (e.g., rag_high_conf_active false), keep the label but lower confidence or explain uncertainty; do not auto-flip to Inactive.

Output schema (exact):
{
  "fusion_label": "Active|Inactive",
  "fusion_conf": "LOW|MEDIUM|HIGH",
  "fusion_reason": "string (detailed sentence(s) citing which rule applied and mapping to evidence)",
  "recommended_next_experiments": ["e.g., 'Perform orthogonal binding assay (SPR) to confirm KD', 'Cellular functional assay in relevant cell line'"],
  "timestamp": "ISO8601 string"
}

Operational constraints:
- Temperature: 0. Seed: 42.""".strip()

    multi_system = """You are DecisionAgent. Input: the SummaryAgent output (the reasoning_tree JSON) and helper fields in the payload (source_labels/source_statuses/evidence_flags). Your task: produce a single JSON object with the final fusion label and a transparent, rules-based justification. Do NOT invent new primary-evidence quotes -- only use the reasoning_tree and the chain_of_evidence provided.

Allowed final labels: "Low", "Moderate", "Strong".
Also provide a confidence: one of "LOW","MEDIUM","HIGH".

Rules (apply deterministically in this priority order):
1. Use only PRESENT sources. Ignore sources that are missing or disabled (see source_statuses if provided).
2. Majority vote among present sources: if two or more present sources agree -> choose that label, confidence HIGH.
3. If no majority, apply biochemical tie-break:
   a) If KG reports a direct 1-hop edge (kg_direct true) AND ml_evidence pKd <= 7 (or ML label Strong) -> choose Strong, confidence MEDIUM.
   b) If RAG contains at least one binding assay with explicit quoted dose-response or KD/Ki numeric value consistent with binding (e.g., pKd/pKi <= threshold), prefer that source (confidence MEDIUM-HIGH depending on orthogonality).
4. If only one present source -> adopt that label, confidence MEDIUM (unless ML presents no quantitative pKd -> LOW).
5. If evidence_flags indicate low confidence for a present label, keep the label but lower confidence or explain uncertainty; do not auto-flip.

Output schema (exact):
{
  "fusion_label": "Low|Moderate|Strong",
  "fusion_conf": "LOW|MEDIUM|HIGH",
  "fusion_reason": "string (detailed sentence(s) citing which rule applied and mapping to evidence: e.g., 'Majority vote: ML+KG -> Strong; KG direct path; RAG supports with BLI KD 45 nM (PMC...)')",
  "recommended_next_experiments": ["e.g., 'Perform orthogonal binding assay (SPR) to confirm KD', 'Cellular functional assay in relevant cell line'"],
  "timestamp": "ISO8601 string"
}

Operational constraints:
- Temperature: 0. Seed: 42.
- If you use any numeric threshold (pKd), state it explicitly in fusion_reason.""".strip()

    use_binary = dag_config.BINARY_MODE if binary_mode is None else bool(binary_mode)
    system = binary_system if use_binary else multi_system

    use_minimal_payload = False
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(_sanitize_payload_for_llm(payload))},
    ]

    max_attempts = 8
    response = None
    for attempt in range(1, max_attempts + 1):
        try:
            def _call_llm():
                try:
                    return client.responses.create(
                        model=cfg["deployment_name"],
                        messages=messages,
                        **request_kwargs,
                    )
                except TypeError:
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

                chat_kwargs = dict(request_kwargs)
                chat_kwargs.pop("reasoning", None)
                return client.chat.completions.create(
                    model=cfg["deployment_name"],
                    messages=messages,
                    **chat_kwargs,
                )

            response = await asyncio.to_thread(_call_llm)
            break
        except Exception as exc:
            if _is_content_filter_error(exc) and not use_minimal_payload:
                use_minimal_payload = True
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(_minimal_payload_for_llm(_sanitize_payload_for_llm(payload)))},
                ]
                continue
            if _is_rate_limit_exception(exc) and attempt < max_attempts:
                await _sleep_with_backoff(exc, attempt)
                continue
            raise

    if response is None:
        raise RuntimeError("No response received from LLM after retries.")

    content = ""
    try:
        if hasattr(response, "choices") and response.choices:
            choice0 = response.choices[0]
            if hasattr(choice0, "message") and getattr(choice0.message, "content", None) is not None:
                content = choice0.message.content or ""
            else:
                try:
                    content = choice0["message"]["content"]
                except Exception:
                    content = ""
        else:
            out = getattr(response, "output", None) or getattr(response, "outputs", None)
            if out and isinstance(out, (list, tuple)) and len(out) > 0:
                first = out[0]
                if isinstance(first, dict):
                    cont = first.get("content") or first.get("body") or first.get("text")
                    if isinstance(cont, list) and cont:
                        if isinstance(cont[0], dict):
                            text = cont[0].get("text") or cont[0].get("value") or cont[0].get("content")
                            if text:
                                content = text
                        elif isinstance(cont[0], str):
                            content = cont[0]
                    elif isinstance(cont, str):
                        content = cont
                else:
                    text_attr = getattr(first, "text", None) or getattr(first, "content", None)
                    if isinstance(text_attr, str):
                        content = text_attr
            if not content and hasattr(response, "output_text"):
                content = getattr(response, "output_text") or ""
            if not content and hasattr(response, "text"):
                content = getattr(response, "text") or ""
    except Exception:
        content = ""

    if not content:
        content = str(response)

    obj = _extract_json_block(content)
    usage = getattr(response, "usage", None)
    tok = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "calls": 1,
    }
    return obj, tok




def _contains_quant_assay(text: str) -> bool:
    """Return True if text mentions numeric assay evidence.

    Args:
        text: Source text.

    Returns:
        True if numeric assay evidence is detected.
    """
    if not text:
        return False
    assay_terms = ["kd", "ki", "ic50", "ec50", "affinity", "potency", "spr", "bli"]
    has_assay = any(term in text.lower() for term in assay_terms)
    has_number = re.search(r"\b\d+(?:\.\d+)?\s*(?:nm|um|\u00b5m|pm|mm)\b", text.lower())
    return bool(has_assay and has_number)


def _rag_is_high_confidence_active(reason: str) -> bool:
    """Check if RAG evidence meets a strict Active criterion.

    Args:
        reason: JSON string from rag_evidence.reason.

    Returns:
        True if RAG evidence shows direct DTI with quantitative assay evidence.
    """
    try:
        obj = json.loads(reason) if isinstance(reason, str) else {}
    except Exception:
        return False
    stage1 = obj.get("stage1") or {}
    pair_evidence = stage1.get("pair_evidence") or {}
    if not pair_evidence.get("direct_dti_in_same_pmcid"):
        return False
    if not pair_evidence.get("mentions_both_explicitly_in_same_pmcid"):
        return False
    quotes = []
    evidence_quotes = obj.get("evidence_quotes") or {}
    quotes.extend(evidence_quotes.get("pair_side") or [])
    quotes.extend((stage1.get("pair_evidence") or {}).get("pair_quotes") or [])
    joined = " ".join(str(q) for q in quotes if q)
    return _contains_quant_assay(joined)


def _kg_is_high_confidence_active(kg_direct: bool, reason: str) -> bool:
    """Check if KG evidence meets a strict Active criterion.

    Args:
        kg_direct: Whether KG has direct 1-hop evidence.
        reason: JSON string from kg_evidence.reason.

    Returns:
        True if KG evidence is direct and includes quantitative assay evidence.
    """
    if not kg_direct:
        return False
    text = reason if isinstance(reason, str) else ""
    return _contains_quant_assay(text)


async def apply_final_decision(
    summary: dict,
    enabled_agents: List[str],
    payload: Dict[str, Any],
    reasoning_effort: Optional[str] = None,
    *,
    binary_mode: Optional[bool] = None,
) -> None:
    """Apply LLM-only decision logic to the reasoning tree.

    Args:
        summary: Summary JSON.
        enabled_agents: Enabled evidence sources.
        payload: Evidence payload.
        reasoning_effort: Optional reasoning effort override.
        binary_mode: Optional override for Active/Inactive mode.
    """
    if not isinstance(summary, dict):
        return
    root = summary.get("root", {}) or {}

    ml = payload.get("ml_evidence") or {}
    kg = payload.get("kg_evidence") or {}
    rag = payload.get("rag_evidence") or {}


    ml_label_raw, ml_status = _get_source_label_with_status(root, "ML", enabled_agents)
    use_binary = dag_config.BINARY_MODE if binary_mode is None else bool(binary_mode)
    kg_label_raw, kg_status = _get_source_label_with_status(root, "KG", enabled_agents)
    rag_label_raw, rag_status = _get_source_label_with_status(root, "RAG", enabled_agents)

    def to_fusion(label: Optional[str], status: str) -> Optional[str]:
        if status != "present":
            return None
        return _normalize_fusion_label(label)

    ml_label = to_fusion(ml_label_raw, ml_status)
    kg_label = to_fusion(kg_label_raw, kg_status)
    rag_label = to_fusion(rag_label_raw, rag_status)

    root["ml_label"] = ml_label or ""
    root["kg_label"] = kg_label or ""
    root["rag_label"] = rag_label or ""

    rag_high_conf_active = False
    kg_high_conf_active = False
    if use_binary:
        rag_high_conf_active = _rag_is_high_confidence_active(rag.get("reason", ""))
        kg_high_conf_active = _kg_is_high_confidence_active(bool(kg.get("kg_direct", False)), kg.get("reason", ""))

    root["rag_high_conf_active"] = rag_high_conf_active
    root["kg_high_conf_active"] = kg_high_conf_active

    enabled_agents = normalize_enabled_agents(enabled_agents)
    root["enabled_agents"] = enabled_agents
    root["disabled_agents"] = [a for a in ALL_EVIDENCE_AGENTS if a not in enabled_agents]

    src_to_label = {"ML": ml_label, "KG": kg_label, "RAG": rag_label}
    available = [(s, src_to_label[s]) for s in ALL_EVIDENCE_AGENTS if s in enabled_agents and src_to_label[s]]
    root["fusion_sources"] = [s for s, _ in available]

    decision_payload = {
        "drug": summary.get("drug", ""),
        "target": summary.get("target", ""),
        "enabled_sources": enabled_agents,
        "summary_tree": summary.get("root", {}) or {},
        "evidence_payload": payload,
        "source_labels": {
            "ML": ml_label or "",
            "KG": kg_label or "",
            "RAG": rag_label or "",
        },
        "source_statuses": {
            "ML": ml_status,
            "KG": kg_status,
            "RAG": rag_status,
        },
        "evidence_flags": {
            "rag_high_conf_active": rag_high_conf_active,
            "kg_high_conf_active": kg_high_conf_active,
        },
    }

    try:
        decision, tok = await _run_llm_final_decision(
            decision_payload, reasoning_effort=reasoning_effort, binary_mode=binary_mode
        )
    except Exception as exc:
        root["fusion_label"] = "NA"
        root["fusion_conf"] = "LOW"
        root["fusion_rule"] = "LLM"
        root["fusion_reason"] = f"LLM decision failed: {exc}"
        root["_token_usage_decision"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }
        summary["root"] = root
        return

    label_raw = decision.get("fusion_label") or decision.get("label") or ""
    label = _normalize_fusion_label(label_raw) or "NA"

    conf_raw = str(decision.get("fusion_conf", "") or decision.get("confidence", "")).strip().upper()
    conf = conf_raw if conf_raw in {"LOW", "MEDIUM", "HIGH"} else "LOW"
    reason = str(decision.get("fusion_reason", "") or decision.get("reason", "")).strip()

    root["fusion_label"] = label
    root["fusion_conf"] = conf
    root["fusion_rule"] = "LLM"
    root["fusion_reason"] = reason
    root["_token_usage_decision"] = tok
    summary["root"] = root


def _split_summary_output(
    summary: dict,
    ablation: str,
    fast_mode: bool,
) -> Tuple[dict, dict, dict]:
    """Split the full summary into summary, metadata, and audit_log.

    Args:
        summary: Full summary JSON object.
        ablation: Ablation mode used for the run.
        fast_mode: Whether fast mode was enabled.

    Returns:
        Tuple of (summary, metadata, audit_log) objects.
    """
    reasoning_tree = copy.deepcopy(summary)
    input_messages = summary.pop("_input_messages", None)
    input_payload = summary.pop("_input_payload", None)

    root = summary.get("root", {}) or {}
    summary_out = {
        "drug": summary.get("drug", ""),
        "target": summary.get("target", ""),
        "fusion_label": root.get("fusion_label", ""),
        "fusion_reason": root.get("fusion_reason", ""),
        "ml_label": root.get("ml_label", ""),
        "kg_label": root.get("kg_label", ""),
        "rag_label": root.get("rag_label", ""),
        "reasoning_effort": root.get("reasoning_effort", ""),
        "summary_reasoning": root.get("summary_reasoning", ""),
        "summary_children": root.get("children", []) or [],
    }
    metadata = {
        "token_usage_total": root.get("token_usage_total", {}),
        "requested_reasoning": root.get("requested_reasoning", {}),
        "ablation_mode": ablation,
        "fast_mode_flag": fast_mode,
    }
    audit_log = {
        "_input_messages": input_messages or [],
        "_input_payload": input_payload or {},
        "reasoning_tree": reasoning_tree,
    }
    return summary_out, metadata, audit_log
