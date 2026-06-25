from __future__ import annotations

import asyncio
import json
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from drugagent.config import _get_config, _get_reasoning_settings
from drugagent.summary import (
    _extract_json_block,
    _get_summary_client,
    _run_summary_with_evidence,
    apply_final_decision,
)
from drugagent.utils import (
    _is_rate_limit_exception,
    _sleep_with_backoff,
    normalize_enabled_agents,
)


class EvalMode(str, Enum):
    DRUGAGENT = "drugagent"
    LLM_ONLY_SUMMARY = "llm_only_summary"
    UNSTRUCTURED_FUSION = "unstructured_fusion"
    LLM_ONLY_DECISION = "llm_only_decision"


async def _run_json_llm(
    payload: Dict[str, Any],
    system_message: str,
    reasoning_effort: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    cfg = _get_config()
    reasoning_settings = _get_reasoning_settings()
    client = _get_summary_client(reasoning_effort)

    request_kwargs = {"temperature": 0, "seed": 42}
    if reasoning_effort is not None:
        request_kwargs["reasoning"] = {"effort": reasoning_effort}
    elif reasoning_settings is not None:
        request_kwargs["reasoning"] = reasoning_settings

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": json.dumps(payload)},
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

                try:
                    return client.responses.create(
                        deployment_id=cfg["deployment_name"],
                        input=messages,
                        **request_kwargs,
                    )
                except TypeError:
                    pass

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


def get_llm_only_summary_system_message(enabled_sources: List[str]) -> str:
    source_lines = []
    if "ML" in enabled_sources:
        source_lines.append("- ML evidence")
    if "KG" in enabled_sources:
        source_lines.append("- KG evidence")
    if "RAG" in enabled_sources:
        source_lines.append("- literature evidence")

    sources_text = "\n".join(source_lines)

    return f"""You are a biomedical assistant.

You will receive evidence about a drug-target pair:
{sources_text}

Your task is to summarize each source briefly.

Return valid JSON:
{{
  "type": "summary_only",
  "source_summaries": [
    {{
      "source": "ML|KG|RAG",
      "label": "Low|Moderate|Strong|NONE",
      "summary": string
    }}
  ],
  "summary_reasoning": string
}}

Constraints:
- Do NOT output a final fusion label
- Do NOT create reasoning trees
- Do NOT invent evidence
"""


def get_unstructured_fusion_system_message(enabled_sources: List[str]) -> str:
    source_lines = []
    if "ML" in enabled_sources:
        source_lines.append("- ML prediction evidence")
    if "KG" in enabled_sources:
        source_lines.append("- KG evidence")
    if "RAG" in enabled_sources:
        source_lines.append("- literature evidence")

    sources_text = "\n".join(source_lines)

    return f"""You are a biomedical reasoning assistant.

You will receive evidence about a drug-target pair from these sources:
{sources_text}

Your task is to directly synthesize the evidence into a final interaction label.

Return exactly one valid JSON object with:
{{
  "fusion_label": "Low" | "Moderate" | "Strong",
  "fusion_conf": "LOW" | "MEDIUM" | "HIGH",
  "fusion_reason": string,
  "recommended_next_experiments": [string, ...],
  "timestamp": string
}}

Constraints:
- Do not create a reasoning tree.
- Do not output thought/action/observation fields.
- Do not apply explicit deterministic decision rules.
- Do not invent evidence.
- Use only the provided evidence.
- Be conservative if the evidence is conflicting or incomplete.
"""


def get_llm_only_decision_system_message() -> str:
    return """You are DecisionAgent.

You will receive a reasoning_tree JSON and the original evidence payload.
Your task is to produce the final fusion decision.

Return exactly one valid JSON object with:
{
  "fusion_label": "Low" | "Moderate" | "Strong",
  "fusion_conf": "LOW" | "MEDIUM" | "HIGH",
  "fusion_reason": string,
  "recommended_next_experiments": [string, ...],
  "timestamp": string
}

Constraints:
- Do not invent new primary evidence.
- Do not use hidden reasoning.
- Do not apply hard-coded deterministic rules.
- Use only the reasoning_tree and the evidence payload.
- Be conservative if the evidence is conflicting or incomplete.
"""


async def run_eval_mode(
    payload: Dict[str, Any],
    enabled_sources: List[str],
    mode: EvalMode,
    ablation: str = "",
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """
    mode switch:
    - LLM_ONLY_SUMMARY: no structure, no final fusion
    - UNSTRUCTURED_FUSION: no structure, final label directly
    - LLM_ONLY_DECISION: structured summary, final decision via LLM
    - DRUGAGENT: current structured summary + current decision flow
    """
    enabled_sources = normalize_enabled_agents(enabled_sources)

    if mode == EvalMode.LLM_ONLY_SUMMARY:
        system_message = get_llm_only_summary_system_message(enabled_sources)
        obj, tok = await _run_json_llm(payload, system_message, reasoning_effort)
        obj["_token_usage_summary"] = tok
        return obj

    if mode == EvalMode.UNSTRUCTURED_FUSION:
        system_message = get_unstructured_fusion_system_message(enabled_sources)
        obj, tok = await _run_json_llm(payload, system_message, reasoning_effort)
        obj["_token_usage_summary"] = tok
        return obj

    if mode == EvalMode.LLM_ONLY_DECISION:
        summary = await _run_summary_with_evidence(
            payload=payload,
            ablation=ablation,
            enabled_sources=enabled_sources,
            reasoning_effort=reasoning_effort,
        )
        decision_payload = {
            "drug": summary.get("drug", ""),
            "target": summary.get("target", ""),
            "reasoning_tree": summary.get("root", {}) or {},
            "evidence_payload": payload,
        }
        decision, tok = await _run_json_llm(
            decision_payload,
            get_llm_only_decision_system_message(),
            reasoning_effort,
        )

        root = summary.setdefault("root", {})
        root["fusion_label"] = str(decision.get("fusion_label", "NA"))
        root["fusion_conf"] = str(decision.get("fusion_conf", "LOW"))
        root["fusion_reason"] = str(decision.get("fusion_reason", ""))
        root["_token_usage_decision"] = tok
        return summary

    summary = await _run_summary_with_evidence(
        payload=payload,
        ablation=ablation,
        enabled_sources=enabled_sources,
        reasoning_effort=reasoning_effort,
    )
    await apply_final_decision(
        summary=summary,
        enabled_agents=enabled_sources,
        payload=payload,
        reasoning_effort=reasoning_effort,
    )
    return summary
