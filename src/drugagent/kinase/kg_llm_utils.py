from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI


def _resolve_repo_root() -> Path:
    """Resolve the repository root by walking up to pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[3]


ROOT_DIR = _resolve_repo_root()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from token_usage import TokenUsage

DEFAULT_LLM_MAX_COMPLETION_TOKENS = 512
DEFAULT_LLM_TEMPERATURE = 0.0
DEFAULT_LLM_SEED = 42


def safe_json_loads(raw: str) -> Optional[dict]:
    """Lenient JSON loader that extracts the last object if needed."""
    if raw is None:
        return None
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def llm_call_azure(
    client: AzureOpenAI,
    deployment_name: str,
    system: Optional[str],
    user: str,
    max_completion_tokens: int = DEFAULT_LLM_MAX_COMPLETION_TOKENS,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    seed: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> tuple[str, Optional[Any]]:
    """Call Azure OpenAI chat completions and return response text and usage.

    Args:
        client: Azure OpenAI client.
        deployment_name: Azure deployment name.
        system: System prompt or None.
        user: User prompt.
        max_completion_tokens: Max completion tokens for Azure SDK.
        temperature: Sampling temperature.
        seed: Optional seed for deterministic output.
        response_format: Optional response_format parameter.

    Returns:
        Tuple[str, Optional[Any]]: Raw response text and usage object.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    kwargs: Dict[str, Any] = {
        "model": deployment_name,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    if seed is not None:
        kwargs["seed"] = seed
    if response_format is not None:
        kwargs["response_format"] = response_format

    response = client.chat.completions.create(**kwargs)
    usage = getattr(response, "usage", None)

    try:
        content = response.choices[0].message.content
        raw_text = (
            json.dumps(content) if isinstance(content, (dict, list)) else str(content)
        )
        return raw_text, usage
    except Exception:
        return str(response), usage


def summarize_edge_with_azure(
    edge_info: Dict[str, Any],
    client: AzureOpenAI,
    deployment_name: str,
    usage: Optional[TokenUsage] = None,
) -> Dict[str, Any]:
    """Summarize a single drug-gene edge using Azure LLM."""
    system = (
        "You are an expert at extracting concise experimental evidence from "
        "biomedical Interaction texts. Return ONLY a single JSON object "
        "following the requested schema."
    )
    user_prompt = f"""
Edge: Drug='{edge_info.get('drug_raw')}' (norm='{edge_info.get('drug_norm')}'), Gene='{edge_info.get('gene_raw')}' (norm='{edge_info.get('gene_norm')}')
Metadata: primary_species={edge_info.get('primary_species')}, has_human={edge_info.get('has_human')}, has_drugbank={edge_info.get('has_drugbank')}, has_ctd={edge_info.get('has_ctd')}
Interaction text:
{edge_info.get('interaction_text')}

Task:
1) Produce a 1-2 sentence summary.
2) Fill the JSON schema exactly:
{{
 "summary_text": "...",
 "evidence_type": "binding|functional|expression|phenotype|exposure|other|unclear",
 "species": "...",
 "human_relevant": true|false|null,
 "numeric_result": "IC50=... nM" or null,
 "supporting_sentence": "...",
 "confidence": 0.0-1.0
}}
Return the JSON object only (no extra commentary).
"""

    raw, u = llm_call_azure(
        client=client,
        deployment_name=deployment_name,
        system=system,
        user=user_prompt,
        max_completion_tokens=DEFAULT_LLM_MAX_COMPLETION_TOKENS,
        response_format={"type": "json_object"},
        temperature=DEFAULT_LLM_TEMPERATURE,
        seed=DEFAULT_LLM_SEED,
    )

    if usage is not None:
        usage.add(
            u,
            tag="edge_summary",
            meta={
                "drug": edge_info.get("drug_norm"),
                "gene": edge_info.get("gene_norm"),
            },
        )

    try:
        parsed = json.loads(raw)
    except Exception:
        txt = (edge_info.get("interaction_text") or "").strip()
        first_sent = txt.splitlines()[0] if txt else ""
        parsed = {
            "summary_text": first_sent[:200],
            "evidence_type": "unclear",
            "species": edge_info.get("primary_species"),
            "human_relevant": edge_info.get("has_human"),
            "numeric_result": None,
            "supporting_sentence": first_sent,
            "confidence": 0.0,
        }
    parsed.setdefault("summary_text", "")
    parsed.setdefault("evidence_type", "unclear")
    parsed.setdefault("species", edge_info.get("primary_species") or "")
    parsed.setdefault(
        "human_relevant",
        edge_info.get("has_human") if "has_human" in edge_info else None,
    )
    parsed.setdefault("numeric_result", None)
    parsed.setdefault("supporting_sentence", "")
    parsed.setdefault("confidence", 0.0)
    return parsed


def summarize_path_with_azure(
    path_id: int,
    edge_summaries: List[Dict[str, Any]],
    client: AzureOpenAI,
    deployment_name: str,
    usage: Optional[TokenUsage] = None,
) -> Dict[str, Any]:
    """Summarize a path using edge summaries with Azure LLM."""
    system = (
        "You are an expert biomedical scientist. Given edge summaries, produce a concise "
        "mechanistic interpretation of how the drug could influence the target through intermediates. "
        "Explicitly name key intermediates and the biological process (e.g., signaling, inflammation, GPCR pathway). "
        "Return only JSON."
    )
    edges_text = []
    for i, e in enumerate(edge_summaries, start=1):
        edges_text.append(
            "Edge#{idx}: summary='{summary}', type={etype}, species={species}, "
            "human={human}".format(
                idx=i,
                summary=e.get("summary_text", "")[:300],
                etype=e.get("evidence_type"),
                species=e.get("species"),
                human=e.get("human_relevant"),
            )
        )
    user = (
        "### Edges:\n" + "\n".join(edges_text) + "\n\nReturn JSON ONLY:\n"
        '{"path_summary":"...", "key_edges":[1,2], "key_intermediates":["GENE1"], "process_tags":["signaling"], "confidence":0.0}'
    )

    raw, u = llm_call_azure(
        client=client,
        deployment_name=deployment_name,
        system=system,
        user=user,
        max_completion_tokens=DEFAULT_LLM_MAX_COMPLETION_TOKENS,
        response_format={"type": "json_object"},
        temperature=DEFAULT_LLM_TEMPERATURE,
        seed=DEFAULT_LLM_SEED,
    )

    if usage is not None:
        usage.add(u, tag="path_summary", meta={"path_id": path_id})

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {
            "path_summary": "summary generation incorrect",
            "key_edges": [],
            "key_intermediates": [],
            "process_tags": [],
            "confidence": 0.0,
        }
    return parsed


def llm_judge_dti_from_paths_azure(
    drug: str,
    gene: str,
    path_summaries: List[Dict[str, Any]],
    client: AzureOpenAI,
    deployment_name: str,
    topn: int = 8,
    usage: Optional[TokenUsage] = None,
    binary_class: bool = False,
) -> Dict[str, Any]:
    """Judge DTI strength given path summaries.

    Args:
        binary_class: If True, use binary Active/Inactive labeling.
    """
    system = """You are an expert biomedical scientist generating *mechanistic hypotheses*
for a drug-target relationship using ONLY the provided KG path summaries.
Your goal is to propose the most plausible mechanism and assign an evidence-strength label.
Return a single JSON object ONLY.

Interpretation:
- The label reflects *support strength for a plausible hypothesis*, not definitive proof.
- Prefer mechanistic coherence, specificity, and convergent support across multiple paths/edge types.
- Penalize generic hub-driven chains and vague associations.

Label rules (multi-class):
- Strong: Convergent and specific support for the hypothesis. Typically >=2 independent supporting paths
  or multiple complementary evidence types (e.g., binding/functional/perturbation) that are consistent,
  and at least one path is relatively direct (short or includes a key edge with high-confidence evidence).
- Moderate: Mechanistically coherent and testable hypothesis supported by at least one reasonable path,
  but evidence is indirect, limited, or relies on intermediate nodes; may lack direct binding/functional proof.
- Weak: Only generic/ambiguous support (e.g., hub-heavy, very indirect, non-specific, inconsistent),
  yet still *biologically plausible* as a hypothesis.
- Use Insufficient ONLY if the provided summaries contain essentially no relevant signal
  (empty/irrelevant/no-path). Otherwise choose Weak.

Label rules (binary):
- Active: Evidence indicates a plausible direct or mechanistically coherent interaction.
  Use when you would otherwise choose Strong or Moderate in the multi-class scheme.
- Inactive: Evidence is weak/insufficient/indirect-only such that you would choose Weak or Insufficient.

Confidence guidance:
- Inactive: 0.00-0.70
- Active: 0.55-0.95

Output schema (JSON only):
{
  "label": "Strong|Moderate|Weak|Insufficient" (multi-class) or "Active|Inactive" (binary),
  "confidence": 0.0,
  "hypothesis": "1-2 sentence mechanistic hypothesis linking the drug to the target",
  "rationale": ["short bullet 1", "short bullet 2"],
  "supporting_paths": [1,2],
  "key_intermediates": ["GENE1","GENE2"],
  "next_experiments": ["experiment 1", "experiment 2"],
  "failure_modes": ["hub_path","indirect_only","non_specific","conflict"]
}
"""

    if binary_class:
        system = system.replace(
            '(multi-class) or "Active|Inactive" (binary)',
            "(binary only: Active/Inactive)",
        )
    else:
        system = system.replace(
            '(multi-class) or "Active|Inactive" (binary)',
            "(multi-class only: Strong/Moderate/Weak/Insufficient)",
        )

    ctx_lines = []
    for i, p in enumerate(path_summaries[:topn], start=1):
        hub = p.get("hub_meta", {}) or {}
        hub_hits = hub.get("hub_hits", 0)
        max_deg = hub.get("max_intermediate_degree", 0)
        max_node = hub.get("max_degree_node", None)
        hub_nodes = hub.get("hub_nodes", []) or []

        hub_nodes_str = ", ".join(
            [
                f'{x.get("type")}:{x.get("value")}({x.get("degree")})'
                for x in hub_nodes[:5]
            ]
        )
        if len(hub_nodes) > 5:
            hub_nodes_str += ", ..."

        ctx_lines.append(
            (
                f"Path#{i} (len={p.get('length')}): {p.get('path_summary')}\n"
                f"Key edges: {p.get('key_edges')}\n"
                f"Key intermediates: {p.get('key_intermediates')}\n"
                f"Process tags: {p.get('process_tags')}\n"
                f"Hub meta: hub_hits={hub_hits}, max_intermediate_degree={max_deg}, "
                f"max_degree_node={max_node}, hub_nodes=[{hub_nodes_str}]"
            )
        )
    user = (
        f"Drug: {drug}\nGene: {gene}\n\nPath summaries (top {min(topn, len(path_summaries))}):\n"
        + "\n\n".join(ctx_lines)
        + """

Task:
Return JSON ONLY with fields:
{
  "label": "Strong"|"Moderate"|"Weak"|"Insufficient" (multi-class) OR "Active"|"Inactive" (binary),
  "confidence": 0.0-1.0,
  "hypothesis": "1-2 sentence mechanistic hypothesis linking the drug to the target",
  "rationale": ["bullet1", "bullet2"],
  "supporting_paths": [1,2],
  "key_intermediates": ["GENE1","GENE2"],
  "next_experiments": ["experiment 1", "experiment 2"],
  "failure_modes": ["hub_path","indirect_only","non_specific","conflict"]
}
"""
    )

    if binary_class:
        user = user.replace(
            '"Strong"|"Moderate"|"Weak"|"Insufficient" (multi-class) OR "Active"|"Inactive" (binary)',
            '"Active"|"Inactive" (binary only)',
        )
    else:
        user = user.replace(
            '"Strong"|"Moderate"|"Weak"|"Insufficient" (multi-class) OR "Active"|"Inactive" (binary)',
            '"Strong"|"Moderate"|"Weak"|"Insufficient" (multi-class only)',
        )

    raw, u = llm_call_azure(
        client=client,
        deployment_name=deployment_name,
        system=system,
        user=user,
        max_completion_tokens=DEFAULT_LLM_MAX_COMPLETION_TOKENS,
        response_format={"type": "json_object"},
        temperature=DEFAULT_LLM_TEMPERATURE,
        seed=DEFAULT_LLM_SEED,
    )

    if usage is not None:
        usage.add(u, tag="judge", meta={"drug": drug, "gene": gene, "topn": topn})

    parsed = safe_json_loads(raw)
    if parsed is None:
        parsed = {
            "label": "Insufficient",
            "confidence": 0.0,
            "hypothesis": "",
            "rationale": ["Parsing error"],
            "supporting_paths": [],
            "key_intermediates": [],
            "next_experiments": [],
            "process_tags": [],
            "failure_modes": ["parse_error"],
        }
    return parsed
