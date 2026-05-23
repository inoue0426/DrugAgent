import json
from typing import Dict, Tuple

from openai import AzureOpenAI

from src.config_utils import get_reasoning_settings, load_azure_openai_config

LABEL_SCORE_MAP = {
    "STRONG": 1.0,
    "MODERATE": 2.0 / 3.0,
    "WEAK": 1.0 / 3.0,
    "NONE": 0.0,
}
REASONING_SETTINGS = get_reasoning_settings()

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict evidence judge for drug-target interactions (DTI). "
    "DTI means physical, direct binding only. "
    "Return only a JSON object with keys: 'drug', 'target', 'label', 'reason'. "
    "Allowed labels: STRONG, MODERATE, WEAK, NONE. "
    "Label definitions: "
    "STRONG: direct physical binding for this exact drug-target pair is shown "
    "(e.g., Kd/Ki/IC50/EC50, SPR/ITC, competition binding, pull-down, "
    "structure/cryo-EM). "
    "MODERATE: no binding measurement, but explicit direct action for this "
    "exact drug-target pair is stated (e.g., 'drug inhibits/poisons/targets "
    "the target') or a curated database records a direct action edge "
    "(DrugBank/ChEMBL/CTD). "
    "WEAK: only indirect or pathway-level associations, biomarker correlations, "
    "or multi-hop traces (3+ hops), or low-specificity intermediates. "
    "NONE: no direct or indirect evidence."
)

_client: AzureOpenAI | None = None
_client_config: Dict[str, str] | None = None


def normalize_label(label: str) -> str:
    """Normalize a label string to uppercase for consistency.

    Args:
        label: Raw label text.

    Returns:
        Uppercased label string.
    """
    return label.strip().upper()


def map_label_to_score(label: str) -> float:
    """Map a label to its numeric score.

    Args:
        label: Label text such as STRONG or WEAK.

    Returns:
        Numeric score mapped from the label.
    """
    return LABEL_SCORE_MAP.get(normalize_label(label), 0.0)


def judge_evidence_strength(
    source: str, drug: str, target: str, evidence: str
) -> Tuple[str, float]:
    """Judge evidence strength using an LLM and return label and score.

    Args:
        source: Evidence source name (e.g., KG or PubMed).
        drug: Drug name.
        target: Target name.
        evidence: Evidence or reasoning text.

    Returns:
        Tuple of (label, score) where label is STRONG/MODERATE/WEAK/NONE.
    """
    if not evidence or not evidence.strip():
        return "NONE", 0.0

    client = _get_client()
    payload = {
        "source": source,
        "drug": drug,
        "target": target,
        "evidence": evidence,
    }
    request_kwargs = {"temperature": 0}
    if REASONING_SETTINGS is not None:
        request_kwargs["reasoning"] = REASONING_SETTINGS
    response = client.chat.completions.create(
        model=_client_config["deployment_name"],
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ],
        **request_kwargs,
    )
    content = response.choices[0].message.content or ""
    label = _extract_label(content)
    return label, map_label_to_score(label)


def _get_client() -> AzureOpenAI:
    """Initialize and cache the AzureOpenAI client."""
    global _client, _client_config
    if _client is None:
        _client_config = load_azure_openai_config()
        _client = AzureOpenAI(
            api_key=_client_config["api_key"],
            azure_endpoint=_client_config["endpoint"],
            api_version=_client_config["api_version"],
        )
    return _client


def _extract_label(content: str) -> str:
    """Extract a label from an LLM response.

    Args:
        content: Raw LLM response content.

    Returns:
        Normalized label, defaulting to NONE on failure.
    """
    text = content.strip()
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict):
        return "NONE"
    label = normalize_label(str(data.get("label", "")))
    return label if label in LABEL_SCORE_MAP else "NONE"
