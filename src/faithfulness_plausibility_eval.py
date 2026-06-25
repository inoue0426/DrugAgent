from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from anthropic import AnthropicFoundry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config_utils import get_reasoning_settings, load_azure_openai_config
from src.faithfulness_plausibility_prompts import (
    FAITHFULNESS_SYSTEM_PROMPT,
    PLAUSIBILITY_SYSTEM_PROMPT,
    build_faithfulness_user_prompt,
    build_plausibility_user_prompt,
)

DEFAULT_INPUT = Path("data/plausibility_faithfulness_demo.jsonl")
DEFAULT_OUTPUT = Path("data/plausibility_faithfulness_results.jsonl")
DEFAULT_CLAUDE_DEPLOYMENT = os.getenv("CLAUDE_DEPLOYMENT", "claude-opus-4-6")
DEFAULT_SUMMARY_OUTPUT = Path("data/plausibility_faithfulness_from_summary.jsonl")


@dataclass(frozen=True)
class PlausibilityResult:
    score: Optional[int]
    label: Optional[str]
    rationale: Optional[str]
    contradicted_by_input: Optional[bool]
    mechanistically_coherent: Optional[bool]
    raw_response: str
    error: Optional[str] = None


@dataclass(frozen=True)
class FaithfulnessResult:
    overall_score: Optional[int]
    grounded_in_input: Optional[bool]
    has_contradiction: Optional[bool]
    has_unsupported_claim: Optional[bool]
    missing_critical_info: Optional[bool]
    rationale: Optional[str]
    raw_response: str
    error: Optional[str] = None


def _parse_json_object(text: str) -> Dict[str, Any]:
    """Parse a JSON object from a model response string.

    Args:
        text: Raw response text.

    Returns:
        Parsed JSON object as a dict.
    """
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def _build_client() -> AnthropicFoundry:
    """Create a Claude client via AnthropicFoundry.

    Returns:
        AnthropicFoundry client.
    """
    config = load_azure_openai_config()
    return AnthropicFoundry(
        api_key=config["api_key"],
        base_url=config["claude_endpoint"],
    )


def _call_judge(client: AnthropicFoundry, system_prompt: str, user_prompt: str) -> str:
    """Call the judge model and return raw text.

    Args:
        client: Azure OpenAI client.
        system_prompt: System prompt for evaluation.
        user_prompt: User prompt with input content.

    Returns:
        Raw response text.
    """
    request_kwargs: Dict[str, Any] = {"temperature": 0, "max_tokens": 512}
    response = client.messages.create(
        model=DEFAULT_CLAUDE_DEPLOYMENT,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        **request_kwargs,
    )
    content = getattr(response, "content", [])
    if content and hasattr(content[0], "text"):
        return content[0].text or ""
    return ""


def judge_plausibility(
    client: AnthropicFoundry, summary: str, drug_name: str, gene_name: str
) -> PlausibilityResult:
    """Evaluate plausibility for a drug-gene association.

    Args:
        client: Azure OpenAI client.
        summary: Summary text used as input for judging.
        drug_name: Drug name.
        gene_name: Gene symbol.

    Returns:
        PlausibilityResult dataclass.
    """
    user_prompt = build_plausibility_user_prompt(summary, drug_name, gene_name)
    raw = _call_judge(client, PLAUSIBILITY_SYSTEM_PROMPT, user_prompt)
    data = _parse_json_object(raw)
    if not data:
        return PlausibilityResult(
            score=None,
            label=None,
            rationale=None,
            contradicted_by_input=None,
            mechanistically_coherent=None,
            raw_response=raw,
            error="Failed to parse JSON response.",
        )
    return PlausibilityResult(
        score=_coerce_int(data.get("score")),
        label=_coerce_str(data.get("label")),
        rationale=_coerce_str(data.get("rationale")),
        contradicted_by_input=_coerce_bool(data.get("contradicted_by_input")),
        mechanistically_coherent=_coerce_bool(data.get("mechanistically_coherent")),
        raw_response=raw,
        error=None,
    )


def judge_faithfulness(
    client: AnthropicFoundry,
    ml_evidence: str,
    kg_evidence: str,
    rag_evidence: str,
    fusion_reason: str,
) -> FaithfulnessResult:
    """Evaluate faithfulness of a fusion explanation.

    Args:
        client: Azure OpenAI client.
        ml_evidence: ML evidence text.
        kg_evidence: KG evidence text.
        rag_evidence: RAG evidence text.
        fusion_reason: Fusion explanation to evaluate.

    Returns:
        FaithfulnessResult dataclass.
    """
    user_prompt = build_faithfulness_user_prompt(
        ml_evidence, kg_evidence, rag_evidence, fusion_reason
    )
    raw = _call_judge(client, FAITHFULNESS_SYSTEM_PROMPT, user_prompt)
    data = _parse_json_object(raw)
    if not data:
        return FaithfulnessResult(
            overall_score=None,
            grounded_in_input=None,
            has_contradiction=None,
            has_unsupported_claim=None,
            missing_critical_info=None,
            rationale=None,
            raw_response=raw,
            error="Failed to parse JSON response.",
        )
    return FaithfulnessResult(
        overall_score=_coerce_int(data.get("overall_score")),
        grounded_in_input=_coerce_bool(data.get("grounded_in_input")),
        has_contradiction=_coerce_bool(data.get("has_contradiction")),
        has_unsupported_claim=_coerce_bool(data.get("has_unsupported_claim")),
        missing_critical_info=_coerce_bool(data.get("missing_critical_info")),
        rationale=_coerce_str(data.get("rationale")),
        raw_response=raw,
        error=None,
    )


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return True
        if lowered in {"false", "no"}:
            return False
    return None


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _coerce_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Load JSONL records from a file.

    Args:
        path: Path to JSONL file.

    Returns:
        Iterable of parsed dicts.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write JSONL records to a file.

    Args:
        path: Output path for JSONL.
        rows: Iterable of dicts to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _validate_record(record: Dict[str, Any]) -> None:
    required = [
        "drug",
        "gene",
        "summary",
        "ml_evidence",
        "kg_evidence",
        "rag_evidence",
        "fusion_reason",
    ]
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(missing)}")


def _safe_json_load(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _load_payload_from_cell(raw: str) -> Dict[str, Any]:
    """Load payload JSON from a CSV cell or referenced file path.

    Args:
        raw: CSV cell content (JSON string or file path).

    Returns:
        Parsed payload dict.
    """
    raw = (raw or "").strip()
    if not raw:
        return {}
    if raw.endswith(".json"):
        path = Path(raw)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return _safe_json_load(raw)


def _find_first_text(payload: Dict[str, Any], keys: Tuple[str, ...]) -> str:
    queue = deque([payload])
    while queue:
        current = queue.popleft()
        if isinstance(current, dict):
            for key in keys:
                if key in current and isinstance(current[key], str):
                    value = current[key].strip()
                    if value:
                        return value
            for value in current.values():
                if isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(current, list):
            for value in current:
                if isinstance(value, (dict, list)):
                    queue.append(value)
    return ""


def _extract_evidence_text(payload: Dict[str, Any], key: str) -> str:
    evidence = payload.get(key)
    if isinstance(evidence, dict):
        for field in ("reason", "summary", "evidence", "rationale"):
            value = evidence.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _summary_row_to_record(row: Dict[str, Any]) -> Dict[str, Any]:
    drug = str(row.get("drug", "")).strip()
    gene = str(row.get("target", "")).strip()
    summary = str(
        row.get("summary_reasoning", "") or row.get("fusion_reason", "")
    ).strip()
    fusion_reason = str(row.get("fusion_reason", "")).strip()
    payload = _load_payload_from_cell(str(row.get("input_payload", "") or ""))
    ml_evidence = _extract_evidence_text(payload, "ml_evidence") or _find_first_text(
        payload, ("ml_evidence", "ml_reason", "ml_summary")
    )
    kg_evidence = _extract_evidence_text(payload, "kg_evidence") or _find_first_text(
        payload, ("kg_evidence", "kg_reason", "kg_summary")
    )
    rag_evidence = _extract_evidence_text(payload, "rag_evidence") or _find_first_text(
        payload, ("rag_evidence", "rag_reason", "rag_summary")
    )
    return {
        "drug": drug,
        "gene": gene,
        "summary": summary,
        "ml_evidence": ml_evidence,
        "kg_evidence": kg_evidence,
        "rag_evidence": rag_evidence,
        "fusion_reason": fusion_reason,
    }


def _load_summary_csv(path: Path) -> Iterable[Dict[str, Any]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def run_batch(
    input_path: Optional[Path], output_path: Path, summary_csv: Optional[Path]
) -> None:
    """Run plausibility and faithfulness evaluations on a JSONL dataset.

    Args:
        input_path: Input JSONL path.
        output_path: Output JSONL path.
        summary_csv: Optional summary CSV path.
    """
    client = _build_client()
    results = []
    records: Iterable[Dict[str, Any]]
    if summary_csv is not None:
        records = (
            _summary_row_to_record(row) for row in _load_summary_csv(summary_csv)
        )
    else:
        if input_path is None:
            raise ValueError("input_path is required when summary_csv is not provided.")
        records = _load_jsonl(input_path)
    for record in records:
        _validate_record(record)
        plausibility = judge_plausibility(
            client=client,
            summary=record["summary"],
            drug_name=record["drug"],
            gene_name=record["gene"],
        )
        faithfulness = judge_faithfulness(
            client=client,
            ml_evidence=record["ml_evidence"],
            kg_evidence=record["kg_evidence"],
            rag_evidence=record["rag_evidence"],
            fusion_reason=record["fusion_reason"],
        )
        row = {
            "drug": record["drug"],
            "gene": record["gene"],
            "plausibility": asdict(plausibility),
            "faithfulness": asdict(faithfulness),
        }
        results.append(row)
        _print_result(row)
    _write_jsonl(output_path, results)
    print(f"Wrote results to {output_path}")


def _print_result(row: Dict[str, Any]) -> None:
    """Print a human-readable summary for a result row.

    Args:
        row: Result row with plausibility/faithfulness entries.
    """
    plaus = row.get("plausibility", {})
    faith = row.get("faithfulness", {})
    print("=" * 72)
    print(f"Drug: {row.get('drug')}")
    print(f"Gene: {row.get('gene')}")
    print("Plausibility:")
    print(
        f"- score={plaus.get('score')} label={plaus.get('label')} "
        f"contradicted={plaus.get('contradicted_by_input')} "
        f"mechanistic={plaus.get('mechanistically_coherent')}"
    )
    if plaus.get("rationale"):
        print(f"- rationale: {plaus.get('rationale')}")
    if plaus.get("error"):
        print(f"- error: {plaus.get('error')}")
    print("Faithfulness:")
    print(
        f"- score={faith.get('overall_score')} grounded={faith.get('grounded_in_input')} "
        f"contradiction={faith.get('has_contradiction')} "
        f"unsupported={faith.get('has_unsupported_claim')} "
        f"missing_info={faith.get('missing_critical_info')}"
    )
    if faith.get("rationale"):
        print(f"- rationale: {faith.get('rationale')}")
    if faith.get("error"):
        print(f"- error: {faith.get('error')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run plausibility and faithfulness evaluations on JSONL data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSONL path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional summary CSV path (output/summary_*.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path. Default: {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()
    output_path = args.output
    if args.summary_csv is not None and args.output == DEFAULT_OUTPUT:
        output_path = DEFAULT_SUMMARY_OUTPUT
    run_batch(args.input, output_path, args.summary_csv)


if __name__ == "__main__":
    main()
