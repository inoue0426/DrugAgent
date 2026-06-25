#!/usr/bin/env python
# coding: utf-8
"""CSV persistence utilities for DrugAgent outputs."""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional

from drugagent.config import ALL_EVIDENCE_AGENTS
from drugagent.utils import config_id_from_enabled, normalize_enabled_agents

OUTPUT_LABEL_MAP = {"Low": "Weak"}
PAYLOAD_DIR = Path("output/input_payloads")


def ensure_csv_schema(filename: str, fields: List[str]) -> None:
    """Ensure a CSV file matches the desired schema, rewriting if needed.

    Args:
        filename: Path to the CSV file.
        fields: Desired header fields.
    """
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
    """Sanitize model name for filenames.

    Args:
        name: Model name.

    Returns:
        Sanitized model name.
    """
    return name.replace(":", "_")


def _normalize_reasoning_token(reasoning_effort: Optional[str]) -> str:
    """Normalize reasoning_effort into a filename-safe token.

    Args:
        reasoning_effort: Raw reasoning effort.

    Returns:
        Filename-safe reasoning token.
    """
    if reasoning_effort is None:
        return "none"
    token = str(reasoning_effort).strip().lower()
    if token == "":
        return "none"
    if token in {"h", "high"}:
        return "high"
    if token in {"m", "med", "medium"}:
        return "medium"
    if token in {"l", "low"}:
        return "low"
    return re.sub(r"[^0-9a-zA-Z_-]", "_", token)


def _summary_filename_for_config(
    enabled_agents: List[str],
    model: Optional[str],
    reasoning_effort: Optional[str],
    save_version: Optional[str] = None,
) -> str:
    """Build a summary CSV filename for a config.

    Args:
        enabled_agents: Enabled evidence sources.
        model: Model name.
        reasoning_effort: Reasoning effort token.
        save_version: Optional version override.

    Returns:
        Filename string.
    """
    enabled = normalize_enabled_agents(enabled_agents)
    config_id = config_id_from_enabled(enabled)
    model_part = "nomodel" if not model else sanitize_model_name(model)
    reason_token = _normalize_reasoning_token(reasoning_effort)

    version_part = save_version or "v0"

    os.makedirs("output", exist_ok=True)
    return f"output/summary_{version_part}_{config_id}_{model_part}_{reason_token}.csv"


def save_summary_to_csv(
    summary: dict,
    ablation: str,
    model: Optional[str],
    reasoning_effort: Optional[str],
    save_version: Optional[str] = None,
) -> None:
    """Persist summary results to CSV.

    Args:
        summary: Summary JSON object.
        ablation: Ablation mode.
        model: Model name.
        reasoning_effort: Reasoning effort token.
        save_version: Optional version override.
    """

    def _normalize_output_label(label: str) -> str:
        """Map internal labels to output labels for CSV exports.

        Args:
            label: Internal label string.

        Returns:
            Output label string.
        """
        return OUTPUT_LABEL_MAP.get(label, label)

    root = summary.get("root", {}) or {}
    fusion_label = str(root.get("fusion_label", "") or "").strip()
    if fusion_label == "" or fusion_label.upper() in {"NA", "N/A", "NONE", "NULL"}:
        return
    fusion_label = _normalize_output_label(fusion_label)
    enabled_agents = root.get("enabled_agents") or ALL_EVIDENCE_AGENTS
    enabled_agents = normalize_enabled_agents(enabled_agents)

    filename = _summary_filename_for_config(
        enabled_agents,
        model,
        reasoning_effort,
        save_version=save_version,
    )

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
        "fusion_label": fusion_label,
        "fusion_conf": root.get("fusion_conf", ""),
        "fusion_rule": root.get("fusion_rule", ""),
        "fusion_reason": root.get("fusion_reason", ""),
        "ml_label": _normalize_output_label(str(root.get("ml_label", "") or "")),
        "kg_label": _normalize_output_label(str(root.get("kg_label", "") or "")),
        "rag_label": _normalize_output_label(str(root.get("rag_label", "") or "")),
        "enabled_agents": ",".join(enabled_agents),
        "config": config_id_from_enabled(enabled_agents),
        "reasoning_effort": reasoning_effort or "",
        "summary_reasoning": root.get("summary_reasoning", ""),
        "fusion_sources": ",".join(root.get("fusion_sources", []) or []),
    }

    input_payload_obj = summary.get("_input_payload") or {}
    input_payload_path = ""
    if input_payload_obj:
        PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
        drug_slug = str(summary.get("drug", "")).strip().replace(" ", "_")
        target_slug = str(summary.get("target", "")).strip().replace(" ", "_")
        payload_name = f"{drug_slug}__{target_slug}.json"
        payload_file = PAYLOAD_DIR / payload_name
        try:
            payload_file.write_text(
                json.dumps(input_payload_obj, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            input_payload_path = str(payload_file)
        except Exception:
            input_payload_path = ""
    try:
        input_payload_json = json.dumps(input_payload_obj, ensure_ascii=True)
        if len(input_payload_json) > 2000:
            input_payload_json = input_payload_json[:2000] + "...(truncated)"
    except Exception:
        input_payload_json = ""
    row["input_payload"] = input_payload_path or input_payload_json

    token_total = root.get("token_usage_total") or {}
    row["token_usage"] = json.dumps(token_total, ensure_ascii=True)

    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0

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
    save_version: Optional[str] = None,
) -> bool:
    """Check per-config+reasoning CSV for an existing processed row.

    Args:
        drug: Drug name.
        gene: Target gene symbol.
        ablation: Ablation mode.
        model: Model name.
        enabled_agents: Enabled evidence sources.
        reasoning_effort: Reasoning effort token.

    Returns:
        True if already processed.
    """
    enabled_agents = normalize_enabled_agents(enabled_agents)
    filename = _summary_filename_for_config(
        enabled_agents,
        model,
        reasoning_effort,
        save_version=save_version,
    )

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
            if (row.get("reasoning_effort") or "") != (reasoning_effort or ""):
                continue
            fusion_label = str(row.get("fusion_label", "") or "").strip()
            if fusion_label == "" or fusion_label.upper() in {
                "NA",
                "N/A",
                "NONE",
                "NULL",
            }:
                continue
            return True
    return False
