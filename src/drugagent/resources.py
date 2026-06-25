#!/usr/bin/env python
# coding: utf-8
"""Local resource loaders for DrugAgent."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

try:
    from drugagent.kinase.rag_utils import load_faiss_index, load_metadata
except Exception:
    from rag_utils import load_faiss_index, load_metadata

from drugagent.config import (KG_PATH, RAG_INDEX_PATH, RAG_META_PATH,
                              RAG_RESULTS_JSONL)

# -----------------------------------------------------------------------------
# Local resource loaders
# -----------------------------------------------------------------------------

_KG_DF: Optional[pd.DataFrame] = None
_RAG_INDEX = None
_RAG_CHUNKS = None

_DEFAULT_RAG_GDRIVE_URL = "https://drive.google.com/drive/folders/1w_QxsiG-Ee9y25JsH7iYxGVdT9G8ybvZ?usp=sharing"


def _get_kg_df() -> pd.DataFrame:
    """Load the KG dataframe once.

    Returns:
        KG dataframe.
    """
    global _KG_DF
    if _KG_DF is None:
        if not KG_PATH.exists():
            raise FileNotFoundError(f"KG file not found: {KG_PATH}")
        _KG_DF = pd.read_csv(KG_PATH, index_col=0)
    return _KG_DF


def _ensure_rag_assets(index_path: Path, meta_path: Path) -> None:
    """Download RAG assets from Google Drive if missing.

    Args:
        index_path: Expected FAISS index path.
        meta_path: Expected metadata JSON path.
    """
    if index_path.exists() and meta_path.exists():
        return

    download_url = os.getenv(
        "DRUGAGENT_RAG_GDRIVE_URL", _DEFAULT_RAG_GDRIVE_URL
    ).strip()
    if not download_url:
        raise FileNotFoundError(
            f"RAG assets not found: {index_path} and {meta_path}. "
            "Set DRUGAGENT_RAG_GDRIVE_URL or provide local files."
        )

    try:
        import gdown  # type: ignore
    except Exception as exc:
        raise FileNotFoundError(
            "RAG assets missing and gdown is not installed. "
            "Install gdown or provide local files."
        ) from exc

    download_dir = Path(os.getenv("DRUGAGENT_RAG_DOWNLOAD_DIR", str(index_path.parent)))
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RAG] Downloading assets from Google Drive into: {download_dir}")
    gdown.download_folder(
        url=download_url, output=str(download_dir), quiet=False, use_cookies=False
    )

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"RAG assets still missing after download: {index_path} or {meta_path}."
        )


def _get_rag_resources() -> Tuple[Any, Any]:
    """Load RAG index and metadata once.

    Returns:
        Tuple of (faiss_index, metadata_chunks).
    """
    global _RAG_INDEX, _RAG_CHUNKS
    _ensure_rag_assets(RAG_INDEX_PATH, RAG_META_PATH)
    if _RAG_INDEX is None:
        _RAG_INDEX = load_faiss_index(str(RAG_INDEX_PATH))
    if _RAG_CHUNKS is None:
        _RAG_CHUNKS = load_metadata(str(RAG_META_PATH))
    return _RAG_INDEX, _RAG_CHUNKS


def _load_rag_result_from_jsonl(
    drug: str, target: str, jsonl_path: Path = RAG_RESULTS_JSONL
) -> Optional[Dict[str, Any]]:
    """Load a precomputed RAG result from a JSONL file for a drug-target pair.

    Args:
        drug: Drug name.
        target: Target gene symbol.
        jsonl_path: Path to the JSONL file with cached RAG results.

    Returns:
        The matched JSON object if found, otherwise None.
    """
    if not jsonl_path.exists():
        return None
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("Drug") == drug and obj.get("Protein") == target:
                    return obj
    except Exception:
        return None
    return None
