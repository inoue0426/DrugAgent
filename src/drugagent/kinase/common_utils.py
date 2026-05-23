from __future__ import annotations

import re
from typing import Optional


def normalize_drug(drug: Optional[str]) -> str:
    """Normalize drug name for matching and caching.

    Args:
        drug: Raw drug name.

    Returns:
        Normalized drug string.
    """
    if drug is None:
        return ""
    return " ".join(str(drug).strip().split()).lower()


def normalize_gene(gene: Optional[str]) -> str:
    """Normalize gene name for matching and caching.

    Args:
        gene: Raw gene name.

    Returns:
        Normalized gene string.
    """
    if gene is None:
        return ""
    return " ".join(str(gene).strip().split()).upper()


def normalize_protein(protein: Optional[str]) -> str:
    """Normalize protein name for matching and caching.

    Args:
        protein: Raw protein name.

    Returns:
        Normalized protein string.
    """
    if protein is None:
        return ""
    return " ".join(str(protein).strip().split()).upper()


def clean_text_for_matching(text: Optional[str]) -> str:
    """Normalize whitespace and lowercase for matching.

    Args:
        text: Input text.

    Returns:
        Lowercased text with collapsed whitespace.
    """
    return re.sub(r"\s+", " ", (text or "").strip()).lower()
