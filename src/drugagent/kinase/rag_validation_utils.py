from __future__ import annotations

import copy
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from common_utils import (clean_text_for_matching, normalize_drug,
                          normalize_protein)

# Tunable thresholds for quote validation.
TOKEN_OVERLAP_ACCEPT_THRESH = 0.6
TOKEN_WINDOW_SCORE_ACCEPT_THRESH = 0.4


def quote_contains_entity(quote: str, entity: str, is_protein: bool = False) -> bool:
    """Check whether a quote contains a normalized drug or protein string.

    Args:
        quote: Quote text.
        entity: Drug or protein name.
        is_protein: Whether the entity is a protein/gene.

    Returns:
        True if the normalized entity appears in the quote.
    """
    if not quote or not entity:
        return False
    q = clean_text_for_matching(quote)
    if is_protein:
        e = normalize_protein(entity).lower()
    else:
        e = normalize_drug(entity).lower()
    return e in q


def hypothesis_mentions_entities(
    hypothesis: str, drug: str, protein: str
) -> Tuple[bool, bool]:
    """Check whether the hypothesis explicitly mentions the drug and protein names.

    Args:
        hypothesis: Hypothesis text from the model.
        drug: Drug name from the query.
        protein: Protein/gene name from the query.

    Returns:
        Tuple of booleans: (drug_in_hypothesis, protein_in_hypothesis).
    """
    if not hypothesis:
        return False, False
    hyp = clean_text_for_matching(hypothesis)
    drug_norm = normalize_drug(drug).lower()
    protein_norm = normalize_protein(protein).lower()
    drug_in = drug_norm in hyp if drug_norm else False
    protein_in = protein_norm in hyp if protein_norm else False
    if not protein_in and protein_norm == "ar":
        for alias in ("androgen receptor", "ar-lbd", "ar-lbp"):
            if alias in hyp:
                protein_in = True
                break
    return drug_in, protein_in


def downgrade_label(label: str) -> str:
    """Downgrade a label by one step in strength.

    Args:
        label: Current label string.

    Returns:
        Downgraded label string if known; otherwise the original label.
    """
    order = ["Weak", "Moderate", "Strong"]
    if label not in order:
        return label
    idx = order.index(label)
    return order[max(0, idx - 1)]


def apply_hypothesis_entity_check(
    stage2_obj: Dict[str, Any], drug: str, protein: str
) -> Dict[str, Any]:
    """Downgrade label if the hypothesis is missing drug or target names.

    Args:
        stage2_obj: Parsed stage2 JSON dict.
        drug: Drug name from the query.
        protein: Protein/gene name from the query.

    Returns:
        Updated stage2_obj with possible label downgrade and validation notes.
    """
    if not isinstance(stage2_obj, dict):
        return stage2_obj
    hypothesis = stage2_obj.get("hypothesis", "")
    drug_in_hypothesis, protein_in_hypothesis = hypothesis_mentions_entities(
        hypothesis, drug, protein
    )
    if not (drug_in_hypothesis and protein_in_hypothesis):
        label = stage2_obj.get("label")
        if isinstance(label, str):
            stage2_obj["label"] = downgrade_label(label)
        notes = stage2_obj.get("validation_notes")
        if notes is None or not isinstance(notes, list):
            stage2_obj["validation_notes"] = []
            notes = stage2_obj["validation_notes"]
        notes.append("hypothesis missing drug/target name; label downgraded")
        notes.append(
            f"drug_in_hypothesis={drug_in_hypothesis}, protein_in_hypothesis={protein_in_hypothesis}"
        )
    return stage2_obj


def _simple_token_overlap(a: str, b: str) -> float:
    """Return word-level Jaccard-like overlap (simple) between a and b."""
    if not a or not b:
        return 0.0
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa or not wb:
        return 0.0
    inter = wa.intersection(wb)
    union = wa.union(wb)
    return float(len(inter)) / float(len(union))


def _try_call(fn: Optional[Callable], *args, **kwargs):
    """Call fn if present, else return None."""
    if fn is None:
        return None
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def validate_stage2_quotes(
    stage2_obj: Dict[str, Any],
    drug: str,
    protein: str,
    *,
    pmc2chunks: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    match_quote_to_pmc_fn: Optional[Callable] = None,
    token_window_score_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Validate/annotate stage2_obj.evidence_quotes using available matching functions.

    Rules:
    - Accept quote if exact match OR token_overlap >= 0.6 OR token_window_score >= 0.4.
    - If accepted, annotate quote with pmc/chunk_id/score/method.
    - If label is Strong/SUPPORTED require at least 1 validated drug_side AND 1 validated
      target_side (else downgrade -> 'insufficient').

    Args:
        stage2_obj: Parsed stage2 JSON dict.
        drug: Drug name from the query.
        protein: Protein/gene name from the query.
        pmc2chunks: Mapping pmc -> list of chunk dicts.
        match_quote_to_pmc_fn: Optional external matcher (exact).
        token_window_score_fn: Optional scorer for approximate matches.

    Returns:
        Updated stage2_obj with validated evidence_quotes and possible label downgrade.
    """
    if not isinstance(stage2_obj, dict):
        return stage2_obj

    ev = stage2_obj.get("evidence_quotes") or {}
    if "evidence_quotes_raw" not in stage2_obj:
        stage2_obj["evidence_quotes_raw"] = copy.deepcopy(ev)
    validated = {"drug_side": [], "target_side": [], "pair_side": []}

    pmcids = stage2_obj.get("pmcids")
    candidate_pmcs = []
    if pmcids:
        try:
            if isinstance(pmcids, str):
                candidate_pmcs = json.loads(pmcids)
            elif isinstance(pmcids, (list, dict)):
                candidate_pmcs = (
                    list(pmcids) if isinstance(pmcids, list) else list(pmcids)
                )
        except Exception:
            candidate_pmcs = []
    if not candidate_pmcs and pmc2chunks:
        candidate_pmcs = list(pmc2chunks.keys())

    def _validate_quote(qtext: str) -> Optional[Dict[str, Any]]:
        if not qtext or len(qtext.strip()) < 5:
            return None
        q = qtext.strip()
        q_match = clean_text_for_matching(q).replace("…", " ").replace("...", " ")

        exact_res = _try_call(match_quote_to_pmc_fn, q)
        if exact_res:
            try:
                method, meta = exact_res
                score = meta.get("score", 1.0) if isinstance(meta, dict) else 1.0
                return {
                    "quote": q,
                    "pmc": meta.get("pmc"),
                    "chunk_id": meta.get("chunk_id"),
                    "method": method,
                    "score": score,
                }
            except Exception:
                pass

        tws = _try_call(token_window_score_fn, q, candidate_pmcs)
        if tws is not None:
            try:
                if isinstance(tws, (int, float)):
                    score = float(tws)
                    if score >= TOKEN_WINDOW_SCORE_ACCEPT_THRESH:
                        return {
                            "quote": q,
                            "pmc": None,
                            "chunk_id": None,
                            "method": "token_window_score",
                            "score": score,
                        }
                elif isinstance(tws, dict):
                    best = max(tws.items(), key=lambda x: x[1])
                    score = float(best[1])
                    if score >= TOKEN_WINDOW_SCORE_ACCEPT_THRESH:
                        return {
                            "quote": q,
                            "pmc": best[0],
                            "chunk_id": None,
                            "method": "token_window_score",
                            "score": score,
                        }
            except Exception:
                pass

        if pmc2chunks:
            best_score = 0.0
            best_meta = None
            for pmc in candidate_pmcs:
                for chunk in pmc2chunks.get(pmc, []):
                    text = chunk.get("text", "")
                    if not text:
                        continue
                    text_match = clean_text_for_matching(text)
                    if q_match and q_match in text_match:
                        return {
                            "quote": q,
                            "pmc": pmc,
                            "chunk_id": chunk.get("chunk_id")
                            or chunk.get("id")
                            or None,
                            "method": "substring_exact",
                            "score": 1.0,
                        }
                    ov = _simple_token_overlap(q_match, text_match)
                    if ov > best_score:
                        best_score = ov
                        best_meta = {
                            "pmc": pmc,
                            "chunk_id": chunk.get("chunk_id")
                            or chunk.get("id")
                            or None,
                        }
            if best_score >= TOKEN_OVERLAP_ACCEPT_THRESH:
                return {
                    "quote": q,
                    "pmc": best_meta.get("pmc"),
                    "chunk_id": best_meta.get("chunk_id"),
                    "method": "token_overlap",
                    "score": best_score,
                }

        return None

    for channel in ("drug_side", "target_side", "pair_side"):
        for q in ev.get(channel) or []:
            qtxt = (
                q
                if isinstance(q, str)
                else (q.get("quote") or q.get("snippet") or str(q))
            )
            res = _validate_quote(qtxt)
            if res:
                validated[channel].append(res)

    stage2_obj.setdefault("evidence_quotes", {})
    stage2_obj["evidence_quotes"]["drug_side"] = validated["drug_side"]
    stage2_obj["evidence_quotes"]["target_side"] = validated["target_side"]
    stage2_obj["evidence_quotes"]["pair_side"] = validated["pair_side"]

    label = (stage2_obj.get("label") or "").lower()
    if label in ("strong", "supported"):
        ok = bool(validated["pair_side"]) or (
            bool(validated["drug_side"]) and bool(validated["target_side"])
        )
        if not ok:
            stage2_obj["label"] = "insufficient"

    stage2_obj.setdefault("validation", {})
    stage2_obj["validation"]["validated_counts"] = {
        k: len(v) for k, v in validated.items()
    }
    return stage2_obj


def _extract_quote_text_normalized(q: Any) -> str:
    """Normalize quote text for downstream checks."""
    if q is None:
        return ""
    if isinstance(q, str):
        return q
    if isinstance(q, dict):
        return q.get("quote") or q.get("snippet") or q.get("text") or ""
    return str(q)


def validate_stage2_output(
    stage2_obj: Dict[str, Any], drug: str, protein: str
) -> Dict[str, Any]:
    """
    Post-validate stage2 results (stage2_obj parsed JSON).
    - If pair_side quotes do not include both drug and protein, mark pair evidence flags accordingly.
    - Remove pair_side quotes that clearly reference other drug names (optional).
    """
    if not isinstance(stage2_obj, dict):
        return {
            "raw": stage2_obj,
            "validation_notes": [
                "stage2 output not JSON/dict; validation applied fallback structure"
            ],
            "evidence_quotes": {"drug_side": [], "target_side": [], "pair_side": []},
            "pair_evidence": {
                "direct_dti_in_same_pmcid": False,
                "mentions_both_explicitly_in_same_pmcid": False,
            },
        }

    ev = stage2_obj.get("evidence_quotes")
    if ev is None or not isinstance(ev, dict):
        stage2_obj["evidence_quotes"] = {
            "drug_side": [],
            "target_side": [],
            "pair_side": [],
        }
        ev = stage2_obj["evidence_quotes"]

    ps = ev.get("pair_side") or []

    validated_pair = []
    for q in ps:
        try:
            qtxt = _extract_quote_text_normalized(q)
            if quote_contains_entity(
                qtxt, drug, is_protein=False
            ) and quote_contains_entity(qtxt, protein, is_protein=True):
                validated_pair.append(q)
        except Exception:
            continue
    stage2_obj["evidence_quotes"]["pair_side"] = validated_pair

    if "pair_evidence" not in stage2_obj or not isinstance(
        stage2_obj.get("pair_evidence"), dict
    ):
        stage2_obj["pair_evidence"] = {
            "direct_dti_in_same_pmcid": False,
            "mentions_both_explicitly_in_same_pmcid": False,
        }

    if not validated_pair:
        stage2_obj["pair_evidence"]["direct_dti_in_same_pmcid"] = False
        stage2_obj["pair_evidence"]["mentions_both_explicitly_in_same_pmcid"] = False

    return stage2_obj
