from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from common_utils import normalize_drug, normalize_protein
from rag_validation_utils import (
    apply_hypothesis_entity_check,
    validate_stage2_output,
    validate_stage2_quotes,
)
from token_usage import TokenUsage

DEFAULT_CACHE_PATH = "rag_pair_cache.sqlite3"
CACHE_VERSION = "v2_science"

INDEX_PATH = "rag_index.faiss"
META_PATH = "rag_metadata.json"

ASSAY_TERMS_DEFAULT = (
    "binding affinity Kd Ki IC50 EC50 potency selectivity "
    "SPR BLI ITC MST DSF thermal shift CETSA "
    "competition radioligand pull-down chemoproteomics "
    "kinase assay enzymatic activity inhibition phosphorylation "
    "target engagement"
)


def _contains_term(text: str, term: str) -> bool:
    """Return True if term appears in text, with stricter matching for short tokens.

    Args:
        text: Source text.
        term: Term to match.

    Returns:
        True if term appears in text.
    """
    if not text or not term:
        return False
    if len(term) <= 2:
        return (
            re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE) is not None
        )
    return term.lower() in text.lower()


def _binary_label_from_strength(label: Optional[str]) -> Optional[str]:
    """Map ordinal labels to binary Active/Inactive labels.

    Args:
        label: Ordinal label string.

    Returns:
        Binary label string or the original label if unknown.
    """
    if label is None:
        return None
    norm = str(label).strip().lower()
    if norm in {"strong", "moderate"}:
        return "Active"
    if norm in {"weak", "insufficient", "low"}:
        return "Inactive"
    return label


def _apply_binary_label_to_rag_obj(obj: Dict[str, Any]) -> None:
    """Apply binary label mapping to RAG result in-place.

    Args:
        obj: RAG result dict.
    """

    def _map(container: Any) -> None:
        if not isinstance(container, dict):
            return
        label = container.get("label")
        binary = _binary_label_from_strength(label)
        if binary is None or binary == label:
            return
        container.setdefault("label_ordinal", label)
        container["label"] = binary

    _map(obj)
    stage2 = obj.get("stage2")
    if isinstance(stage2, dict):
        _map(stage2)


class PairCacheSQLite:
    """SQLite-backed cache for (drug, protein) RAG results.

    Args:
        path: SQLite file path.
    """

    def __init__(self, path: str = DEFAULT_CACHE_PATH) -> None:
        self._path = path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pair_cache (
                    key_hash TEXT PRIMARY KEY,
                    key_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def get(self, key_hash: str) -> Optional[Any]:
        """Fetch cached result by key hash."""
        with sqlite3.connect(self._path) as conn:
            row = conn.execute(
                "SELECT result_json FROM pair_cache WHERE key_hash = ?",
                (key_hash,),
            ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    def set(self, key_hash: str, key_json: str, result: Any) -> None:
        """Insert or replace cached result."""
        result_json = json.dumps(result, ensure_ascii=True, sort_keys=True)
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pair_cache
                    (key_hash, key_json, result_json, created_at)
                VALUES
                    (?, ?, ?, ?)
                """,
                (key_hash, key_json, result_json, time.time()),
            )
            conn.commit()


def make_pair_cache_key(
    drug_norm: str,
    protein_norm: str,
    deployment_name: str,
    kg_version: str,
    max_hops: int,
    max_paths: int,
    avoid_hubs: bool,
    hub_degree_cutoff: int,
    topn_paths_for_judge: int,
    assay_terms: str,
    k: int,
    top_pair_pmc: int,
    top_drug_pmc: int,
    top_target_pmc: int,
    chunks_per_pmc: int,
    min_chunk_words: int,
    cache_version: str = CACHE_VERSION,
) -> Tuple[str, str]:
    """Create a stable JSON cache key and its SHA256 hash.

    NOTE: include retrieval & bundling hyperparameters so cached results remain valid.
    """
    key_obj = {
        "avoid_hubs": avoid_hubs,
        "cache_version": cache_version,
        "deployment_name": deployment_name,
        "drug_norm": drug_norm,
        "gene_norm": protein_norm,
        "hub_degree_cutoff": hub_degree_cutoff,
        "kg_version": kg_version,
        "max_hops": max_hops,
        "max_paths": max_paths,
        "topn_paths_for_judge": topn_paths_for_judge,
        # ---- new retrieval params ----
        "assay_terms": assay_terms,
        "k": int(k),
        "top_pair_pmc": int(top_pair_pmc),
        "top_drug_pmc": int(top_drug_pmc),
        "top_target_pmc": int(top_target_pmc),
        "chunks_per_pmc": int(chunks_per_pmc),
        "min_chunk_words": int(min_chunk_words),
    }
    key_json = json.dumps(
        key_obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    key_hash = hashlib.sha256(key_json.encode("utf-8")).hexdigest()
    return key_hash, key_json


def load_faiss_index(index_path: str = INDEX_PATH) -> faiss.Index:
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    return faiss.read_index(index_path)


def load_metadata(meta_path: str = META_PATH) -> Any:
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    if "429" in msg:
        return True
    if "rate limit" in msg:
        return True
    if "too many requests" in msg:
        return True
    return False


def get_retry_after(e: Exception):
    if hasattr(e, "response") and hasattr(e.response, "headers"):
        ra = e.response.headers.get("Retry-After")
        if ra:
            try:
                return float(ra)
            except:
                return None
    return None


import random
import time


def call_with_retry(func, max_retries=6, base_backoff=1.5, max_backoff=120):

    for attempt in range(max_retries + 1):

        try:
            return func()

        except Exception as e:

            if not is_rate_limit_error(e):
                raise  # Non-429 errors should fail fast.

            if attempt == max_retries:
                raise

            retry_after = get_retry_after(e)
            print(
                f"[call_with_retry] got error: {str(e)[:200]} retry_after={retry_after}",
                flush=True,
            )

            if retry_after:
                sleep_time = retry_after + random.uniform(0, 1.0)
            else:
                sleep_time = min(
                    max_backoff,
                    base_backoff * (2**attempt) * (0.8 + 0.4 * random.random()),
                )

            print(f"[429] sleeping {sleep_time:.1f}s (attempt {attempt+1})", flush=True)
            time.sleep(sleep_time)


# NOTE: `filtered_chunks` must exist in runtime scope. Keep it global if your notebook defines it.
# filtered_chunks: List[Dict[str, Any]] = ...


# --------------------------------------------------------------------------------------
# LLM labeling
# --------------------------------------------------------------------------------------


def clean_excerpt_text(s: str) -> str:
    # remove control chars, collapse whitespace
    s2 = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", s or "")
    s2 = re.sub(r"\s+", " ", s2).strip()
    # optionally truncate very long metadata-like sequences
    return s2


def generate_strength_label_science(
    query_text: str,
    chunks: List[Dict[str, Any]],
    pmcids: Any,
    client: Any,
    deployment_name: str,
    usage: Optional[TokenUsage] = None,
) -> str:
    """
    Single-stage version using string.Template to avoid brace-format errors.
    Returns pretty JSON string or error JSON string.
    """
    import json
    from string import Template

    # Normalize pmcids into a JSON literal string safe to embed in the prompt
    if isinstance(pmcids, (dict, list)):
        pmcids_json = json.dumps(pmcids, sort_keys=True, ensure_ascii=False)
    elif isinstance(pmcids, str) and pmcids.strip().startswith(("[", "{")):
        pmcids_json = pmcids
    else:
        pmcids_json = json.dumps(pmcids, ensure_ascii=False)

    # Deterministic chunk ordering
    chunks = sorted(
        chunks,
        key=lambda c: (
            c.get("pmc", ""),
            c.get("token_start", 0),
            c.get("token_end", 0),
            c.get("text", "")[:50],
        ),
    )

    formatted: List[str] = []
    for j, c in enumerate(chunks):
        c_copy = {**c}
        c_copy["text"] = clean_excerpt_text(c_copy.get("text", ""))
        header = f"[PMCID={c_copy['pmc']} | chunk={j} | tokens={c_copy.get('token_start')}-{c_copy.get('token_end')}]"
        formatted.append(header + "\n" + c_copy["text"])

    excerpt = "\n\n---\n\n".join(formatted)

    # -------------------------
    # Use string.Template for safe interpolation
    # -------------------------
    prompt_template = Template(
        r"""
You are an expert pharmacology and molecular biology annotator and hypothesis writer.

Task: Using ONLY the provided excerpt, produce structured evidence (stage1) and then a hypothesis-oriented judgment (stage2).
Return ONLY valid JSON with exactly two top-level keys: "stage1" and "stage2".

REQUIREMENTS:
- Do NOT use outside knowledge. Use only text in the excerpt.
- If a requested field is not present, use "unknown" or [] as appropriate.
- Keep any verbatim quotes <=25 words.
- Output must be strictly valid JSON (no commentary, no extra keys beyond those described).

1) stage1 (structured evidence) schema:
{
  "drug_profile": {
    "mentions_drug": true/false,
    "drug_entities_found": ["..."],
    "assay_signals": ["binding","functional","engagement","selectivity","docking","none/unknown"],
    "direct_measurements": ["Kd","Ki","IC50","EC50","none/unknown"],
    "experimental_context_quotes": ["<=25 words", "..."]
  },
  "target_profile": {
    "mentions_target": true/false,
    "target_entities_found": ["..."],
    "assay_signals": ["binding","functional","engagement","constitutive_activity","selectivity","docking","none/unknown"],
    "experimental_context_quotes": ["<=25 words", "..."]
  },
  "pair_evidence": {
    "mentions_both_explicitly_in_same_pmcid": true/false,
    "direct_dti_in_same_pmcid": true/false,
    "co_mention_chunks": [ {"pmc":"PMC123", "chunk_id":3, "snippet":"<=200 chars"} , ... ],
    "pair_quotes": ["<=25 words","..."]
  },
  "notes": "<1-3 sentences grounded in excerpt>"
}

2) stage2 (judgment) schema:
{
  "label": "<Strong|Moderate|Weak>",
  "reason": "<1-4 sentences grounded in stage1 + excerpt>",
  "hypothesis": "<1-2 sentences: how Drug X might act on Target Y>",
  "next_experiment": "<1-2 sentences: most direct validation experiment>",
  "evidence_quotes": {
    "drug_side": ["<=25 words", ...],
    "target_side": ["<=25 words", ...],
    "pair_side": ["<=25 words", ...]
  },
  "pmcids": $pmcids_json,
  "stage1": "<FULL stage1 object defined above>"
}

DEFINITIONS and RULES (for stage2 - use stage1 fields only, do not invent):
- "Experimental signals" = any of assay_signals contains ["binding","functional","engagement"].
- "Docking-only" = assay_signals contains "docking" AND does NOT contain any experimental signals.
- "Unknown" = mentions_drug or mentions_target is false OR assay_signals is ["none/unknown"].
- "Strong side evidence" = experimental signals present AND at least one supporting quote exists.

Label rules (conservative):
- Strong: explicit direct DTI measurement (Kd/Ki/IC50/EC50) AND explicit statement linking the drug and target in the SAME PMCID, with at least one pair_side quote. If any uncertainty, do NOT use Strong.
- Moderate: experimental signals are present (binding/functional/engagement) but Strong criteria are NOT met (e.g., one-sided evidence, no same-PMCID linkage, or quotes are incomplete).
- Weak: explicitly negative statements in the excerpt ("no binding", "not a target"), OR evidence is docking-only, OR both sides are Unknown / only speculative.

Output instructions:
- Provide 1-2 verbatim quotes for drug-side and 1-2 for target-side when available (<=25 words each).
- Provide a concrete "next_experiment" that could validate the DTI, chosen based on what is missing in stage1.
- The hypothesis must explicitly mention the drug and target names as written in the excerpt.

BEGIN EXCERPT:
<<<BEGIN EXCERPT>>>
$excerpt
<<<END EXCERPT>>>
""".strip()
    )

    prompt = prompt_template.substitute(excerpt=excerpt, pmcids_json=pmcids_json)

    # Single model call
    resp = call_with_retry(
        lambda: client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            seed=42,
            response_format={"type": "json_object"},
            timeout=240.0,
        ),
        max_retries=4,
        base_backoff=1.5,
    )

    if usage is not None:
        usage.add(
            getattr(resp, "usage", None),
            tag="judge_single_stage",
            meta={"deployment": deployment_name},
        )

    result_text = resp.choices[0].message.content.strip()

    # Validate JSON and required keys
    try:
        parsed = json.loads(result_text)
    except Exception:
        return json.dumps(
            {"error": "invalid_json_from_model", "raw": result_text}, ensure_ascii=False
        )

    if not isinstance(parsed, dict) or "stage1" not in parsed or "stage2" not in parsed:
        return json.dumps(
            {"error": "missing_stage1_or_stage2", "raw": parsed}, ensure_ascii=False
        )

    return json.dumps(parsed, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------------------
# Retrieval helpers (depend on `filtered_chunks`)
# --------------------------------------------------------------------------------------


def pick_top_pmcids(
    score_dict: Dict[int, float],
    *,
    filtered_chunks: List[Dict[str, Any]],
    top_pmc: int = 3,
    drug: Optional[str] = None,
    protein: Optional[str] = None,
    protein_aliases: Optional[List[str]] = None,
    require_drug: bool = False,
    require_protein: bool = False,
) -> List[str]:
    def _pmc_terms() -> tuple[list[str], list[str]]:
        drug_terms: list[str] = []
        protein_terms: list[str] = []
        if drug:
            raw = drug.strip().lower()
            norm = normalize_drug(drug).lower() if drug else ""
            for t in (raw, norm):
                if t and t not in drug_terms:
                    drug_terms.append(t)
        if protein:
            if protein_aliases:
                for alias in protein_aliases:
                    a = (alias or "").strip().lower()
                    if a and a not in protein_terms:
                        protein_terms.append(a)
            raw_p = protein.strip().lower()
            norm_p = normalize_protein(protein).lower() if protein else ""
            for t in (raw_p, norm_p):
                if t and t not in protein_terms:
                    protein_terms.append(t)
        return drug_terms, protein_terms

    def _pmc_mentions(pmc: str, terms: list[str]) -> bool:
        if not terms:
            return False
        for idx in score_dict.keys():
            chunk = filtered_chunks[int(idx)]
            if chunk.get("pmc") != pmc:
                continue
            text = (chunk.get("text", "") or "").lower()
            if any(_contains_term(text, t) for t in terms):
                return True
        return False

    pmc_score: Dict[str, float] = {}
    for idx, score in score_dict.items():
        pmc = filtered_chunks[int(idx)]["pmc"]
        pmc_score[pmc] = max(pmc_score.get(pmc, -1e9), float(score))

    drug_terms, protein_terms = _pmc_terms()
    if require_drug:
        pmc_score = {
            pmc: score
            for pmc, score in pmc_score.items()
            if _pmc_mentions(pmc, drug_terms)
        }
    if require_protein:
        pmc_score = {
            pmc: score
            for pmc, score in pmc_score.items()
            if _pmc_mentions(pmc, protein_terms)
        }
    if not pmc_score:
        pmc_score = {}
        for idx, score in score_dict.items():
            pmc = filtered_chunks[int(idx)]["pmc"]
            pmc_score[pmc] = max(pmc_score.get(pmc, -1e9), float(score))

    ranked = sorted(pmc_score.items(), key=lambda x: (-x[1], x[0]))
    return [pmc for pmc, _ in ranked[:top_pmc]]


def collect_chunks_for_pmc(
    score_dict: Dict[int, float],
    pmcid: str,
    *,
    filtered_chunks: List[Dict[str, Any]],
    top_chunks: int = 4,
    drug: Optional[str] = None,
    protein: Optional[str] = None,
    protein_aliases: Optional[List[str]] = None,
    require_drug: bool = False,
    require_protein: bool = False,
    results_bonus=0.5,
    methods_bonus=-0.2,
    co_mention_weight=1.0,
    single_mention_bonus=0.1,
    min_chunk_words: int = 15,  # Added: default 15 words.
) -> List[Dict[str, Any]]:
    """
    Collect top chunks for a given PMCID, with bonuses:
      - Results section boost
      - Co-mention boost if chunk contains both drug and protein strings (case-insensitive)
    Exclude chunks shorter than `min_chunk_words` (by word count).
    """

    def section_bonus(text: str) -> float:
        t = (text or "")[:120].lower()
        if "results" in t:
            return results_bonus
        if "methods" in t:
            return methods_bonus
        return 0.0

    def _build_terms() -> tuple[list[str], list[str]]:
        drug_terms: list[str] = []
        protein_terms: list[str] = []
        if drug:
            raw = drug.strip().lower()
            norm = normalize_drug(drug).lower() if drug else ""
            for t in (raw, norm):
                if t and t not in drug_terms:
                    drug_terms.append(t)
        if protein:
            if protein_aliases:
                for alias in protein_aliases:
                    a = (alias or "").strip().lower()
                    if a and a not in protein_terms:
                        protein_terms.append(a)
            raw_p = protein.strip().lower()
            norm_p = normalize_protein(protein).lower() if protein else ""
            if norm_p != "ar":
                for t in (raw_p, norm_p):
                    if t and t not in protein_terms:
                        protein_terms.append(t)
        return drug_terms, protein_terms

    def _mentions_any(text: str, terms: list[str]) -> bool:
        if not terms:
            return False
        txt = (text or "").lower()
        return any(_contains_term(txt, t) for t in terms)

    def co_mention_bonus(
        text: str, drug: Optional[str], protein: Optional[str]
    ) -> float:
        b = 0.0
        if not drug and not protein:
            return 0.0
        txt = (text or "").lower()
        if drug and drug.lower() in txt and protein and protein.lower() in txt:
            b += co_mention_weight  # big bonus for explicit co-mention
        elif drug and drug.lower() in txt:
            b += single_mention_bonus
        elif protein and protein.lower() in txt:
            b += single_mention_bonus
        return b

    def word_count(text: str) -> int:
        if not text:
            return 0
        # simple whitespace split is sufficient here
        return len(re.findall(r"\w+", text))

    idxs = [
        int(idx)
        for idx in score_dict.keys()
        if filtered_chunks[int(idx)]["pmc"] == pmcid
    ]

    drug_terms, protein_terms = _build_terms()

    def _select(require_drug_flag: bool, require_protein_flag: bool) -> List[int]:
        scored = []
        for idx in idxs:
            chunk = filtered_chunks[idx]
            text = chunk.get("text", "")
            if require_drug_flag and not _mentions_any(text, drug_terms):
                continue
            if require_protein_flag and not _mentions_any(text, protein_terms):
                continue
            base = float(score_dict[idx])
            bonus = section_bonus(text) + co_mention_bonus(text, drug, protein)
            scored.append((base + bonus, idx))

        scored.sort(key=lambda x: x[0], reverse=True)

        selected_idxs: List[int] = []
        for _, idx in scored:
            if len(selected_idxs) >= top_chunks:
                break
            chunk = filtered_chunks[idx]
            wc = word_count(chunk.get("text", ""))
            if wc < min_chunk_words:
                continue
            selected_idxs.append(idx)
        return selected_idxs

    selected_idxs = _select(require_drug, require_protein)
    if not selected_idxs and (require_drug or require_protein):
        selected_idxs = _select(False, False)

    return [filtered_chunks[idx] for idx in selected_idxs]


# --------------------------------------------------------------------------------------
# Multi-channel retrieval (pair/drug/target) + bundling
# --------------------------------------------------------------------------------------


def _expand_target_aliases(protein: str) -> list[str]:
    """Return optional aliases for short/ambiguous protein symbols.

    Args:
        protein: Protein/gene symbol.

    Returns:
        List of alias strings (may be empty).
    """
    normalized = (protein or "").strip().lower()
    if normalized == "ar":
        return ["androgen receptor", "AR-LBD", "AR-LBP", "NR3C4"]
    return []


def retrieve_evidence_bundle(
    drug: str,
    protein: str,
    *,
    client: Any,
    config: Dict[str, Any],
    index: faiss.Index,
    filtered_chunks: List[Dict[str, Any]],
    assay_terms: str = ASSAY_TERMS_DEFAULT,
    k: int = 80,
    top_pair_pmc: int = 3,
    top_drug_pmc: int = 1,
    top_target_pmc: int = 1,
    chunks_per_pmc: int = 6,
    usage: Optional[TokenUsage] = None,
    min_chunk_words: int = 15,  # Added: default 15 words.
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], str, Dict[str, Any]]:
    """
    Returns:
      - combined chunks with [PAIR]/[DRUG]/[TARGET] prefix in text (each chunk has meta.retrieval and meta.pair_id)
      - pmcid_bundle dict
      - query_text
      - retrieval_meta dict (provenance for the retrieval run)
    """
    assert isinstance(drug, str) and drug.strip(), "drug must be a non-empty string"
    assert (
        isinstance(protein, str) and protein.strip()
    ), "protein must be a non-empty string"

    target_aliases = _expand_target_aliases(protein)
    target_query_base = protein
    if target_aliases:
        if (protein or "").strip().lower() == "ar":
            target_query_base = " OR ".join(target_aliases)
        else:
            target_query_base = protein + " (" + " OR ".join(target_aliases) + ")"

    queries = {
        "pair": [
            f"{drug} ({target_query_base})",
            f"{drug} ({target_query_base}) {assay_terms}",
        ],
        "drug": [
            f"{drug} {assay_terms}",
        ],
        "target": [
            f"({target_query_base}) {assay_terms}",
        ],
    }

    flat_queries = [(cat, q) for cat, qs in queries.items() for q in qs]

    resp = call_with_retry(
        lambda: client.embeddings.create(
            model=config["embedding_deployment"],
            input=[q for _, q in flat_queries],
        )
    )

    if usage is not None:
        usage.add(
            getattr(resp, "usage", None),
            tag="embeddings",
            meta={"n_queries": len(flat_queries)},
        )

    Q = np.array([d.embedding for d in resp.data], dtype=np.float32)
    faiss.normalize_L2(Q)

    D_all, I_all = index.search(Q, k)

    cat_scores: Dict[str, Dict[int, float]] = {cat: {} for cat in queries.keys()}
    for qi, (cat, _) in enumerate(flat_queries):
        for d, idx in zip(D_all[qi], I_all[qi]):
            idx = int(idx)
            prev = cat_scores[cat].get(idx, -1e9)
            cat_scores[cat][idx] = max(prev, float(d))

    top_pair_pmcids = pick_top_pmcids(
        cat_scores["pair"],
        filtered_chunks=filtered_chunks,
        top_pmc=top_pair_pmc,
        drug=drug,
        protein=protein,
        protein_aliases=target_aliases,
        require_drug=True,
        require_protein=True,
    )
    top_drug_pmcids = pick_top_pmcids(
        cat_scores["drug"],
        filtered_chunks=filtered_chunks,
        top_pmc=top_drug_pmc,
        drug=drug,
        require_drug=True,
    )
    top_target_pmcids = pick_top_pmcids(
        cat_scores["target"],
        filtered_chunks=filtered_chunks,
        top_pmc=top_target_pmc,
        protein=protein,
        protein_aliases=target_aliases,
        require_protein=True,
    )

    pmcid_bundle = {
        "pair": top_pair_pmcids,
        "drug": top_drug_pmcids,
        "target": top_target_pmcids,
    }

    pair_chunks: List[Dict[str, Any]] = []
    for pmc in top_pair_pmcids:
        pair_chunks += collect_chunks_for_pmc(
            cat_scores["pair"],
            pmc,
            filtered_chunks=filtered_chunks,
            top_chunks=chunks_per_pmc,
            drug=drug,
            protein=protein,
            protein_aliases=target_aliases,
            require_drug=True,
            require_protein=True,
            min_chunk_words=min_chunk_words,
        )

    drug_chunks: List[Dict[str, Any]] = []
    for pmc in top_drug_pmcids:
        drug_chunks += collect_chunks_for_pmc(
            cat_scores["drug"],
            pmc,
            filtered_chunks=filtered_chunks,
            top_chunks=chunks_per_pmc,
            drug=drug,
            protein=protein,
            protein_aliases=target_aliases,
            require_drug=True,
            min_chunk_words=min_chunk_words,
        )

    target_chunks: List[Dict[str, Any]] = []
    for pmc in top_target_pmcids:
        target_chunks += collect_chunks_for_pmc(
            cat_scores["target"],
            pmc,
            filtered_chunks=filtered_chunks,
            top_chunks=chunks_per_pmc,
            drug=drug,
            protein=protein,
            protein_aliases=target_aliases,
            require_protein=True,
            min_chunk_words=min_chunk_words,
        )

    combined = (
        [{**c, "text": "[PAIR EVIDENCE]\n" + c["text"]} for c in pair_chunks]
        + [{**c, "text": "[DRUG EVIDENCE]\n" + c["text"]} for c in drug_chunks]
        + [{**c, "text": "[TARGET EVIDENCE]\n" + c["text"]} for c in target_chunks]
    )

    # --- Build retrieval metadata for provenance ---
    retrieval_meta: Dict[str, Any] = {
        "retrieval_id": f"retrieval-{int(time.time())}",
        "index_path": INDEX_PATH,
        # index.meta may not exist; try to get a sensible version string
        "index_version": (
            getattr(index, "meta", {}).get("version", "unknown")
            if hasattr(index, "meta")
            else "unknown"
        ),
        "embedding_model": config.get("embedding_deployment"),
        "queries": queries,
        "k": int(k),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pmcid_bundle": pmcid_bundle,
    }

    # attach pair_id and retrieval_meta to every chunk in combined
    norm_drug = normalize_drug(drug)
    norm_protein = normalize_protein(protein)
    pair_id = f"{norm_drug}__{norm_protein}"

    def attach_meta(chunk: Dict[str, Any]) -> Dict[str, Any]:
        c = dict(chunk)  # shallow copy to avoid mutating caller's list
        meta = dict(c.get("meta") or {})
        # attach provenance under meta.retrieval (do not overwrite existing retrieval if present)
        if "retrieval" not in meta:
            meta["retrieval"] = retrieval_meta
        meta["pair_id"] = pair_id
        c["meta"] = meta
        return c

    combined_with_meta = [attach_meta(c) for c in combined]

    def _build_topk(cat_scores, pmc_list, n=4):
        out = []
        for pmc in pmc_list:
            idxs = [
                int(idx)
                for idx in cat_scores.keys()
                if filtered_chunks[int(idx)]["pmc"] == pmc
            ]
            scored = sorted([(float(cat_scores[i]), i) for i in idxs], reverse=True)
            for score, idx in scored[:n]:
                ch = filtered_chunks[idx]
                out.append(
                    {
                        "pmc": pmc,
                        "chunk_idx": idx,
                        "score": float(score),
                        "snippet": ch.get("text", "")[:300].replace("\n", " "),
                    }
                )
        return out

    retrieval_meta["top_k_snippets"] = {
        "pair": _build_topk(cat_scores["pair"], top_pair_pmcids, n=chunks_per_pmc),
        "drug": _build_topk(cat_scores["drug"], top_drug_pmcids, n=chunks_per_pmc),
        "target": _build_topk(
            cat_scores["target"], top_target_pmcids, n=chunks_per_pmc
        ),
    }

    query_text = (
        f"Using drug-only, target-only, and pair excerpts, generate a testable hypothesis "
        f"about whether {drug} could directly interact with {protein}."
    )
    return combined_with_meta, pmcid_bundle, query_text, retrieval_meta


def run_dti_rag(
    drug: str,
    protein: str,
    *,
    client: Any,
    config: Dict[str, Any],
    index: faiss.Index,
    filtered_chunks: List[Dict[str, Any]],
    assay_terms: str = ASSAY_TERMS_DEFAULT,
    k: int = 80,
    top_pair_pmc: int = 3,
    top_drug_pmc: int = 1,
    top_target_pmc: int = 1,
    chunks_per_pmc: int = 6,
    kg_version: str = "unknown",
    max_hops: int = 0,
    max_paths: int = 0,
    avoid_hubs: bool = False,
    hub_degree_cutoff: int = 0,
    topn_paths_for_judge: int = 0,
    cache: Optional[PairCacheSQLite] = None,
    enable_cache: bool = True,
    min_chunk_words: int = 15,
    binary_class: bool = False,
) -> str:
    """
    Convenience wrapper: retrieval + LLM labeling.

    Args:
        binary_class: If True, map labels to Active/Inactive.
    """

    usage = TokenUsage()

    drug_norm = normalize_drug(drug)
    protein_norm = normalize_protein(protein)
    key_hash = ""
    key_json = ""
    if enable_cache:
        cache = cache or PairCacheSQLite()
        key_hash, key_json = make_pair_cache_key(
            drug_norm=drug_norm,
            protein_norm=protein_norm,
            deployment_name=config["deployment_name"],
            kg_version=kg_version,
            max_hops=max_hops,
            max_paths=max_paths,
            avoid_hubs=avoid_hubs,
            hub_degree_cutoff=hub_degree_cutoff,
            topn_paths_for_judge=topn_paths_for_judge,
            assay_terms=assay_terms,
            k=k,
            top_pair_pmc=top_pair_pmc,
            top_drug_pmc=top_drug_pmc,
            top_target_pmc=top_target_pmc,
            chunks_per_pmc=chunks_per_pmc,
            min_chunk_words=min_chunk_words,
        )
        cached = cache.get(key_hash)

        if cached is not None:
            if isinstance(cached, dict):
                cached["token_usage"] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
                if binary_class:
                    _apply_binary_label_to_rag_obj(cached)
                return json.dumps(cached, ensure_ascii=True, sort_keys=True)
            # Defensive fallback.
            return json.dumps(
                {
                    "raw": cached,
                    "token_usage": {
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                },
                ensure_ascii=True,
            )

    combined, pmcid_bundle, query_text, retrieval_meta = retrieve_evidence_bundle(
        drug,
        protein,
        client=client,
        config=config,
        index=index,
        filtered_chunks=filtered_chunks,
        assay_terms=assay_terms,
        k=k,
        top_pair_pmc=top_pair_pmc,
        top_drug_pmc=top_drug_pmc,
        top_target_pmc=top_target_pmc,
        chunks_per_pmc=chunks_per_pmc,
        usage=usage,  # Added.
        min_chunk_words=min_chunk_words,
    )

    pmcid_bundle_full = {
        "pmcid_bundle": pmcid_bundle,
        "retrieval_meta": retrieval_meta,
        "query_text": query_text,
    }
    pmcid_bundle_json = json.dumps(
        pmcid_bundle_full, sort_keys=True, ensure_ascii=False
    )
    deployment_name = config["deployment_name"]
    result = generate_strength_label_science(
        query_text,
        combined,
        pmcid_bundle_json,
        client=client,
        deployment_name=deployment_name,
        usage=usage,
    )

    # result is expected to be a JSON string.
    try:
        obj = json.loads(result)
    except Exception:
        obj = {"raw": result}

    # Ensure top-level obj contains provenance and pair_id for auditability
    if isinstance(obj, dict):
        obj.setdefault("meta", {})
        # attach retrieval_meta and pmcid_bundle (so downstream code can reconstruct evidence provenance)
        obj["meta"]["retrieval"] = retrieval_meta
        obj["meta"]["pmcid_bundle"] = pmcid_bundle
        # canonical pair_id (consistent with chunks)
        norm_drug = normalize_drug(drug)
        norm_protein = normalize_protein(protein)
        obj["meta"]["pair_id"] = f"{norm_drug}__{norm_protein}"

    # --- attach retrieval_combined into obj (safe, truncated, JSON-serializable) ---
    def _sanitize_for_json(x):
        # simple sanitizer: convert numpy numbers, etc., to python builtins; truncate long strings
        import numbers

        if isinstance(x, dict):
            return {str(k): _sanitize_for_json(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_sanitize_for_json(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_sanitize_for_json(v) for v in x)
        if isinstance(x, str):
            return x if len(x) <= 1000 else x[:1000]  # truncate very long strings
        if isinstance(x, numbers.Number):
            try:
                return float(x)
            except Exception:
                return str(x)
        # fallback
        return str(x)

    try:
        # combined is the retrieve_evidence_bundle return (combined_with_meta) passed into run_dti_rag.
        obj.setdefault("meta", {})
        # store only compact info to avoid exploding cache size
        obj["meta"]["retrieval_combined"] = [
            {
                "pmc": c.get("pmc"),
                "chunk_id": c.get("chunk_id") or c.get("id") or None,
                "text_snippet": (
                    c.get("text", "")[:400]
                    if isinstance(c.get("text", ""), str)
                    else str(c.get("text", ""))[:400]
                ),
                "meta": _sanitize_for_json(c.get("meta", {})),
            }
            for c in combined
        ]
        # also ensure retrieval_meta and top_k_snippets in obj.meta are json-safe
        if "retrieval" in obj["meta"]:
            obj["meta"]["retrieval"] = _sanitize_for_json(obj["meta"]["retrieval"])
        else:
            # if retrieval_meta was returned separately, attach it too (if available in locals)
            try:
                obj["meta"]["retrieval"] = _sanitize_for_json(retrieval_meta)
            except Exception:
                pass
    except Exception:
        # defensive: continue even if something goes wrong
        pass
    # --- end attach ---

    # --- New: validate generated quotes against chunks --------------------------------
    # build pmc2chunks mapping from 'combined' (which contains chunk dicts with 'pmc')
    pmc2chunks: Dict[str, List[Dict[str, Any]]] = {}
    for c in combined:
        pmc = c.get("pmc") or c.get("pmcid") or c.get("pmc_id") or "unknown"
        pmc2chunks.setdefault(pmc, []).append(c)

    # Try to use existing matching functions if they exist in the module's scope.
    match_fn = globals().get("match_quote_to_pmc")  # if you have an existing function
    tws_fn = globals().get("token_window_score")  # if you have an existing function

    obj = validate_stage2_quotes(
        obj,
        drug,
        protein,
        pmc2chunks=pmc2chunks,
        match_quote_to_pmc_fn=match_fn,
        token_window_score_fn=tws_fn,
    )
    # -----------------------------------------------------------------------------------

    obj = validate_stage2_output(obj, drug, protein)
    obj = apply_hypothesis_entity_check(obj, drug, protein)

    obj["token_usage"] = {
        "calls": usage.calls,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "breakdown": usage.breakdown,
    }

    # --- FINAL: sanitize obj for JSON, cache, and return ---
    def _final_sanitize(x):
        # reuse the previous sanitizer but stricter for final output
        import math
        import numbers

        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                try:
                    out[str(k)] = _final_sanitize(v)
                except Exception:
                    out[str(k)] = str(v)
            return out
        if isinstance(x, list):
            return [_final_sanitize(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_final_sanitize(v) for v in x)
        if isinstance(x, str):
            return x if len(x) <= 2000 else x[:2000]  # truncate very long strings
        if isinstance(x, bool) or x is None:
            return x
        if isinstance(x, numbers.Number):
            # convert numpy numbers; filter NaN/Inf
            try:
                f = float(x)
                if math.isnan(f) or math.isinf(f):
                    return None
                return f
            except Exception:
                return str(x)
        return str(x)

    # create sanitized copy for cache & return
    safe_obj = _final_sanitize(obj)

    # cache it (PairCacheSQLite.set will json.dumps the dict)
    if enable_cache and cache is not None:
        try:
            cache.set(key_hash=key_hash, key_json=key_json, result=safe_obj)
        except Exception as e:
            # don't fail the whole pipeline on cache write error
            print("[run_dti_rag] cache.set failed:", repr(e), flush=True)
    if binary_class:
        _apply_binary_label_to_rag_obj(safe_obj)

    # return JSON string of sanitized object
    return json.dumps(safe_obj, ensure_ascii=True, sort_keys=True)
    # --- end FINAL ---

    if enable_cache and cache is not None:
        cache.set(key_hash=key_hash, key_json=key_json, result=obj)

    return json.dumps(obj, ensure_ascii=True, sort_keys=True)


# --------------------------------------------------------------------------------------
# Example (keep disabled in library usage)
# --------------------------------------------------------------------------------------
# if __name__ == "__main__":
#     import pandas as pd
#     df = pd.read_csv("sampled_900_v2.csv")
#     DRUG = df["Drug"][0]
#     PROTEIN = df["Protein"][0]
#     result = run_dti_rag(
#         DRUG,
#         PROTEIN,
#         client=client,
#         config=load_azure_openai_config(),
#         index=index,
#         filtered_chunks=filtered_chunks,
#     )
#     print(result)
