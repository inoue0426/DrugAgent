from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
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

from .kg_cache_utils import PairCacheSQLite, make_pair_cache_key
from .kg_graph_utils import (
    Edge,
    _path_edges_from_nodes,
    build_bipartite_index,
    find_paths_drug_to_gene,
    make_path_evidence_text,
    normalize_drug,
    normalize_gene,
    path_hub_meta,
    rank_paths_structural,
)
from .kg_llm_utils import (
    TokenUsage,
    llm_judge_dti_from_paths_azure,
)

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


def _apply_binary_label_to_kg_result(result: Dict[str, Any]) -> None:
    """Apply binary labels to KG judgement in-place.

    Args:
        result: KG result dict with llm_judgement.
    """
    judgement = result.get("llm_judgement")
    if not isinstance(judgement, dict):
        return
    label = judgement.get("label")
    binary = _binary_label_from_strength(label)
    if binary is None or binary == label:
        return
    judgement.setdefault("label_ordinal", label)
    judgement["label"] = binary



def predict_dti_strength_from_kg_paths(
    kg_df: pd.DataFrame,
    drug: str,
    gene: str,
    max_hops: int = 5,
    max_paths: int = 10,
    avoid_hubs: bool = True,
    hub_degree_cutoff: int = 150,
    per_edge_char_cap: int = 800,
    max_total_chars: int = 6000,
) -> Dict[str, Any]:
    """Build evidence text for a drug-gene pair from KG paths.

    Args:
        kg_df: Knowledge graph dataframe.
        drug: Drug name (raw).
        gene: Gene name (raw).
        max_hops: Max hop count in BFS.
        max_paths: Max number of paths to return.
        avoid_hubs: Whether to avoid high-degree nodes.
        hub_degree_cutoff: Degree cutoff for hubs.
        per_edge_char_cap: Per-edge interaction text cap.
        max_total_chars: Max total characters across paths.

    Returns:
        Dict[str, Any]: Query, evidence text, and trace info.
    """
    idx = build_bipartite_index(kg_df)

    paths = find_paths_drug_to_gene(
        idx=idx,
        start_drug=drug,
        target_gene=gene,
        max_hops=max_hops,
        max_paths=max_paths,
        avoid_hubs=avoid_hubs,
        hub_degree_cutoff=hub_degree_cutoff,
    )

    evidence_text, trace = make_path_evidence_text(
        idx=idx,
        paths=paths,
        per_edge_char_cap=per_edge_char_cap,
        max_total_chars=max_total_chars,
    )

    return {
        "query": {
            "drug_raw": drug,
            "gene_raw": gene,
            "drug_norm": normalize_drug(drug),
            "gene_norm": normalize_gene(gene),
        },
        "num_paths_found": len(paths),
        "evidence_text": evidence_text,
        "trace": trace,
        "llm_judgement": None,
    }


# ============================================================
# COMPLETE DTI PIPELINE (KG -> Path -> Judge)
# ============================================================


def predict_dti_strength_full_pipeline(
    kg_df: pd.DataFrame,
    drug: str,
    gene: str,
    client: AzureOpenAI,
    deployment_name: str,
    kg_version: str = "unknown",
    max_hops: int = 5,
    max_paths: int = 20,
    avoid_hubs: bool = False,
    hub_degree_cutoff: int = 150,
    per_edge_char_cap: int = 800,
    topn_paths_for_judge: int = 3,
    cache: Optional[PairCacheSQLite] = None,
    enable_cache: bool = True,
    verbose: bool = False,
    path_summary_char_cap: int = 1600,
    binary_class: bool = False,
) -> Dict[str, Any]:
    """Full DTI pipeline: paths, edge summaries, path summaries, and judgement.

    Args:
        kg_df: Knowledge graph dataframe.
        drug: Drug name (raw).
        gene: Gene name (raw).
        client: Azure OpenAI client.
        deployment_name: Azure deployment name.
        kg_version: KG version label for cache keys.
        max_hops: Max hop count in BFS.
        max_paths: Max number of paths to return.
        avoid_hubs: Whether to avoid high-degree nodes.
        hub_degree_cutoff: Degree cutoff for hubs.
        per_edge_char_cap: Per-edge interaction text cap.
        topn_paths_for_judge: Number of top paths to include in judgement.
        cache: Optional cache instance.
        enable_cache: Whether to use cache.
        verbose: Whether to print per-edge and per-path details.
        path_summary_char_cap: Max characters for each path summary.
        binary_class: If True, map labels to Active/Inactive.

    Returns:
        Dict[str, Any]: Full structured result.
    """
    usage = TokenUsage()
    debug_messages: List[str] = []

    def log(message: str) -> None:
        if verbose:
            debug_messages.append(message)
            print(message)

    drug_norm = normalize_drug(drug)
    gene_norm = normalize_gene(gene)
    key_hash = ""
    key_json = ""
    if enable_cache:
        cache = cache or PairCacheSQLite()
        key_hash, key_json = make_pair_cache_key(
            drug_norm=drug_norm,
            gene_norm=gene_norm,
            deployment_name=deployment_name,
            kg_version=kg_version,
            max_hops=max_hops,
            max_paths=max_paths,
            avoid_hubs=avoid_hubs,
            hub_degree_cutoff=hub_degree_cutoff,
            topn_paths_for_judge=topn_paths_for_judge,
        )
        cached = cache.get(key_hash)
        if cached is not None:
            cached["cache"] = {"hit": True}
            if binary_class:
                _apply_binary_label_to_kg_result(cached)
            return cached

    idx = build_bipartite_index(kg_df)

    paths = find_paths_drug_to_gene(
        idx=idx,
        start_drug=drug,
        target_gene=gene,
        max_hops=max_hops,
        max_paths=max_paths,
        avoid_hubs=avoid_hubs,
        hub_degree_cutoff=hub_degree_cutoff,
    )

    if len(paths) == 0:
        log("No connecting paths found.")
        result = {
            "query": {
                "drug_raw": drug,
                "gene_raw": gene,
                "drug_norm": drug_norm,
                "gene_norm": gene_norm,
            },
            "num_paths_found": 0,
            "edge_summaries": {},
            "path_summaries": [],
            "llm_judgement": {
                "label": "Insufficient",
                "confidence": 0.01,
                "rationale": ["No connecting paths found in KG."],
                "supporting_paths": [],
                "failure_modes": ["no_path"],
            },
        }
        if binary_class:
            result["llm_judgement"]["label_ordinal"] = result["llm_judgement"]["label"]
            result["llm_judgement"]["label"] = "Inactive"
        result["cache"] = {"hit": False} if enable_cache else {"hit": None}
        if enable_cache and cache is not None:
            cache.set(key_hash=key_hash, key_json=key_json, result=result)
        return result

    ranked = rank_paths_structural(
        idx=idx,
        paths=paths,
        hub_degree_cutoff=hub_degree_cutoff,
        avoid_hubs=avoid_hubs,
    )

    preselect_k = min(len(ranked), topn_paths_for_judge)
    selected_ids = [pid for (pid, _) in ranked[:preselect_k]]
    log(f"Selected path IDs (preselect_k={preselect_k}): {selected_ids}")

    selected_edges: Set[Edge] = set()
    for pid in selected_ids:
        path = paths[pid - 1]
        selected_edges.update(_path_edges_from_nodes(path))

    edge_summaries: Dict[Edge, Dict[str, Any]] = {}
    for e in sorted(selected_edges):
        row_id = idx.edge2row.get(e)
        if row_id is None:
            continue

        edge_info = {
            "drug_raw": idx.kg.loc[row_id, "Drug"],
            "drug_norm": e[0],
            "gene_raw": idx.kg.loc[row_id, "Gene"],
            "gene_norm": e[1],
            "interaction_text": idx.edge2text[e],
            **idx.edge2meta[e],
        }
        log(f"Edge {e} info: {edge_info}")
        edge_summaries[e] = edge_info

    path_summaries = []
    for pid in selected_ids:
        path = paths[pid - 1]
        edge_list = _path_edges_from_nodes(path)

        edge_info_list = [edge_summaries[e] for e in edge_list if e in edge_summaries]
        if len(edge_info_list) != len(edge_list):
            continue

        edge_texts = []
        for info in edge_info_list:
            txt = str(info.get("interaction_text", "") or "").strip()
            if per_edge_char_cap and len(txt) > per_edge_char_cap:
                txt = txt[:per_edge_char_cap].rstrip() + " ...[truncated]"
            edge_texts.append(f"{info.get('drug_norm')} <-> {info.get('gene_norm')}: {txt}")

        path_str = " -> ".join([f"{typ}:{val}" for typ, val in path])
        path_summary = f"{path_str}\n" + "\n".join(edge_texts)
        if path_summary_char_cap and len(path_summary) > path_summary_char_cap:
            path_summary = path_summary[:path_summary_char_cap].rstrip() + " ...[truncated]"

        hub_meta = path_hub_meta(idx=idx, path=path, hub_degree_cutoff=hub_degree_cutoff)
        intermediates = [
            val
            for typ, val in path[1:-1]
            if isinstance(val, str) and val and typ in {"Drug", "Gene"}
        ]
        key_intermediates = list(dict.fromkeys(intermediates))

        path_summaries.append(
            {
                "path_id": pid,
                "length": len(edge_list),
                "edges": edge_list,
                "hub_meta": hub_meta,
                "path_summary": path_summary,
                "key_edges": edge_list,
                "key_intermediates": key_intermediates,
                "process_tags": [],
            }
        )
        log(f"Path {pid} edges: {edge_list}")
        log(f"Path {pid} summary: {path_summaries[-1]}")

    path_summaries_for_judge = path_summaries[:topn_paths_for_judge]

    path_summaries = sorted(path_summaries, key=lambda x: x.get("path_id", 10**9))

    judgement = llm_judge_dti_from_paths_azure(
        drug=drug,
        gene=gene,
        path_summaries=path_summaries_for_judge,
        client=client,
        deployment_name=deployment_name,
        topn=topn_paths_for_judge,
        usage=usage,
        binary_class=binary_class,
    )

    try:
        if path_summaries_for_judge:
            hub_heavy = sum(
                1
                for p in path_summaries_for_judge
                if (p.get("hub_meta", {}).get("hub_hits", 0) or 0) > 0
            )
            if hub_heavy >= max(1, len(path_summaries_for_judge) // 2):
                fm = set(judgement.get("failure_modes", []) or [])
                fm.add("hub_path")
                judgement["failure_modes"] = sorted(fm)
    except Exception:
        pass

    result = {
        "query": {
            "drug_raw": drug,
            "gene_raw": gene,
            "drug_norm": drug_norm,
            "gene_norm": gene_norm,
        },
        "num_paths_found": len(paths),
        "edge_summaries": edge_summaries,
        "path_summaries": path_summaries,
        "llm_judgement": judgement,
    }
    if verbose:
        result["debug_messages"] = debug_messages
    result["cache"] = {"hit": False} if enable_cache else {"hit": None}
    result["token_usage"] = {
        "calls": usage.calls,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "breakdown": usage.breakdown,
    }
    failure_modes = set(result["llm_judgement"].get("failure_modes", []) or [])

    do_cache = True
    if "parse_error" in failure_modes:
        do_cache = False

    if enable_cache and cache is not None and do_cache:
        cache.set(key_hash=key_hash, key_json=key_json, result=result)
    if binary_class:
        _apply_binary_label_to_kg_result(result)
    return result
