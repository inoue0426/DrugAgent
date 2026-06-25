from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from common_utils import normalize_drug, normalize_gene

Node = Tuple[str, str]  # ("Drug"|"Gene", value)
Edge = Tuple[str, str]  # (drug, gene) normalized keys


@dataclass
class KGIndex:
    kg: pd.DataFrame
    drug2genes: Dict[str, Set[str]]
    gene2drugs: Dict[str, Set[str]]
    edge2row: Dict[Edge, int]
    edge2text: Dict[Edge, str]
    edge2meta: Dict[Edge, Dict[str, Any]]
    drug_degree: Dict[str, int]
    gene_degree: Dict[str, int]


def _path_edges_from_nodes(path: List[Node]) -> List[Edge]:
    edges: List[Edge] = []
    for a, b in zip(path, path[1:]):
        if a[0] == "Drug" and b[0] == "Gene":
            edges.append((a[1], b[1]))
        else:
            edges.append((b[1], a[1]))
    return edges


def rank_paths_structural(
    idx: KGIndex,
    paths: List[List[Node]],
    hub_degree_cutoff: int = 300,
    avoid_hubs: bool = True,
    lambda_val: float = 0.6,
) -> List[Tuple[int, float]]:
    ranked = []

    for i, path in enumerate(paths, start=1):
        n_edges = max(len(path) - 1, 1)
        score = 1.0 / n_edges

        hub_hits = 0
        if avoid_hubs and len(path) > 2:
            for typ, val in path[1:-1]:
                deg = (
                    idx.drug_degree.get(val, 0)
                    if typ == "Drug"
                    else idx.gene_degree.get(val, 0)
                )
                if deg >= hub_degree_cutoff:
                    hub_hits += 1

        score *= lambda_val**hub_hits
        ranked.append((i, score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def path_hub_meta(
    idx: KGIndex, path: List[Node], hub_degree_cutoff: int = 150
) -> Dict[str, Any]:
    """
    Compute hub-related metadata for a path (used for LLM interpretation, not for pruning).
    """
    max_deg = 0
    max_node: Optional[Node] = None
    hub_hits = 0
    hub_nodes: List[Dict[str, Any]] = []

    for typ, val in path[1:-1]:
        deg = (
            idx.drug_degree.get(val, 0)
            if typ == "Drug"
            else idx.gene_degree.get(val, 0)
        )
        if deg > max_deg:
            max_deg = deg
            max_node = (typ, val)
        if deg >= hub_degree_cutoff:
            hub_hits += 1
            hub_nodes.append({"type": typ, "value": val, "degree": int(deg)})

    return {
        "hub_degree_cutoff": hub_degree_cutoff,
        "hub_hits": int(hub_hits),
        "hub_nodes": hub_nodes,
        "max_intermediate_degree": int(max_deg),
        "max_degree_node": (
            {"type": max_node[0], "value": max_node[1]} if max_node else None
        ),
    }


def build_bipartite_index(
    kg_df: pd.DataFrame,
    drug_col: str = "Drug",
    gene_col: str = "Gene",
    interaction_col: str = "Interaction",
    meta_cols: Optional[List[str]] = None,
) -> KGIndex:
    """
    Build fast adjacency + edge lookup from a KG dataframe whose rows represent Drug-Gene edges.
    """
    if meta_cols is None:
        meta_cols = [
            c for c in kg_df.columns if c not in {drug_col, gene_col, interaction_col}
        ]

    drug2genes: Dict[str, Set[str]] = defaultdict(set)
    gene2drugs: Dict[str, Set[str]] = defaultdict(set)
    edge2row: Dict[Edge, int] = {}
    edge2text: Dict[Edge, str] = {}
    edge2meta: Dict[Edge, Dict[str, Any]] = {}

    for ridx, row in kg_df.iterrows():
        d_raw = row.get(drug_col, "")
        g_raw = row.get(gene_col, "")
        d = normalize_drug(d_raw)
        g = normalize_gene(g_raw)
        if not d or not g:
            continue

        e: Edge = (d, g)
        if e in edge2row:
            continue

        drug2genes[d].add(g)
        gene2drugs[g].add(d)

        edge2row[e] = int(ridx) if isinstance(ridx, (int,)) else ridx
        edge2text[e] = (
            ""
            if pd.isna(row.get(interaction_col, ""))
            else str(row.get(interaction_col, ""))
        )
        edge2meta[e] = {c: row.get(c, None) for c in meta_cols}

    drug_degree = {d: len(gs) for d, gs in drug2genes.items()}
    gene_degree = {g: len(ds) for g, ds in gene2drugs.items()}

    return KGIndex(
        kg=kg_df,
        drug2genes=dict(drug2genes),
        gene2drugs=dict(gene2drugs),
        edge2row=edge2row,
        edge2text=edge2text,
        edge2meta=edge2meta,
        drug_degree=drug_degree,
        gene_degree=gene_degree,
    )


def find_paths_drug_to_gene(
    idx: KGIndex,
    start_drug: str,
    target_gene: str,
    max_hops: int = 5,
    max_paths: int = 20,
    avoid_hubs: bool = True,
    hub_degree_cutoff: int = 150,
) -> List[List[Node]]:
    """
    Find simple paths from Drug(start_drug) to Gene(target_gene) on a bipartite graph.
    """
    s = normalize_drug(start_drug)
    t = normalize_gene(target_gene)

    if not s or not t:
        return []

    if s not in idx.drug2genes:
        return []

    start: Node = ("Drug", s)
    q = deque([(start, [start], 0)])
    found: List[List[Node]] = []

    def is_hub(node: Node) -> bool:
        if not avoid_hubs:
            return False
        typ, val = node
        if typ == "Drug":
            return idx.drug_degree.get(val, 0) >= hub_degree_cutoff
        return idx.gene_degree.get(val, 0) >= hub_degree_cutoff

    while q and len(found) < max_paths:
        cur, path, depth = q.popleft()
        if depth >= max_hops:
            continue

        cur_type, cur_val = cur

        if cur_type == "Drug":
            nbrs = sorted(idx.drug2genes.get(cur_val, []))
            for g in nbrs:
                nxt: Node = ("Gene", g)
                if nxt in path:
                    continue
                if is_hub(nxt) and g != t:
                    continue

                new_path = path + [nxt]
                new_depth = depth + 1

                if g == t:
                    found.append(new_path)
                    if len(found) >= max_paths:
                        break
                else:
                    q.append((nxt, new_path, new_depth))
        else:
            nbrs = sorted(idx.gene2drugs.get(cur_val, []))
            for d in nbrs:
                nxt = ("Drug", d)
                if nxt in path:
                    continue
                if is_hub(nxt):
                    continue
                q.append((nxt, path + [nxt], depth + 1))

    return found


def make_path_evidence_text(
    idx: KGIndex,
    paths: List[List[Node]],
    per_edge_char_cap: int = 800,
    max_total_chars: int = 6000,
    include_meta_keys: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Convert paths into a single evidence text string for LLM."""
    if include_meta_keys is None:
        include_meta_keys = [
            "primary_species",
            "has_human",
            "has_drugbank",
            "has_ctd",
            "interaction_n_sent",
            "interaction_token_est",
        ]

    chunks: List[str] = []
    trace_paths: List[Dict[str, Any]] = []
    total = 0

    for i, path in enumerate(paths, start=1):
        edges: List[Edge] = []
        row_ids: List[Any] = []
        edge_blocks: List[str] = []

        for (a_type, a_val), (b_type, b_val) in zip(path, path[1:]):
            if a_type == "Drug" and b_type == "Gene":
                e = (a_val, b_val)
            elif a_type == "Gene" and b_type == "Drug":
                e = (b_val, a_val)
            else:
                continue

            edges.append(e)
            row_id = idx.edge2row.get(e, None)
            row_ids.append(row_id)

            txt = idx.edge2text.get(e, "").strip()
            if per_edge_char_cap and len(txt) > per_edge_char_cap:
                txt = txt[:per_edge_char_cap].rstrip() + " ...[truncated]"

            meta = idx.edge2meta.get(e, {})
            meta_line_parts = []
            for k in include_meta_keys:
                val = meta.get(k)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                meta_line_parts.append(f"{k}={val}")
            meta_line = (" | " + ", ".join(meta_line_parts)) if meta_line_parts else ""

            txt_indented = txt.replace("\n", "\n  ")
            edge_blocks.append(
                f"- Edge: Drug='{e[0]}' <-> Gene='{e[1]}' (row_id={row_id}){meta_line}\n"
                f"  Interaction:\n"
                f"  {txt_indented}"
            )

        path_str = " -> ".join([f"{typ}:{val}" for typ, val in path])
        block = f"### Path #{i}\n{path_str}\n{chr(10).join(edge_blocks)}\n"

        if total + len(block) > max_total_chars:
            break

        chunks.append(block)
        total += len(block)
        trace_paths.append(
            {
                "path_id": i,
                "nodes": path,
                "edges": edges,
                "row_ids": row_ids,
            }
        )

    evidence_text = "\n".join(chunks).strip()
    trace = {
        "num_paths_provided": len(trace_paths),
        "paths": trace_paths,
        "max_total_chars": max_total_chars,
        "per_edge_char_cap": per_edge_char_cap,
    }
    return evidence_text, trace
