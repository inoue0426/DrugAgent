#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np
import pandas as pd

# =========================
# Utilities
# =========================


def clean_ic50_series(s: pd.Series) -> pd.Series:
    """IC50列（文字列含む）を数値に正規化。<, > を除去して numeric 化。"""
    s = s.astype("string")
    s = s.str.replace("<", "", regex=False).str.replace(">", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def safe_edge_weight_from_ic50(ic50_nm: pd.Series) -> pd.Series:
    """
    edge_weight = -log10(IC50_nM)
    IC50<=0, NaN, inf を安全に処理
    """
    x = ic50_nm.astype("float64")
    w = -np.log10(x)
    w[~np.isfinite(w)] = np.nan
    return pd.Series(w, index=ic50_nm.index, dtype="float64")


# =========================
# KG Builder
# =========================


@dataclass(frozen=True)
class KGArtifacts:
    edges: pd.DataFrame
    drug_nodes: pd.DataFrame
    target_nodes: pd.DataFrame


class KGBuilder:
    """
    train dataframe から、(drug_key=monomer:ID)-(target_key=uniprot:ID) の
    bipartite edge/node を作るビルダー。
    """

    def __init__(self, edge_cols: Optional[Iterable[str]] = None):
        self.edge_cols = (
            list(edge_cols)
            if edge_cols is not None
            else [
                "drug_key",
                "target_key",
                "MonomerID",
                "UniProtID",
                "ReactantSetID",
                "DrugBankID",
                "PubChemCID",
                "PubChemSID",
                "ChEBI",
                "ChEMBL",
                "ZINC",
                "IUPHAR",
                "KEGG",
                "SMILES",
                "InChI",
                "InChIKey",
                "LigandName",
                "TargetName",
                "EntryName",
                "Sequence",
                "IC50_nM",
                "Ki_nM",
                "Kd_nM",
                "EC50_nM",
                "pH",
                "Temp_C",
                "class",
                "label",
                "PublicationDate",
                "BindingDBDate",
                "year",
                "PMID",
                "Patent",
                "DataSource",
            ]
        )

    def build_from_train(
        self,
        df_train: pd.DataFrame,
        label_value: str = "active",
        source: str = "BindingDB_train_active",
        edge_type: str = "binds",
        make_edge_weight: bool = True,
    ) -> KGArtifacts:
        df = df_train.copy()

        kg_src = df[df["label"].eq(label_value)].copy()
        kg_src["MonomerID"] = pd.to_numeric(
            kg_src["MonomerID"], errors="coerce"
        ).astype("Int64")
        kg_src["UniProtID"] = kg_src["UniProtID"].astype("string").str.strip()
        kg_src = kg_src.dropna(subset=["MonomerID", "UniProtID"]).copy()

        kg_src["drug_key"] = "monomer:" + kg_src["MonomerID"].astype(str)
        kg_src["target_key"] = "uniprot:" + kg_src["UniProtID"].astype(str)

        edge_cols = [c for c in self.edge_cols if c in kg_src.columns]
        edges = kg_src[edge_cols].copy()

        # IC50 cleaning + optional edge_weight
        if "IC50_nM" in edges.columns:
            edges["IC50_nM"] = clean_ic50_series(edges["IC50_nM"])
            if make_edge_weight:
                edges["edge_weight"] = safe_edge_weight_from_ic50(edges["IC50_nM"])

        edges["source"] = source
        edges["edge_type"] = edge_type

        # Nodes
        drug_node_cols = [
            c
            for c in [
                "drug_key",
                "MonomerID",
                "DrugBankID",
                "InChIKey",
                "PubChemCID",
                "ChEMBL",
                "SMILES",
                "InChI",
                "LigandName",
            ]
            if c in edges.columns
        ]
        drug_nodes = (
            edges[drug_node_cols]
            .drop_duplicates(subset=["drug_key"])
            .reset_index(drop=True)
        )

        target_node_cols = [
            c
            for c in ["target_key", "UniProtID", "EntryName", "TargetName", "Sequence"]
            if c in edges.columns
        ]
        target_nodes = (
            edges[target_node_cols]
            .drop_duplicates(subset=["target_key"])
            .reset_index(drop=True)
        )

        return KGArtifacts(
            edges=edges, drug_nodes=drug_nodes, target_nodes=target_nodes
        )

    @staticmethod
    def save_parquet(
        artifacts: KGArtifacts,
        edges_path: str,
        drug_nodes_path: str,
        target_nodes_path: str,
    ) -> None:
        artifacts.edges.to_parquet(edges_path, index=False)
        artifacts.drug_nodes.to_parquet(drug_nodes_path, index=False)
        artifacts.target_nodes.to_parquet(target_nodes_path, index=False)


# =========================
# Context Generator
# =========================


class ContextGenerator:
    """
    edges + nodes から各種統計・隣接辞書を作って、(drug_key, target_key) の説明文を生成。
    shortest path は scipy ではなく BFS で「必要なペアだけ」計算（早期打ち切り）して高速化。
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        edges: pd.DataFrame,
        target_nodes: pd.DataFrame,
        max_bfs_depth: int = 6,
    ):
        self.df_train = df_train
        self.edges = edges
        self.target_nodes = target_nodes
        self.max_bfs_depth = max_bfs_depth

        # --- degree (unweighted) ---
        self.deg_target = self.edges["target_key"].value_counts()
        self.deg_drug = self.edges["drug_key"].value_counts()
        self.target_pct = self.deg_target.rank(pct=True)

        # --- IC50 stats ---
        if "IC50_nM" in self.edges.columns:
            self.target_ic50_median = self.edges.groupby("target_key")[
                "IC50_nM"
            ].median()
            self.target_ic50_min = self.edges.groupby("target_key")["IC50_nM"].min()
        else:
            self.target_ic50_median = pd.Series(dtype="float64")
            self.target_ic50_min = pd.Series(dtype="float64")

        # --- adjacency dicts (2-hop etc.) ---
        self.drug_to_targets = (
            self.edges.groupby("drug_key")["target_key"]
            .apply(lambda x: set(x.values))
            .to_dict()
        )
        self.target_to_drugs = (
            self.edges.groupby("target_key")["drug_key"]
            .apply(lambda x: set(x.values))
            .to_dict()
        )

        # --- name/meta maps ---
        self.drug_name_map = (
            self.df_train.groupby("MonomerID")["name"].first().to_dict()
        )
        self.target_meta_map = self.target_nodes.set_index("target_key").to_dict(
            orient="index"
        )

        # --- shortest path cache: (drug_key,target_key)->dist ---
        self._sp_cache: Dict[Tuple[str, str], Optional[int]] = {}

    def _bfs_shortest_path_bipartite(
        self, drug_key: str, target_key: str
    ) -> Optional[int]:
        """
        bipartite graph on-the-fly BFS:
        drug -> targets -> drugs -> targets ...
        目的 target_key に到達した最短 hop 長（drug->target が 1）を返す。
        見つからなければ None。
        """
        cache_key = (drug_key, target_key)
        if cache_key in self._sp_cache:
            return self._sp_cache[cache_key]

        if (
            drug_key not in self.drug_to_targets
            or target_key not in self.target_to_drugs
        ):
            self._sp_cache[cache_key] = None
            return None

        # BFS state: node is ("drug", key) or ("target", key)
        start = ("drug", drug_key)
        goal = ("target", target_key)

        q = deque([(start, 0)])
        seen = {start}

        while q:
            (kind, key), dist = q.popleft()
            if dist >= self.max_bfs_depth:
                continue

            if (kind, key) == goal:
                self._sp_cache[cache_key] = dist
                return dist

            if kind == "drug":
                # next: targets
                for t in self.drug_to_targets.get(key, ()):
                    nxt = ("target", t)
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append((nxt, dist + 1))
            else:
                # kind == "target" -> next: drugs
                for d in self.target_to_drugs.get(key, ()):
                    nxt = ("drug", d)
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append((nxt, dist + 1))

        self._sp_cache[cache_key] = None
        return None

    @staticmethod
    def _infer_protein_class(gene_name: Optional[str]) -> Optional[str]:
        if not gene_name:
            return None
        if gene_name.endswith("K"):
            return "kinase"
        if gene_name.endswith("R"):
            return "receptor"
        return None

    def generate(self, drug_key: str, target_key: str) -> str:
        lines = []

        # ========= Drug info =========
        try:
            monomer_id = int(drug_key.split(":")[1])
        except Exception:
            monomer_id = None

        drug_name = (
            self.drug_name_map.get(monomer_id, drug_key)
            if monomer_id is not None
            else drug_key
        )

        # ========= Target info =========
        t_meta = self.target_meta_map.get(target_key, {})
        protein_name = (
            t_meta.get("TargetName") or t_meta.get("target_name") or target_key
        )
        gene_name = t_meta.get("gene_name")
        protein_class = self._infer_protein_class(gene_name)
        entry_name = t_meta.get("EntryName")
        func_hint = t_meta.get("target_function_hint")

        lines.append(f"Drug: {drug_name} ({drug_key})")
        if gene_name:
            if protein_class:
                lines.append(
                    f"Target: {protein_name} ({gene_name}, {protein_class}, {target_key})"
                )
            else:
                lines.append(f"Target: {protein_name} ({gene_name}, {target_key})")
        else:
            lines.append(f"Target: {protein_name} ({target_key})")

        if entry_name:
            lines.append(f"EntryName: {entry_name}")
        lines.append("")

        # ========= Direct interaction =========
        has_direct = target_key in self.drug_to_targets.get(drug_key, set())
        lines.append(f"Direct interaction in training graph: {has_direct}.")
        lines.append("")

        # ========= Connectivity =========
        t_deg = int(self.deg_target.get(target_key, 0))
        d_deg = int(self.deg_drug.get(drug_key, 0))
        t_pct = float(self.target_pct.get(target_key, 0)) * 100.0

        lines.append("Connectivity:")
        lines.append(
            f"- Target connected to {t_deg} drugs (top {t_pct:.1f}% centrality)."
        )
        lines.append(f"- Drug binds {d_deg} target(s).")

        if t_pct > 95:
            lines.append("- The target is a highly promiscuous hub.")
        elif t_pct > 70:
            lines.append("- The target shows moderate connectivity.")
        else:
            lines.append("- The target is relatively selective.")

        if d_deg <= 1:
            lines.append("- The drug appears selective.")
        elif d_deg > 5:
            lines.append("- The drug appears promiscuous.")
        lines.append("")

        # ========= Shared neighbors (as in original; note: cross-type overlap may be 0 by construction) =========
        drug_targets = self.drug_to_targets.get(drug_key, set())
        target_drugs = self.target_to_drugs.get(target_key, set())

        # 元コードは drug_targets & target_drugs を取っていたが、
        # 片方は target集合、片方は drug集合なので通常 0 になりやすい。
        # ここでは「2-hop根拠」として、targetの近傍drugが持つtarget集合の重なりを計算する例を入れる。
        shared_2hop_targets: Set[str] = set()
        for d in target_drugs:
            shared_2hop_targets |= self.drug_to_targets.get(d, set())
        shared_2hop_targets.discard(target_key)

        overlap = len(drug_targets & shared_2hop_targets)
        overlap_ratio = overlap / max(len(drug_targets), 1)

        lines.append("Network proximity:")
        lines.append(
            f"- 2-hop shared targets via target-neighbor drugs: {overlap} (ratio vs drug targets: {overlap_ratio:.4f})."
        )

        if overlap == 0:
            lines.append("- No 2-hop network support found.")
        elif overlap < 2:
            lines.append("- Minimal 2-hop network support.")
        else:
            lines.append("- Some 2-hop network support.")
        lines.append("")

        # ========= Shortest path (BFS) =========
        sp = self._bfs_shortest_path_bipartite(drug_key, target_key)
        if sp is None:
            lines.append(
                "Shortest path (unweighted, bipartite BFS): no path (or beyond depth limit)."
            )
        else:
            if sp == 1:
                lines.append("Shortest path (unweighted, bipartite BFS): 1 (direct).")
            elif sp == 2:
                lines.append(
                    "Shortest path (unweighted, bipartite BFS): 2 (very close)."
                )
            elif sp == 3:
                lines.append("Shortest path (unweighted, bipartite BFS): 3 (moderate).")
            else:
                lines.append(f"Shortest path (unweighted, bipartite BFS): {sp}.")
        lines.append("")

        # ========= IC50 profile =========
        median_ic50 = self.target_ic50_median.get(target_key, np.nan)
        min_ic50 = self.target_ic50_min.get(target_key, np.nan)

        lines.append("Target bioactivity profile:")
        lines.append(
            f"- Median IC50 across drugs: {median_ic50 if np.isfinite(median_ic50) else 'N/A'} nM."
        )
        lines.append(
            f"- Strongest known IC50: {min_ic50 if np.isfinite(min_ic50) else 'N/A'} nM."
        )

        if np.isfinite(median_ic50):
            if median_ic50 < 10:
                lines.append("- The target is highly druggable.")
            elif median_ic50 < 100:
                lines.append("- The target shows moderate druggability.")
            else:
                lines.append("- The target shows weak druggability.")

        # ========= Functional hint =========
        if func_hint:
            lines.append("")
            lines.append("Functional context:")
            lines.append(f"- {func_hint}")

        # ========= Interpretation =========
        lines.append("")
        lines.append("Interpretation context:")
        lines.append(
            "- Direct evidence supports interaction."
            if has_direct
            else "- No direct evidence in training data."
        )
        if overlap == 0:
            lines.append("- No strong 2-hop network support for this specific pair.")
        elif overlap < 2:
            lines.append("- Weak 2-hop network support for this specific pair.")
        else:
            lines.append("- Moderate 2-hop network support for this specific pair.")

        return "\n".join(lines)


# =========================
# Example usage
# =========================


def main():
    # df_train = pd.read_csv('../data/train.csv.gz', index_col=0)
    # ↑は環境に合わせて
    raise SystemExit("Set df_train loading path and call builder/generator as needed.")


if __name__ == "__main__":
    main()
