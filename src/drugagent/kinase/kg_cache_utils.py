from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any, Dict, Optional, Tuple

DEFAULT_CACHE_PATH = "dti_pair_cache.sqlite3"
CACHE_VERSION = "v4_hypothesis"


class PairCacheSQLite:
    """SQLite-backed cache for (drug, gene) pair results.

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

    def get(self, key_hash: str) -> Optional[Dict[str, Any]]:
        """Fetch cached result by key hash."""
        with sqlite3.connect(self._path) as conn:
            row = conn.execute(
                "SELECT result_json FROM pair_cache WHERE key_hash = ?",
                (key_hash,),
            ).fetchone()
        if not row:
            return None
        try:
            return self._deserialize_result(json.loads(row[0]))
        except Exception:
            return None

    def set(self, key_hash: str, key_json: str, result: Dict[str, Any]) -> None:
        """Insert or replace cached result."""
        result_json = json.dumps(
            self._serialize_result(result), ensure_ascii=True, sort_keys=True
        )
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

    def _serialize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert non-JSON-friendly structures to serializable forms."""
        serialized = dict(result)
        edge_summaries = serialized.get("edge_summaries")
        if isinstance(edge_summaries, dict):
            edge_list = []
            for edge_key, summary in edge_summaries.items():
                if isinstance(edge_key, tuple) and len(edge_key) == 2:
                    edge_list.append(
                        {"edge": [edge_key[0], edge_key[1]], "summary": summary}
                    )
            serialized["edge_summaries"] = edge_list
        path_summaries = serialized.get("path_summaries")
        if isinstance(path_summaries, list):
            for item in path_summaries:
                if (
                    isinstance(item, dict)
                    and "edges" in item
                    and isinstance(item["edges"], list)
                ):
                    item["edges"] = [
                        [e[0], e[1]] if isinstance(e, tuple) and len(e) == 2 else e
                        for e in item["edges"]
                    ]
        return serialized

    def _deserialize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Restore serialized structures back to the original forms."""
        deserialized = dict(result)
        edge_summaries = deserialized.get("edge_summaries")
        if isinstance(edge_summaries, list):
            rebuilt: Dict[Tuple[str, str], Any] = {}
            for item in edge_summaries:
                if isinstance(item, dict) and "edge" in item and "summary" in item:
                    edge = item["edge"]
                    if isinstance(edge, list) and len(edge) == 2:
                        rebuilt[(edge[0], edge[1])] = item["summary"]
            deserialized["edge_summaries"] = rebuilt
        path_summaries = deserialized.get("path_summaries")
        if isinstance(path_summaries, list):
            for item in path_summaries:
                if (
                    isinstance(item, dict)
                    and "edges" in item
                    and isinstance(item["edges"], list)
                ):
                    item["edges"] = [
                        (e[0], e[1]) if isinstance(e, list) and len(e) == 2 else e
                        for e in item["edges"]
                    ]
        return deserialized


def make_pair_cache_key(
    drug_norm: str,
    gene_norm: str,
    deployment_name: str,
    kg_version: str,
    max_hops: int,
    max_paths: int,
    avoid_hubs: bool,
    hub_degree_cutoff: int,
    topn_paths_for_judge: int,
    cache_version: str = CACHE_VERSION,
) -> Tuple[str, str]:
    """Create a stable JSON cache key and its SHA256 hash."""
    key_obj = {
        "avoid_hubs": avoid_hubs,
        "cache_version": cache_version,
        "deployment_name": deployment_name,
        "drug_norm": drug_norm,
        "gene_norm": gene_norm,
        "hub_degree_cutoff": hub_degree_cutoff,
        "kg_version": kg_version,
        "max_hops": max_hops,
        "max_paths": max_paths,
        "topn_paths_for_judge": topn_paths_for_judge,
    }
    key_json = json.dumps(
        key_obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    key_hash = hashlib.sha256(key_json.encode("utf-8")).hexdigest()
    return key_hash, key_json
