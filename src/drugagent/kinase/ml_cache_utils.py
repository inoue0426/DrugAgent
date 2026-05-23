from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Callable, Optional

from diskcache import Cache

from src.utils import get_sequence_from_target_name, get_smiles_from_compound_name

# Configuration constants
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_repo_root() -> Path:
    """Resolve the repository root by walking up to pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[3]


PROJECT_ROOT_DIR = _resolve_repo_root()
CACHE_BASE_DIR = PROJECT_ROOT_DIR / "output"
ML_SCORE_CACHE_DIR = CACHE_BASE_DIR / "ml_dti_scores"
ML_LOOKUP_CACHE_DIR = CACHE_BASE_DIR / "ml_lookup_cache"
SMILES_CACHE_PREFIX = "smiles"
SEQUENCE_CACHE_PREFIX = "sequence"
LOCAL_CAS2DRUG2SMILES_PATH = CURRENT_SCRIPT_DIR / "cas2drug2smiles.csv"
LOCAL_AR2SMILES_PATH = PROJECT_ROOT_DIR / "NR_antagonist" / "AR2SMILES.txt.gz"

_LOCAL_SMILES_MAP: Optional[dict[str, str]] = None
_LOCAL_AR2SMILES_MAP: Optional[dict[str, str]] = None


def init_cache(cache_dir: Path) -> Cache:
    """Initialize a DiskCache directory.

    Args:
        cache_dir: Path to the cache directory.

    Returns:
        A DiskCache Cache instance.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Cache(str(cache_dir))


def cached_lookup(cache: Cache, key: str, fetcher: Callable[[], Optional[str]]) -> Optional[str]:
    """Return a cached value or fetch and store it.

    Args:
        cache: DiskCache instance for storing lookups.
        key: Cache key for the lookup.
        fetcher: Callable that fetches the value if not cached.

    Returns:
        The cached or fetched value, which may be None.
    """
    if key in cache:
        return cache[key]
    value = fetcher()
    cache[key] = value
    return value


def get_cached_smiles(cache: Cache, drug_name: str) -> Optional[str]:
    """Fetch SMILES with caching to avoid repeated lookups.

    Args:
        cache: DiskCache instance for storing SMILES lookups.
        drug_name: Drug name to resolve.

    Returns:
        SMILES string if found, otherwise None.
    """
    cache_key = f"{SMILES_CACHE_PREFIX}:{drug_name}"
    return cached_lookup(cache, cache_key, lambda: get_smiles_from_local_or_global(drug_name))


def get_cached_sequence(cache: Cache, target_name: str) -> Optional[str]:
    """Fetch protein sequence with caching to avoid repeated lookups.

    Args:
        cache: DiskCache instance for storing sequence lookups.
        target_name: Target name to resolve.

    Returns:
        Protein sequence string if found, otherwise None.
    """
    cache_key = f"{SEQUENCE_CACHE_PREFIX}:{target_name}"
    return cached_lookup(cache, cache_key, lambda: get_sequence_from_target_name(target_name))


def get_smiles_from_local_or_global(drug_name: str) -> Optional[str]:
    """Resolve SMILES using local resources before external lookup.

    Args:
        drug_name: Drug name to resolve.

    Returns:
        SMILES string if found, otherwise None.
    """
    local_smiles = lookup_local_ar2smiles(drug_name)
    if local_smiles:
        return local_smiles
    local_smiles = lookup_local_cas2drug2smiles(drug_name)
    if local_smiles:
        return local_smiles
    return get_smiles_from_compound_name(drug_name)


def lookup_local_ar2smiles(drug_name: str) -> Optional[str]:
    """Lookup SMILES in the local AR2SMILES file.

    Args:
        drug_name: Drug name to resolve.

    Returns:
        SMILES string if found, otherwise None.
    """
    if not LOCAL_AR2SMILES_PATH.exists():
        return None

    global _LOCAL_AR2SMILES_MAP
    if _LOCAL_AR2SMILES_MAP is None:
        _LOCAL_AR2SMILES_MAP = {}
        try:
            with gzip.open(LOCAL_AR2SMILES_PATH, "rt", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("	")
                    if len(parts) < 2:
                        continue
                    name = parts[0].strip().lower()
                    smiles = parts[1].strip()
                    if name and smiles and name not in _LOCAL_AR2SMILES_MAP:
                        _LOCAL_AR2SMILES_MAP[name] = smiles
        except Exception:
            _LOCAL_AR2SMILES_MAP = {}
            return None

    return _LOCAL_AR2SMILES_MAP.get(str(drug_name).strip().lower())


def lookup_local_cas2drug2smiles(drug_name: str) -> Optional[str]:
    """Lookup SMILES in the local cas2drug2smiles CSV.

    Args:
        drug_name: Drug name to resolve.

    Returns:
        SMILES string if found, otherwise None.
    """
    if not LOCAL_CAS2DRUG2SMILES_PATH.exists():
        return None

    global _LOCAL_SMILES_MAP
    if _LOCAL_SMILES_MAP is None:
        _LOCAL_SMILES_MAP = {}
        try:
            with LOCAL_CAS2DRUG2SMILES_PATH.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    name = str(row.get("Drug", "")).strip().lower()
                    smiles = str(row.get("SMILES", "")).strip()
                    if name and smiles and name not in _LOCAL_SMILES_MAP:
                        _LOCAL_SMILES_MAP[name] = smiles
        except Exception:
            _LOCAL_SMILES_MAP = {}
            return None

    return _LOCAL_SMILES_MAP.get(str(drug_name).strip().lower())
