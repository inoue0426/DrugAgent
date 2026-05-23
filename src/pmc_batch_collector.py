"""
Refactored + Accelerated PMC batch collector (v2)

What this does
- PubMed ESearch (pair / drug-only / gene-only) with caching + backoff
- PMID -> PMCID via NCBI idconv (cached)
- Download PMC Open Access XML from S3 (pmc-oa-opendata) with local cache
- NEW: Adaptive retrieval for single-term (drug / gene):
    "download at least K XMLs" by paging (retstart) instead of re-fetching top-N
- Faster: reuse a singleton boto3 S3 client, avoid redundant exists checks
- Optional: sort="pub date" to bias toward newer OA-heavy papers (often improves yield)

Notes
- downloaded = available in output_dir (new + already existed)
- You can still use fixed-top-N behavior by setting k_single=0 and max_search_results_single>0
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from diskcache import Cache
from dotenv import load_dotenv

# -------------------------
# Config / globals
# -------------------------

load_dotenv()
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PMC_ID_CONVERT_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

DEFAULT_EMAIL = os.environ.get("NCBI_EMAIL", "inouey2@nih.gov")

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("src.pmc_batch_collector")

# Cache
os.makedirs("cache", exist_ok=True)
cache = Cache("cache")

# Reuse S3 client (big speedup)
_S3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))


@dataclass(frozen=True)
class ESearchResult:
    ok: bool
    pmids: List[str]
    term: str
    error: Optional[str] = None


# -------------------------
# Utility: polite HTTP
# -------------------------


def _sleep_jitter(base_sec: float, jitter_sec: float = 0.15) -> None:
    """Small jitter helps avoid bursts when batching."""
    if base_sec <= 0:
        return
    time.sleep(base_sec + (jitter_sec * (os.getpid() % 7) / 7.0))


def http_get_with_backoff(
    url: str,
    params: Dict[str, Any],
    timeout: Tuple[int, int] = (8, 30),
    max_retries: int = 5,
    base_wait_sec: float = 0.5,
    respect_retry_after: bool = True,
) -> requests.Response:
    """
    requests.get with exponential backoff for 429/5xx/network errors.

    - For 429: uses Retry-After if present (optional), else exponential backoff.
    - For 5xx/timeouts/connection errors: exponential backoff.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)

            # Handle 429 explicitly
            if resp.status_code == 429:
                wait = base_wait_sec * (2**attempt)
                if respect_retry_after:
                    ra = resp.headers.get("Retry-After")
                    if ra:
                        try:
                            wait = max(wait, float(ra))
                        except ValueError:
                            pass
                logger.warning(
                    f"HTTP 429 for {url} (attempt {attempt+1}/{max_retries}); sleeping {wait:.2f}s"
                )
                _sleep_jitter(wait)
                continue

            # Retry 5xx
            if 500 <= resp.status_code < 600:
                wait = base_wait_sec * (2**attempt)
                logger.warning(
                    f"HTTP {resp.status_code} for {url} (attempt {attempt+1}/{max_retries}); sleeping {wait:.2f}s"
                )
                _sleep_jitter(wait)
                continue

            resp.raise_for_status()
            return resp

        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_exc = e
            wait = base_wait_sec * (2**attempt)
            logger.warning(
                f"HTTP error for {url} (attempt {attempt+1}/{max_retries}): {e}; sleeping {wait:.2f}s"
            )
            _sleep_jitter(wait)

    raise RuntimeError(
        f"GET failed after {max_retries} retries: {url}; last={last_exc!r}"
    )


# -------------------------
# PubMed / PMC ID conversion
# -------------------------


def check_pmcid(pmid: str, email: str = DEFAULT_EMAIL) -> Optional[str]:
    """
    Convert PMID -> PMCID (if available), with caching.
    """
    key = f"pmcid_{pmid}"
    if key in cache:
        return cache[key]

    params = {"ids": pmid, "format": "json", "tool": "return_pmcid", "email": email}

    try:
        resp = http_get_with_backoff(
            PMC_ID_CONVERT_URL,
            params=params,
            timeout=(3, 10),
            max_retries=4,
            base_wait_sec=0.5,
        )
        data = resp.json()
        records = data.get("records", [])
        pmcid = records[0].get("pmcid") if records else None
        cache[key] = pmcid
        return pmcid
    except Exception as e:
        logger.error(f"Error looking up PMCID for PMID {pmid}: {e}")
        cache[key] = None
        return None


def pubmed_esearch(
    term: str,
    retstart: int = 0,
    retmax: int = 50,
    maxdate: Optional[str] = None,
    api_key: Optional[str] = None,
    sort: str = "pub date",
    polite_delay_sec: float = 0.0,
) -> ESearchResult:
    cache_key = f"esearch::{term}::retstart={retstart}::retmax={retmax}::maxdate={maxdate}::sort={sort}"
    cached = cache.get(cache_key)
    if cached is not None:
        return ESearchResult(
            ok=cached["ok"],
            pmids=cached["pmids"],
            term=cached["term"],
            error=cached.get("error"),
        )

    params: Dict[str, Any] = {
        "db": "pubmed",
        "term": term,
        "retstart": retstart,
        "retmax": retmax,
        "retmode": "xml",
        "datetype": "pdat",
        "sort": sort,
    }
    if maxdate:
        params["maxdate"] = maxdate
    if api_key:
        params["api_key"] = api_key

    try:
        resp = http_get_with_backoff(
            PUBMED_ESEARCH_URL,
            params=params,
            timeout=(8, 30),  # <- ここ重要
            max_retries=5,
            base_wait_sec=0.5,
        )
        root = ET.fromstring(resp.content)
        pmids = [elem.text for elem in root.findall("IdList/Id") if elem.text]

        out = ESearchResult(ok=True, pmids=pmids, term=term, error=None)
        cache[cache_key] = {
            "ok": out.ok,
            "pmids": out.pmids,
            "term": out.term,
            "error": out.error,
        }
        return out

    except Exception as e:
        # 失敗はキャッシュしない（Good）
        return ESearchResult(ok=False, pmids=[], term=term, error=repr(e))

    finally:
        if polite_delay_sec > 0:
            _sleep_jitter(polite_delay_sec)


def get_pmids_for_drug_gene(
    drug: str,
    gene: str,
    retmax: int = 50,
    retstart: int = 0,
    maxdate: Optional[str] = None,
    api_key: Optional[str] = None,
    sort: str = "pub date",
    polite_delay_sec: float = 0.0,
) -> ESearchResult:
    return pubmed_esearch(
        term=f'({drug}) AND ({_expand_gene_query(gene)}) AND "pubmed pmc"[sb]',
        retstart=retstart,
        retmax=retmax,
        maxdate=maxdate,
        api_key=api_key,
        sort=sort,
        polite_delay_sec=polite_delay_sec,
    )


def get_pmids_for_term(
    term: str,
    retmax: int = 50,
    retstart: int = 0,
    maxdate: Optional[str] = None,
    api_key: Optional[str] = None,
    sort: str = "pub date",
    polite_delay_sec: float = 0.0,
) -> ESearchResult:
    return pubmed_esearch(
        term=term,
        retstart=retstart,
        retmax=retmax,
        maxdate=maxdate,
        api_key=api_key,
        sort=sort,
        polite_delay_sec=polite_delay_sec,
    )





def _expand_gene_query(gene: str) -> str:
    """Expand ambiguous gene symbols for PubMed search.

    Args:
        gene: Gene/protein symbol.

    Returns:
        Expanded query string for the gene.
    """
    norm = (gene or "").strip().lower()
    if norm == "ar":
        return '("androgen receptor" OR NR3C4)'
    return gene


# -------------------------
# PMC OA download from S3
# -------------------------


def try_download_with_retries(
    s3_client,
    bucket_name: str,
    key: str,
    cache_path: str,
    max_retries: int = 3,
    wait_sec: int = 2,
) -> bool:
    """
    Try downloading from S3 with basic retries.
    """
    for attempt in range(max_retries):
        try:
            s3_client.download_file(bucket_name, key, cache_path)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "Unknown")
            if code == "404":
                # 404 is common; keep it quiet unless you want noisy logs
                logger.debug(f"S3 not found (404): {key}")
                return False
            logger.warning(
                f"S3 attempt {attempt+1}/{max_retries} ClientError {code} for {key}: {e}"
            )
        except Exception as e:
            logger.warning(f"S3 attempt {attempt+1}/{max_retries} error for {key}: {e}")
        time.sleep(wait_sec)
    return False


# 改良版 download_pmc_s3（貼り替え）
def download_pmc_s3(
    pmc_id: str,
    file_type: str = "xml",
    output_dir: str = "output/pmc",
    cache_dir: str = "cache/pmc",
    bucket_name: str = "pmc-oa-opendata",
    s3_client=_S3,
) -> bool:
    """
    Download PMC OA XML from S3, using local cache.
    Tries S3 keys first then NCBI fallback. Returns True if output file exists at end.
    """
    if not pmc_id:
        logger.warning("pmc_id is empty or None; skipping download.")
        return False

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{pmc_id}.{file_type}")
    cache_path = os.path.join(cache_dir, f"{pmc_id}.{file_type}")

    # Output exists => success
    if os.path.exists(output_path):
        return True

    # Known-fail cache
    if cache.get(f"pmc_fail_{pmc_id}"):
        logger.debug(f"Skipping known-fail PMCID {pmc_id}")
        return False

    # Local cache hit -> copy
    if os.path.exists(cache_path):
        try:
            shutil.copy(cache_path, output_path)
            return os.path.exists(output_path)
        except Exception as e:
            logger.warning(f"Failed to copy cached {cache_path} -> {output_path}: {e}")

    # Try S3 common keys
    s3_keys = [
        f"oa_comm/{file_type}/all/{pmc_id}.{file_type}",
        f"oa_noncomm/{file_type}/all/{pmc_id}.{file_type}",
        f"author_manuscript/{file_type}/all/{pmc_id}.{file_type}",
    ]

    for key in s3_keys:
        if try_download_with_retries(s3_client, bucket_name, key, cache_path):
            try:
                shutil.copy(cache_path, output_path)
                logger.info(f"Downloaded from S3 {key} -> {output_path}")
                return os.path.exists(output_path)
            except Exception as e:
                logger.warning(
                    f"Failed to copy S3 cache {cache_path} -> {output_path}: {e}"
                )

    # All methods failed -> mark known-fail
    logger.error(f"Failed to download {pmc_id}.{file_type} from all known sources.")
    cache[f"pmc_fail_{pmc_id}"] = True
    return False


# -------------------------
# Collection helpers
# -------------------------


def download_pmcs_for_pmids(
    pmids: List[str],
    output_dir: str = "output/pmc",
    cache_dir: str = "cache/pmc",
) -> Dict[str, Any]:
    """
    Given PMIDs, resolve PMCIDs and ensure OA XMLs exist in output_dir.

    IMPORTANT:
      - 'downloaded' counts how many PMCIDs are AVAILABLE in output_dir after this call.
        (includes newly downloaded + already-existing files)
      - 'skipped_exists' is still tracked separately for audit.
    """
    stats: Dict[str, Any] = {
        "pmid_count": len(pmids),
        "pmcid_count": 0,
        "downloaded": 0,  # available count
        "skipped_no_pmcid": 0,
        "skipped_exists": 0,
        "download_failed": 0,
        "available_pmcids": set(),  # ✅ ここ（ループ外）
    }

    seen_pmcids = set()  # avoid double-counting if multiple PMIDs map to same PMCID

    for pmid in pmids:
        try:
            pmcid = check_pmcid(pmid)
            if not pmcid:
                stats["skipped_no_pmcid"] += 1
                continue

            # dedupe by PMCID
            if pmcid in seen_pmcids:
                continue
            seen_pmcids.add(pmcid)

            stats["pmcid_count"] += 1

            out_path = os.path.join(output_dir, f"{pmcid}.xml")
            if os.path.exists(out_path):
                stats["skipped_exists"] += 1
                stats["downloaded"] += 1
                stats["available_pmcids"].add(pmcid)  # ✅
                continue

            ok = download_pmc_s3(pmcid, output_dir=output_dir, cache_dir=cache_dir)
            if ok:
                stats["downloaded"] += 1
                stats["available_pmcids"].add(pmcid)  # ✅（これが無いとダメ）
            else:
                stats["download_failed"] += 1

        except Exception as e:
            stats["download_failed"] += 1
            logger.error(f"Error processing PMID {pmid}: {e}")

    return stats


def download_k_range_xml_for_term(
    term: str,
    k_min: int = 2,
    k_max: int = 10,
    page_size: int = 20,
    max_pages: int = 10,
    sort: str = "pub date",
    maxdate: Optional[str] = None,
    output_dir: str = "output/pmc",
    cache_dir: str = "cache/pmc",
    polite_delay_sec: float = 0.3,
) -> Dict[str, Any]:
    downloaded = 0
    visited_pmids = set()

    seen_pmcids_total = set()
    fail_streak = 0
    dup_streak = 0
    zero_progress_streak = 0  # ✅ ここで初期化（for page の外）

    reason = None
    pages_used = 0

    for page in range(max_pages):
        pages_used += 1
        retstart = page * page_size

        res = get_pmids_for_term(
            term=term,
            retmax=page_size,
            retstart=retstart,
            maxdate=maxdate,
            api_key=NCBI_API_KEY,
            sort=sort,
            polite_delay_sec=polite_delay_sec,
        )

        if not res.ok:
            fail_streak += 1
            if fail_streak >= 3:
                reason = "esearch_failed"
                break
            continue
        fail_streak = 0

        if not res.pmids:
            reason = "no_more_pmids"
            break

        new_pmids = [p for p in res.pmids if p not in visited_pmids]
        if not new_pmids:
            dup_streak += 1
            if dup_streak >= 3:
                reason = "no_more_pmids"
                break
            continue
        dup_streak = 0
        visited_pmids.update(new_pmids)

        remaining = k_max - downloaded
        if remaining <= 0:
            reason = "reached_k_max"
            break

        new_pmids = new_pmids[:remaining]

        dl = download_pmcs_for_pmids(
            new_pmids, output_dir=output_dir, cache_dir=cache_dir
        )

        # ✅ ここ：new_avail を計算した直後に “進捗ゼロ” を判定
        new_avail = dl["available_pmcids"] - seen_pmcids_total
        seen_pmcids_total |= dl["available_pmcids"]

        if len(new_avail) == 0:
            zero_progress_streak += 1
            if zero_progress_streak >= 5:
                reason = "no_progress"
                break
        else:
            zero_progress_streak = 0

        downloaded += len(new_avail)

        if downloaded >= k_max:
            reason = "reached_k_max"
            break

    if reason is None:
        reason = "page_limit"

    return {
        "term": term,
        "downloaded": downloaded,
        "k_min": k_min,
        "k_max": k_max,
        "met_min": downloaded >= k_min,
        "reason": reason,
        "pages_used": pages_used,
        "page_size": page_size,
        "max_pages": max_pages,
    }


# -------------------------
# Main collectors
# -------------------------


def collect_papers(
    drug: str,
    gene: str,
    max_search_results: int = 10,
    maxdate: Optional[str] = None,
    output_dir: str = "output/pmc",
    cache_dir: str = "cache/pmc",
    polite_delay_sec: float = 0.3,
    sort: str = "pub date",
) -> Dict[str, Any]:
    """
    Pair-only: drug AND gene, then download PMC OA XMLs (if any).
    Returns a row-dict suitable for pandas.
    """
    row: Dict[str, Any] = {
        "drug": drug,
        "gene": gene,
        "maxdate": maxdate,
        "pair_ok": None,
        "pair_term": None,
        "pair_pmid_count": None,
        "pair_error": None,
        "pair_pmcid_count": 0,
        "pair_downloaded": 0,
        "pair_skipped_no_pmcid": 0,
        "pair_skipped_exists": 0,
        "pair_download_failed": 0,
    }

    res = get_pmids_for_drug_gene(
        drug=drug,
        gene=gene,
        retmax=max_search_results,
        retstart=0,
        maxdate=maxdate,
        api_key=NCBI_API_KEY,
        sort=sort,
        polite_delay_sec=polite_delay_sec,
    )

    row["pair_ok"] = res.ok
    row["pair_term"] = res.term
    row["pair_pmid_count"] = len(res.pmids)
    row["pair_error"] = res.error

    if res.ok:
        logger.info(f"Found {len(res.pmids)} PMID(s) for '{drug} × {gene}'")
        pair_out = os.path.join(output_dir, "pair")
        dl = download_pmcs_for_pmids(
            res.pmids, output_dir=pair_out, cache_dir=cache_dir
        )
        row["pair_pmcid_count"] = dl["pmcid_count"]
        row["pair_downloaded"] = dl["downloaded"]
        row["pair_skipped_no_pmcid"] = dl["skipped_no_pmcid"]
        row["pair_skipped_exists"] = dl["skipped_exists"]
        row["pair_download_failed"] = dl["download_failed"]
    else:
        logger.error(f"PubMed search failed for '{drug} × {gene}': {res.error}")

    return row


def collect_papers_with_marginals(
    drug: str,
    gene: str,
    max_search_results_pair: int = 10,
    max_search_results_single: int = 10,
    maxdate: Optional[str] = None,
    output_dir: str = "output/pmc",
    cache_dir: str = "cache/pmc",
    download_pair_pmcs: bool = False,
    download_drug_pmcs: bool = False,
    download_gene_pmcs: bool = False,
    polite_delay_sec: float = 0.3,
    sort: str = "pub date",
    # NEW adaptive mode for single-term:
    k_single_min: int = 2,
    k_single_max: int = 10,
    single_page_size: int = 20,
    single_max_pages: int = 5,
) -> Dict[str, Any]:
    """
    Collect pair + marginals:
      - pair: drug AND gene (Top N)
      - drug-only: drug (Top N) OR adaptive "download >=K"
      - gene-only: gene (Top N) OR adaptive "download >=K"

    Returns one row-dict suitable for pandas.
    """
    row: Dict[str, Any] = {
        "drug": drug,
        "gene": gene,
        "maxdate": maxdate,
        "pair_ok": None,
        "pair_term": None,
        "pair_pmid_count": None,
        "pair_error": None,
        "drug_ok": None,
        "drug_term": None,
        "drug_pmid_count": None,
        "drug_error": None,
        "gene_ok": None,
        "gene_term": None,
        "gene_pmid_count": None,
        "gene_error": None,
        # DL stats (3 buckets)
        "pair_downloaded": 0,
        "drug_downloaded": 0,
        "gene_downloaded": 0,
        "pair_download_failed": 0,
        "drug_download_failed": 0,
        "gene_download_failed": 0,
        "pair_skipped_no_pmcid": 0,
        "drug_skipped_no_pmcid": 0,
        "gene_skipped_no_pmcid": 0,
        "pair_skipped_exists": 0,
        "drug_skipped_exists": 0,
        "gene_skipped_exists": 0,
    }

    pair_out = os.path.join(output_dir, "pair")
    drug_out = os.path.join(output_dir, "drug")
    gene_out = os.path.join(output_dir, "gene")

    # --- Search: pair (top-N) ---
    pair = get_pmids_for_drug_gene(
        drug=drug,
        gene=gene,
        retmax=max_search_results_pair,
        retstart=0,
        maxdate=maxdate,
        api_key=NCBI_API_KEY,
        sort=sort,
        polite_delay_sec=polite_delay_sec,
    )
    row["pair_ok"] = pair.ok
    row["pair_term"] = pair.term
    row["pair_pmid_count"] = len(pair.pmids)
    row["pair_error"] = pair.error

    # --- Search: drug-only (top-N for counting, always recorded) ---
    query = f'("{drug}") AND "pubmed pmc"[sb]'
    d = get_pmids_for_term(
        term=query,
        retmax=max_search_results_single,
        retstart=0,
        maxdate=maxdate,
        api_key=NCBI_API_KEY,
        sort=sort,
        polite_delay_sec=polite_delay_sec,
    )
    row["drug_ok"] = d.ok
    row["drug_term"] = d.term
    row["drug_pmid_count"] = len(d.pmids)
    row["drug_error"] = d.error

    # --- Search: gene-only (top-N for counting, always recorded) ---
    query = f'({_expand_gene_query(gene)}) AND "pubmed pmc"[sb]'
    g = get_pmids_for_term(
        term=query,
        retmax=max_search_results_single,
        retstart=0,
        maxdate=maxdate,
        api_key=NCBI_API_KEY,
        sort=sort,
        polite_delay_sec=polite_delay_sec,
    )
    row["gene_ok"] = g.ok
    row["gene_term"] = g.term
    row["gene_pmid_count"] = len(g.pmids)
    row["gene_error"] = g.error

    def _apply_dl_stats(prefix: str, stats: Dict[str, int]) -> None:
        row[f"{prefix}_downloaded"] = stats["downloaded"]
        row[f"{prefix}_download_failed"] = stats["download_failed"]
        row[f"{prefix}_skipped_no_pmcid"] = stats["skipped_no_pmcid"]
        row[f"{prefix}_skipped_exists"] = stats["skipped_exists"]

    # --- Optional downloads: pair bucket (fixed top-N only) ---
    if download_pair_pmcs and pair.ok and pair.pmids:
        _apply_dl_stats(
            "pair",
            download_pmcs_for_pmids(
                pair.pmids, output_dir=pair_out, cache_dir=cache_dir
            ),
        )

    # drug
    if download_drug_pmcs and d.ok:
        dlr = download_k_range_xml_for_term(
            term=d.term,
            k_min=k_single_min,
            k_max=k_single_max,
            page_size=single_page_size,
            max_pages=single_max_pages,
            sort=sort,
            maxdate=maxdate,
            output_dir=drug_out,  # ★ここ
            cache_dir=cache_dir,
            polite_delay_sec=polite_delay_sec,
        )
        row["drug_downloaded"] = dlr["downloaded"]
        row["drug_met_min"] = dlr["met_min"]
        row["drug_reason"] = dlr["reason"]
        row["drug_pages_used"] = dlr["pages_used"]

    # gene
    if download_gene_pmcs and g.ok:
        dlr = download_k_range_xml_for_term(
            term=g.term,
            k_min=k_single_min,
            k_max=k_single_max,
            page_size=single_page_size,
            max_pages=single_max_pages,
            sort=sort,
            maxdate=maxdate,
            output_dir=gene_out,  # ★ここ
            cache_dir=cache_dir,
            polite_delay_sec=polite_delay_sec,
        )
        row["gene_downloaded"] = dlr["downloaded"]
        row["gene_met_min"] = dlr["met_min"]
        row["gene_reason"] = dlr["reason"]
        row["gene_pages_used"] = dlr["pages_used"]

    return row


# -------------------------
# CLI demo
# -------------------------

if __name__ == "__main__":
    DRUG_NAME = "Gefitinib"
    GENE_NAME = "EGFR"

    # Pair-only (with download)
    out = collect_papers(
        DRUG_NAME, GENE_NAME, max_search_results=20, maxdate="2018/12/31"
    )
    logger.info(f"Done: {out}")

    # Pair + marginals (adaptive: try >=2 XML for drug & gene)
    out2 = collect_papers_with_marginals(
        DRUG_NAME,
        GENE_NAME,
        max_search_results_pair=20,
        max_search_results_single=10,
        maxdate="2018/12/31",
        download_pair_pmcs=True,
        download_drug_pmcs=True,
        download_gene_pmcs=True,
        k_single_min=2,
        k_single_max=10,
        single_page_size=20,
        single_max_pages=5,
        sort="pub date",
    )
    logger.info(f"Done (marginals adaptive): {out2}")
