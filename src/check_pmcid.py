import json
import logging
import os
import time
import xml.etree.ElementTree as ET

import requests
from diskcache import Cache
from dotenv import load_dotenv

# Ensure cache directory exists
os.makedirs("cache", exist_ok=True)
cache = Cache("cache")


def check_pmcid(pmid, email="inouey2@nih.gov"):
    key = f"pmcid_{pmid}"
    if key in cache:
        return cache[key]

    base_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    params = {"ids": pmid, "format": "json", "tool": "return_pmcid", "email": email}

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if "records" in data and data["records"]:
            record = data["records"][0]
            if "pmcid" in record:
                cache[key] = record["pmcid"]
                return record["pmcid"]
            else:
                cache[key] = None
                return None
        else:
            cache[key] = None
            return None

    except requests.exceptions.RequestException:
        # Log the error if needed, but don't expose the exception
        print(f"Error looking up PMCID for {pmid}")
        cache[key] = None
        return None


# --- NCBI E-utilities のエンドポイント ---
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PMC_ID_CONVERT_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"


def get_pmcids_for_drug_gene(
    drug: str, gene: str, max_search_results: int = 50, maxdate: str = None
) -> list[str]:
    query = f"{drug} AND {gene}"
    pmids = []
    try:
        esearch_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_search_results,
            "retmode": "xml",
            "maxdate": maxdate,
            "datetype": "pdat",  # 発行日を基準
        }
        response = requests.get(PUBMED_ESEARCH_URL, params=esearch_params)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        pmids = [id_elem.text for id_elem in root.findall("IdList/Id")]

        if not pmids:
            return []

    except requests.exceptions.RequestException as e:
        return []
    except ET.ParseError as e:
        return []

    return pmids
