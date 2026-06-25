import pandas as pd
import requests
from tqdm import tqdm


def fetch_pubchem_name(cid):
    """
    Fetch compound name from PubChem CID.
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{int(cid)}/property/Title/JSON"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return data["PropertyTable"]["Properties"][0]["Title"]
    except:
        pass
    return None


def build_drug_name_dict(cid_series):
    """
    Build a {cid: drug_name} mapping from a CID list.
    """
    cid_unique = cid_series.dropna().unique()
    mapping = {}
    for cid in tqdm(cid_unique):
        mapping[cid] = fetch_pubchem_name(cid)
    return mapping


def fetch_uniprot_gene(uniprot_id):
    """
    Fetch Gene Symbol from UniProt ID.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            genes = data.get("genes", [])
            if genes:
                return genes[0]["geneName"]["value"]
    except:
        pass
    return None


def build_gene_name_dict(uniprot_series):
    """
    Build a {uniprot_id: gene_symbol} mapping from a UniProt ID list.
    """
    ids_unique = uniprot_series.dropna().unique()
    mapping = {}
    for uid in tqdm(ids_unique):
        mapping[uid] = fetch_uniprot_gene(uid)
    return mapping


def merge_drug_gene_names(df, drug_id_col="Drug_ID", target_id_col="Target_ID"):
    """
    Add Drug_Name and Gene_Name from Drug_ID (PubChem CID) and Target_ID (UniProt ID).
    """

    drug_dict = build_drug_name_dict(df[drug_id_col])
    gene_dict = build_gene_name_dict(df[target_id_col])

    df = df.copy()
    df["Drug_Name"] = df[drug_id_col].map(drug_dict)
    df["Gene_Name"] = df[target_id_col].map(gene_dict)

    return df
