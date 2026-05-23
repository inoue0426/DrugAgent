import pandas as pd
import requests
from tqdm import tqdm


# ================================
# PubChem CID → Drug Name
# ================================
def fetch_pubchem_name(cid):
    """
    PubChem CID から化合物名を取得
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
    CIDリストから {cid: drug_name} 辞書を作成
    """
    cid_unique = cid_series.dropna().unique()
    mapping = {}
    for cid in tqdm(cid_unique):
        mapping[cid] = fetch_pubchem_name(cid)
    return mapping


# ================================
# UniProt ID → Gene Symbol
# ================================
def fetch_uniprot_gene(uniprot_id):
    """
    UniProt ID から Gene Symbol を取得
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
    UniProt IDリストから {uniprot_id: gene_symbol} 辞書を作成
    """
    ids_unique = uniprot_series.dropna().unique()
    mapping = {}
    for uid in tqdm(ids_unique):
        mapping[uid] = fetch_uniprot_gene(uid)
    return mapping


# ================================
# メインMerge関数
# ================================
def merge_drug_gene_names(df, drug_id_col="Drug_ID", target_id_col="Target_ID"):
    """
    Drug_ID (PubChem CID) と Target_ID (UniProt ID)
    から Drug_Name, Gene_Name を付与する
    """

    # 1. マッピング辞書作成
    drug_dict = build_drug_name_dict(df[drug_id_col])
    gene_dict = build_gene_name_dict(df[target_id_col])

    # 2. map
    df = df.copy()
    df["Drug_Name"] = df[drug_id_col].map(drug_dict)
    df["Gene_Name"] = df[target_id_col].map(gene_dict)

    return df
