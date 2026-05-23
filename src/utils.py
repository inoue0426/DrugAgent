import ast
import datetime
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests  # type: ignore
from autogen_agentchat.messages import TextMessage, ToolCallSummaryMessage
from DeepPurpose.dataset import load_broad_repurposing_hub
from dotenv import load_dotenv
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT_DIR / "data"
NSC_SMILES_PATH = DATA_DIR / "nsc_cid_smiles_class_name.csv"
DRUG2SMILES_PATH = DATA_DIR / "drug2smiles.csv.gz"
DRUGBANK_SMILES_PATH = DATA_DIR / "drugbank_2021_smiles.csv"
DRUGBANK_STR_PATH = DATA_DIR / "drugbank_str.csv"
PROCESSED_KG_PATH = DATA_DIR / "processed_kg.csv"
GPCR2SMILES_PATH = DATA_DIR / "gpcr2smiles.csv"


# gpcr2smiles 用のインメモリ辞書（初回のみ構築）
_GPCR2SMILES_MAP = None

SALT_BASE = r"(hydrochloride|hcl|sulfate|sulphate|mesylate|maleate|fumarate|tartrate|citrate|oxalate|nitrate|benzoate|bitartrate|tosylate|succinate|phosphate|potassium|sodium|monohydrate|dihydrate|trihydrate|bromide|chloride|iodide)"
SALT_WITH_PREFIX = rf"(?:d|l|dl|d,l)\s*-\s*{SALT_BASE}"
SALT_SUFFIX_RE = rf"(?:{SALT_WITH_PREFIX}|{SALT_BASE})"

def _normalize_name(x: str) -> str:
    if x is None:
        return ""
    s = str(x).lower().strip()
    s = re.sub(r"\s+", " ", s)

    # "(+)-", "(-)-", "(±)-" みたいな先頭の立体表記を除去
    s = re.sub(r"^\s*\((?:\+|\-|\±)\)\s*-\s*", "", s)

    # 末尾の塩/水和物を繰り返し除去（"tartrate dihydrate" みたいなのに対応）
    while True:
        s2 = re.sub(rf"\s+{SALT_SUFFIX_RE}\s*$", "", s).strip()
        if s2 == s:
            break
        s = s2

    # "(R)", "(S)" の除去
    s = re.sub(r"\(\s*(?:r|s)\s*\)", "", s).strip()

    # 先頭のゴミ
    s = re.sub(r"^[\-\s]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def to_camel_case(s):
    s = s.lower().strip()  # 小文字＋前後空白除去
    parts = s.split()  # スペースで分割（単語の区切り想定）
    return "".join(p.capitalize() for p in parts)


def validate_dti_output(output: dict) -> bool:
    required_keys = {"drug", "target", "score", "reason"}
    if not required_keys.issubset(output.keys()):
        return False
    if not isinstance(output["drug"], str) or not isinstance(output["target"], str):
        return False
    if not (
        isinstance(output["score"], (float, int))
        and 0.0 <= float(output["score"]) <= 1.0
    ):
        return False
    if not isinstance(output["reason"], str):
        return False
    return True


def save_dti_results(drugs, targets, result, path):
    """Save DTI results to CSV file

    Args:
        drugs (List[str]): List of drug names
        targets (List[str]): List of target names
        result (DTIScore): DTI score results
        path (str): Path to save CSV file
    """
    # Convert DTIScore object to DataFrame
    df = pd.DataFrame(
        {
            "drug": drugs,
            "gene": targets,
            "ml_score": result.ml_dti_scores,
            "kg_score": result.kg_dti_scores,
            "search_score": result.search_dti_scores,
            "final_score": result.final_dti_scores,
            "reasoning": result.reasoning,
        }
    )

    try:
        existing_result = pd.read_csv(path, encoding="utf-8", encoding_errors="ignore")
        merged_result = pd.concat([existing_result, df]).drop_duplicates(
            subset=["drug", "gene"], keep="last"
        )
        merged_result.to_csv(path, index=False)
    except FileNotFoundError:
        df.to_csv(path, index=False)


def calculate_binary_metrics(y_true, y_pred, y_pred_proba=None, decimals=4):
    """
    Calculate binary classification metrics and return as a simple pandas DataFrame.

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
    y_pred_proba : array-like, optional
        Predicted probabilities for the positive class
    decimals : int, default=4
        Number of decimal places to round to

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing metrics and their values
    """
    metrics = {}

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Core metrics
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Balanced Accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred)
    metrics["Recall"] = recall_score(y_true, y_pred)
    metrics["Specificity"] = tn / (tn + fp)
    metrics["F1"] = f1_score(y_true, y_pred)
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred)

    # Error rates
    metrics["FPR"] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics["FNR"] = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Add probability metrics if proba is provided
    if y_pred_proba is not None:
        metrics["AUC-ROC"] = roc_auc_score(y_true, y_pred_proba)
        metrics["AUPRC"] = average_precision_score(y_true, y_pred_proba)
        metrics["Log Loss"] = log_loss(y_true, y_pred_proba)
        metrics["Brier Score"] = np.mean((y_pred_proba - y_true) ** 2)

    # Create DataFrame and round values
    metrics_df = pd.DataFrame(
        {"metric_name": list(metrics.keys()), "value": list(metrics.values())}
    )

    metrics_df["value"] = metrics_df["value"].round(decimals)

    return metrics_df.sort_values("metric_name").reset_index(drop=True)


import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)


def specificity_score(y_true, y_pred):
    """特異度（Specificity）"""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return np.nan


def safe_log_loss(y_true, prob):
    try:
        return log_loss(y_true, prob)
    except Exception:
        return np.nan


def safe_roc_auc(y_true, prob):
    try:
        return roc_auc_score(y_true, prob)
    except Exception:
        return np.nan


def safe_avg_precision(y_true, prob):
    try:
        return average_precision_score(y_true, prob)
    except Exception:
        return np.nan


def evaluate_metrics(
    y_df, class_col="class", pred_col="pred", prob_col="prob", group_col="flag"
):
    """
    Calculate multiple evaluation metrics for a DataFrame, aggregated by group_col.

    Parameters:
        y_df (pd.DataFrame): Input data. Must contain columns: class, pred, flag, prob.
        class_col (str): Column name for true labels (default: "class")
        pred_col (str): Column name for predicted labels (default: "pred")
        prob_col (str): Column name for predicted probability of the positive class (default: "prob")
        group_col (str): Column name for grouping/aggregation (default: "flag")

    Returns:
        pd.DataFrame: Aggregated evaluation metrics.
    """

    def compute_metrics(group):
        y_true = group[class_col]
        y_pred = group[pred_col]
        prob = group[prob_col] if prob_col in group else None

        return pd.Series(
            {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "specificity": specificity_score(y_true, y_pred),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "MCC": matthews_corrcoef(y_true, y_pred),
                "kappa": cohen_kappa_score(y_true, y_pred),
                "log_loss": safe_log_loss(y_true, prob) if prob is not None else np.nan,
                "roc_auc": safe_roc_auc(y_true, prob) if prob is not None else np.nan,
                "aupr": (
                    safe_avg_precision(y_true, prob) if prob is not None else np.nan
                ),
            }
        )

    res = y_df.groupby(group_col).apply(compute_metrics).reset_index()
    mean = res.drop(columns=["flag"]).mean(numeric_only=True)
    std = res.drop(columns=["flag"]).std(numeric_only=True)
    summary_mean_std = mean.map("{:.3f}".format) + " (± " + std.map("{:.3f})".format)

    return res, pd.DataFrame(summary_mean_std).T


def format_dti_report_from_messages(messages: List[Any]) -> str:
    report_lines = []

    # ユーザーの質問を抽出
    user_query = None
    for msg in messages:
        if isinstance(msg, TextMessage) and msg.source == "user":
            user_query = msg.content.strip()
            break
    if user_query:
        report_lines.append(f"## Drug-Target Interaction Analysis Report\n")
        report_lines.append(f"**User Query:** {user_query}\n")
    else:
        report_lines.append(f"## Drug-Target Interaction Analysis Report\n")
        report_lines.append(
            f"**User Query:** Not available (could not extract from messages)\n"
        )

    report_lines.append("### Individual Agent Findings:\n")

    agent_findings: Dict[str, List[Dict[str, Any]]] = {}
    processed_tool_summaries = set()

    for msg in reversed(messages):
        if (
            isinstance(msg, ToolCallSummaryMessage)
            and msg.source not in processed_tool_summaries
        ):
            source_agent = msg.source
            try:
                parsed_content = ast.literal_eval(msg.content)
                if (
                    isinstance(parsed_content, list)
                    and len(parsed_content) > 0
                    and isinstance(parsed_content[0], list)
                ):
                    for item in parsed_content:
                        if len(item) >= 4:
                            drug, target, score, reasoning = (
                                item[0],
                                item[1],
                                item[2],
                                item[3],
                            )
                            if source_agent not in agent_findings:
                                agent_findings[source_agent] = []
                            agent_findings[source_agent].append(
                                {
                                    "drug": drug,
                                    "target": target,
                                    "score": score,
                                    "reasoning": reasoning,
                                }
                            )
                    processed_tool_summaries.add(source_agent)
            except (ValueError, SyntaxError) as e:
                if source_agent not in agent_findings:
                    agent_findings[source_agent] = []
                agent_findings[source_agent].append(
                    {
                        "raw_content": msg.content,
                        "error": f"Failed to parse tool output: {e}",
                    }
                )
                processed_tool_summaries.add(source_agent)

    for agent_name in ["WebSearchAgent", "MLAgent", "KGAgent"]:
        if agent_name in agent_findings:
            report_lines.append(f"**{agent_name}:**\n")
            for finding in agent_findings[agent_name]:
                if "score" in finding:
                    report_lines.append(
                        f"- Drug: {finding['drug']}, Target: {finding['target']}, Score: {finding['score']}\n"
                    )
                    reasoning_full_text = (
                        finding["reasoning"].replace("\\n", " ").strip()
                    )
                    report_lines.append(f"  Reasoning: {reasoning_full_text}\n")
                else:
                    report_lines.append(
                        f"- Raw Output: {finding.get('raw_content', 'N/A')}\n"
                    )
                    report_lines.append(f"  Error: {finding.get('error', 'N/A')}\n")
            report_lines.append("\n")

    # SummaryAgent の最終メッセージから最も「Coincise」な結論を抽出
    final_conclusion_text = None
    for msg in reversed(messages):
        if isinstance(msg, TextMessage) and msg.source == "SummaryAgent":
            content = msg.content.strip()

            # 1. "Final Output:" で始まる行を優先して探す (SummaryAgentが生成する場合)
            match_final_output = re.search(
                r"Final Output:\s*(.+?)(?:\s*TERMINATE)?$", content, re.DOTALL
            )
            if match_final_output:
                extracted_string = match_final_output.group(1).strip()
                try:
                    # リスト形式の文字列であればパースし、最後の要素 (理由付け) を取得
                    parsed_list = ast.literal_eval(extracted_string)
                    if isinstance(parsed_list, list) and len(parsed_list) > 6:
                        final_conclusion_text = str(parsed_list[6])
                    else:
                        # リスト形式だが不完全、または想定外のリスト形式の場合
                        final_conclusion_text = extracted_string
                except (ValueError, SyntaxError):
                    # リスト形式ではない場合 (直接テキストの場合)
                    final_conclusion_text = extracted_string

                # 不要なクォートと改行コードを処理
                if final_conclusion_text.startswith(
                    '"'
                ) and final_conclusion_text.endswith('"'):
                    final_conclusion_text = final_conclusion_text[1:-1]
                if final_conclusion_text.startswith(
                    "'"
                ) and final_conclusion_text.endswith("'"):
                    final_conclusion_text = final_conclusion_text[1:-1]
                final_conclusion_text = final_conclusion_text.replace(
                    "\\n", "\n"
                ).strip()
                break  # 適切な結論が見つかったのでループを終了

            # 2. `Final Output:` がないが、コンテンツ全体がリスト形式の文字列の場合 (提供されたログの最後の形式)
            #    ただし、これはログの最後のメッセージにのみ適用されるべき
            if content.startswith("[") and content.endswith("]") and "," in content:
                try:
                    parsed_list = ast.literal_eval(content)
                    if isinstance(parsed_list, list) and len(parsed_list) > 6:
                        final_conclusion_text = str(parsed_list[6])
                        if final_conclusion_text.startswith(
                            '"'
                        ) and final_conclusion_text.endswith('"'):
                            final_conclusion_text = final_conclusion_text[1:-1]
                        if final_conclusion_text.startswith(
                            "'"
                        ) and final_conclusion_text.endswith("'"):
                            final_conclusion_text = final_conclusion_text[1:-1]
                        final_conclusion_text = final_conclusion_text.replace(
                            "\\n", "\n"
                        ).strip()
                        break  # 適切な結論が見つかったのでループを終了
                except (ValueError, SyntaxError):
                    pass  # パースできない場合は次の条件を試す

            # 3. 上記のいずれにも当てはまらないが、"TERMINATE" を含むメッセージの場合 (思考プロセスを含む可能性あり)
            #    この場合は、"TERMINATE" 以前の全てを結論とするが、最も簡潔なものが優先されるため、
            #    このフォールバックは最後に評価されるべき。
            #    しかし、今回のログでは、最後のメッセージがリスト形式なので、ここは通常ヒットしない。
            #    もし LLM が `Final Output:` やリスト形式でなく、そのままのテキストを返した時に対応する。
            if "TERMINATE" in content:
                temp_conclusion = content[: content.rfind("TERMINATE")].strip()
                if not final_conclusion_text:  # まだ結論が見つかっていない場合のみ更新
                    final_conclusion_text = temp_conclusion.replace("\\n", "\n").strip()
                    # SummaryAgent の Thought/Action/Observation のブロックを削除
                    # より一般的なパターンで Thought: から次の Thought: または Action: までを削除
                    final_conclusion_text = re.sub(
                        r"(Thought:|Action:|Observation:).*?(?=(Thought:|Action:|Observation:|$))",
                        "",
                        final_conclusion_text,
                        flags=re.DOTALL,
                    )
                    final_conclusion_text = final_conclusion_text.strip()
                    break  # これを最終結論として採用し、ループを終了

    if final_conclusion_text:
        report_lines.append("### Final Conclusion from SummaryAgent:\n")
        report_lines.append(final_conclusion_text + "\n")
    else:
        report_lines.append("### Final Conclusion from SummaryAgent:\n")
        report_lines.append(
            "No comprehensive conclusion was provided by the SummaryAgent or could not be extracted.\n"
        )

    return "\n".join(report_lines)


def save_dti_results(drugs, targets, result, path):
    """Save DTI results to CSV file

    Args:
        drugs (List[str]): List of drug names
        targets (List[str]): List of target names
        result (DTIScore): DTI score results
        path (str): Path to save CSV file
    """
    # Convert DTIScore object to DataFrame
    df = pd.DataFrame(
        {
            "drug": drugs,
            "gene": targets,
            "ml_score": result.ml_dti_scores,
            "kg_score": result.kg_dti_scores,
            "search_score": result.search_dti_scores,
            "final_score": result.final_dti_scores,
            "reasoning": result.reasoning,
        }
    )

    try:
        existing_result = pd.read_csv(path, encoding="utf-8", encoding_errors="ignore")
        merged_result = pd.concat([existing_result, df]).drop_duplicates(
            subset=["drug", "gene"], keep="last"
        )
        merged_result.to_csv(path, index=False)
    except FileNotFoundError:
        df.to_csv(path, index=False)


def get_target_name_from_uniprot(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("genes", [{}])[0].get("geneName", {}).get("value")
    except requests.RequestException as e:
        print(f"Error retrieving gene name for UniProt ID {uniprot_id}: {e}")
        return None


def _lookup_smiles_from_csv(path: Path, name_column: str, smiles_column: str, compound_name: str) -> Optional[str]:
    try:
        df = pd.read_csv(path, usecols=[name_column, smiles_column])
        key = _normalize_name(compound_name)
        mask = df[name_column].astype(str).map(_normalize_name) == key
        if mask.any():
            value = df.loc[mask, smiles_column].iloc[0]
            return str(value)
    except Exception as e:
        print(f"Error reading {path.name}: {e}")
    return None

def _get_gpcr2smiles_map() -> Dict[str, str]:
    global _GPCR2SMILES_MAP
    if _GPCR2SMILES_MAP is None:
        if not GPCR2SMILES_PATH.exists():
            _GPCR2SMILES_MAP = {}
        else:
            df = pd.read_csv(GPCR2SMILES_PATH, usecols=["Compound name", "SMILES"])
            df = df.dropna(subset=["Compound name", "SMILES"]).drop_duplicates(subset=["Compound name"])
            _GPCR2SMILES_MAP = {
                _normalize_name(n): str(s)
                for n, s in zip(df["Compound name"].astype(str), df["SMILES"].astype(str))
            }
    return _GPCR2SMILES_MAP

def _lookup_smiles_from_gpcr2smiles(compound_name: str) -> Optional[str]:
    m = _get_gpcr2smiles_map()
    return m.get(_normalize_name(compound_name))

def get_smiles_from_compound_name(compound_name):
    """Resolve SMILES by compound name using local files and PubChem fallback."""
    # 0) GPCR curated mapping (highest priority)
    smiles = _lookup_smiles_from_gpcr2smiles(compound_name)
    if smiles:
        return smiles

    # 1) existing local files...
    if DRUG2SMILES_PATH.exists():
        smiles = _lookup_smiles_from_csv(DRUG2SMILES_PATH, "name", "SMILES", compound_name)
        if smiles:
            return smiles

    try:
        df = pd.read_csv(
            NSC_SMILES_PATH, usecols=["NAME", "SMILES"]
        )
        # Look for matching compound name
        match = df[df["NAME"].astype(str).str.lower() == compound_name.lower()]
        if not match.empty:
            return match.iloc[0]["SMILES"]
    except Exception as e:
        print(f"Error reading local SMILES data: {e}")

    # Additional local fallbacks
    if DRUGBANK_SMILES_PATH.exists():
        smiles = _lookup_smiles_from_csv(
            DRUGBANK_SMILES_PATH, "Name", "SMILES", compound_name
        )
        if smiles:
            return smiles

    if DRUGBANK_STR_PATH.exists():
        smiles = _lookup_smiles_from_csv(
            DRUGBANK_STR_PATH, "drug_name", "SMILES", compound_name
        )
        if smiles:
            return smiles
        smiles = _lookup_smiles_from_csv(
            DRUGBANK_STR_PATH, "Name", "SMILES", compound_name
        )
        if smiles:
            return smiles

    if PROCESSED_KG_PATH.exists():
        smiles = _lookup_smiles_from_csv(
            PROCESSED_KG_PATH, "compound", "SMILES", compound_name
        )
        if smiles:
            return smiles

    # If not found locally, try PubChem API
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/property/CanonicalSMILES/JSON"
    try:
        response = _get_session().get(url, timeout=(5, 15))
        response.raise_for_status()
        data = response.json()
        return (
            data.get("PropertyTable", {})
            .get("Properties", [{}])[0]
            .get("CanonicalSMILES")
        )
    except requests.RequestException as e:
        print(f"Error retrieving SMILES for compound name {compound_name}: {e}")
        return None

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# グローバル session（再利用で高速＆安定）
_session = None


def _get_session():
    global _session
    if _session is None:
        session = requests.Session()

        retry = Retry(
            total=4,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )

        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        _session = session
    return _session


def get_sequence_from_target_name(target_name: str):
    """
    Robust UniProt sequence fetch.
    - timeout指定
    - retry対応
    - human gene限定
    - reviewed優先
    """

    # ヒト限定 & gene名検索
    query = f"gene_exact:{target_name} AND organism_id:9606"

    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "fields": "sequence",
        "format": "json",
        "size": 1,  # 最初の1件のみ
    }

    try:
        session = _get_session()

        response = session.get(
            url,
            params=params,
            timeout=(5, 15),  # (connect, read)
        )

        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            return None

        return results[0].get("sequence", {}).get("value")

    except requests.RequestException as e:
        print(f"[UniProt ERROR] {target_name}: {e}")
        return None


def get_compound_name(smiles):
    SAVE_PATH = "./saved_path"
    X_repurpose, drug_name, drug_cid = load_broad_repurposing_hub(SAVE_PATH)
    return drug_name[X_repurpose == smiles][0]


def calculate_binary_metrics(y_true, y_pred, y_pred_proba=None, decimals=4):
    """
    Calculate binary classification metrics and return as a simple pandas DataFrame.

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
    y_pred_proba : array-like, optional
        Predicted probabilities for the positive class
    decimals : int, default=4
        Number of decimal places to round to

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing metrics and their values
    """
    metrics = {}

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Core metrics
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Balanced Accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred)
    metrics["Recall"] = recall_score(y_true, y_pred)
    metrics["Specificity"] = tn / (tn + fp)
    metrics["F1"] = f1_score(y_true, y_pred)
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred)

    # Error rates
    metrics["FPR"] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics["FNR"] = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Add probability metrics if proba is provided
    if y_pred_proba is not None:
        metrics["AUC-ROC"] = roc_auc_score(y_true, y_pred_proba)
        metrics["AUPRC"] = average_precision_score(y_true, y_pred_proba)
        metrics["Log Loss"] = log_loss(y_true, y_pred_proba)
        metrics["Brier Score"] = np.mean((y_pred_proba - y_true) ** 2)

    # Create DataFrame and round values
    metrics_df = pd.DataFrame(
        {"metric_name": list(metrics.keys()), "value": list(metrics.values())}
    )

    metrics_df["value"] = metrics_df["value"].round(decimals)

    return metrics_df.sort_values("metric_name").reset_index(drop=True)


def get_smiles_from_kca_graph(drug):
    df = pd.read_csv(
        "data/kca_graph.csv.gz", index_col=0, usecols=["Drug", "SMILES"]
    ).drop_duplicates()
    return df[df["Drug"] == drug].iloc[0]["SMILES"]


def get_seq_from_kca_graph(target):
    df = pd.read_csv(
        "data/kca_graph.csv.gz", index_col=0, usecols=["Gene", "Seq"]
    ).drop_duplicates()
    return df[df["Gene"] == target].iloc[0]["Seq"]
