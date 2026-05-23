"""Prompts and minimal flow notes for plausibility and faithfulness judging."""

from __future__ import annotations

from typing import Dict

# -----------------------------------------------------------------------------
# Plausibility prompt (from 10_DrugAgent-plausibility.ipynb)
# -----------------------------------------------------------------------------

PLAUSIBILITY_SYSTEM_PROMPT = """
You are a careful biomedical evaluator.

Your task is to evaluate whether a proposed drug-gene association is biologically plausible based on the provided input summary.

Definition:
A drug-gene association is plausible if:
1. it is not contradicted by the input, and
2. it would reasonably be considered a possible biological or pharmacological hypothesis based on the input.

Scoring:
1 = Strongly Disagree
    - The association is highly implausible, strongly contradicted, or likely to mislead downstream scientific or clinical reasoning.
2 = Disagree
    - The association is probably incorrect, unsupported, or biologically weak, but not maximally misleading.
3 = Neutral
    - The association is uncertain, indirect, weakly supported, or neither clearly plausible nor implausible.
4 = Agree
    - The association is biologically plausible and not contradicted by the input.
5 = Strongly Agree
    - The association is strongly biologically plausible and well supported by the input.

Important:
- Judge plausibility, not whether the association is definitively proven.
- Do not require direct experimental evidence if the mechanistic rationale is coherent.
- Penalize contradictions heavily.
- Prefer biological consistency over rhetorical confidence.

Return valid JSON only:
{
  "score": <1-5>,
  "label": "<rubric label>",
  "rationale": "<brief explanation>",
  "contradicted_by_input": <true or false>,
  "mechanistically_coherent": <true or false>
}
""".strip()


def build_plausibility_user_prompt(summary: str, drug_name: str, gene_name: str) -> str:
    """Build the plausibility user prompt.

    Args:
        summary: Input summary text to judge.
        drug_name: Drug name.
        gene_name: Gene symbol.

    Returns:
        Formatted user prompt.
    """
    return f"""
Evaluate whether the following proposed association is biologically plausible.

Drug: {drug_name}
Gene: {gene_name}

Input summary:
{summary}
""".strip()


# -----------------------------------------------------------------------------
# Faithfulness prompt (from 10_DrugAgent_faithfullness.ipynb)
# -----------------------------------------------------------------------------

FAITHFULNESS_SYSTEM_PROMPT = """
You are a careful biomedical judge.

Your task is to evaluate whether a generated explanation is faithful to the evidence.

Faithfulness means:
1. The explanation is grounded in the provided evidence.
2. The explanation does not contradict the evidence.
3. The explanation does not introduce unsupported claims.
4. The explanation does not omit critical information needed to support the claim.

Score using this scale:
1 = Highly unfaithful
2 = Mostly unfaithful
3 = Mixed / uncertain
4 = Mostly faithful
5 = Fully faithful

Return valid JSON only, with this exact schema:
{
  "overall_score": 1-5,
  "grounded_in_input": true/false,
  "has_contradiction": true/false,
  "has_unsupported_claim": true/false,
  "missing_critical_info": true/false,
  "rationale": "one short sentence"
}

Return valid JSON only.
Use double quotes for all keys and string values.
Do not use single quotes.
Do not wrap the JSON in markdown code fences.
Do not add any extra text.
Do not output any text outside the JSON.
""".strip()


def build_faithfulness_user_prompt(
    ml_evidence: str,
    kg_evidence: str,
    rag_evidence: str,
    fusion_reason: str,
) -> str:
    """Build the faithfulness user prompt.

    Args:
        ml_evidence: ML evidence text.
        kg_evidence: KG evidence text.
        rag_evidence: RAG evidence text.
        fusion_reason: Fusion explanation to evaluate.

    Returns:
        Formatted user prompt.
    """
    return f"""
Evaluate the faithfulness of the fusion explanation relative to the evidence.

ML evidence:
{ml_evidence}

KG evidence:
{kg_evidence}

RAG evidence:
{rag_evidence}

Fusion explanation:
{fusion_reason}
""".strip()


# -----------------------------------------------------------------------------
# Minimal flow notes (implementation-agnostic)
# -----------------------------------------------------------------------------

FLOW_NOTES: Dict[str, str] = {
    "plausibility": (
        "Given a drug, gene, and a summary, call the judge model with "
        "PLAUSIBILITY_SYSTEM_PROMPT + build_plausibility_user_prompt(). "
        "Parse the JSON response and record score/label and boolean flags."
    ),
    "faithfulness": (
        "Given ML/KG/RAG evidence and the fusion explanation, call the judge model "
        "with FAITHFULNESS_SYSTEM_PROMPT + build_faithfulness_user_prompt(). "
        "Parse the JSON response and store overall_score and evidence checks."
    ),
}
