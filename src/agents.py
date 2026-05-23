from typing import List

from autogen_agentchat.agents import AssistantAgent

from .kg_utils import get_dti_score as kg_score
from .ml_utils import get_dti_score as ml_score
from .rag_utils import get_dti_score as pubmed_score

EVIDENCE_SOURCES = ["ML", "KG", "PubMed"]


def _normalize_ablation(ablation: str) -> str:
    """Normalize ablation string for consistent comparisons.

    Args:
        ablation: Raw ablation string.

    Returns:
        Lowercased ablation value with whitespace removed.
    """
    return str(ablation).strip().lower()


def _resolve_enabled_sources(ablation: str) -> List[str]:
    """Resolve enabled evidence sources from an ablation flag.

    Args:
        ablation: Ablation mode such as full, no_ml, no_kg, no_rag.

    Returns:
        Ordered list of enabled evidence sources.
    """
    mode = _normalize_ablation(ablation)
    if mode == "minimal":
        return []
    sources = EVIDENCE_SOURCES[:]
    if mode == "no_ml" and "ML" in sources:
        sources.remove("ML")
    if mode == "no_kg" and "KG" in sources:
        sources.remove("KG")
    if mode in {"no_rag", "no_pubmed"} and "PubMed" in sources:
        sources.remove("PubMed")
    return sources


def _build_planning_system_message(enabled_sources: List[str]) -> str:
    """Build a planning agent system message for enabled sources.

    Args:
        enabled_sources: Enabled evidence source names.

    Returns:
        Planning agent system message string.
    """
    agent_lines = []
    if "PubMed" in enabled_sources:
        agent_lines.append("- PubmedAgent: Searches for information online.")
    if "ML" in enabled_sources:
        agent_lines.append("- MLAgent: Performs machine learning predictions.")
    if "KG" in enabled_sources:
        agent_lines.append("- KGAgent: Calculates scores from knowledge graph data.")
    agent_lines.append(
        "- SummaryAgent: Synthesizes findings and produces the final report."
    )
    team_agents = "\n".join(agent_lines)

    return f"""
You are the PlanningAgent.

Role:
- Decompose complex user requests into clear, manageable subtasks.
- Assign those subtasks to appropriate team agents.
- You DO NOT perform any tasks yourself.

Team Agents:
{team_agents}

Execution Rules:
- ALWAYS delegate tasks to ALL relevant agents.
- Use this format for assignments:
    <agent>: <task description>

Workflow:
1. Upon receiving a new task, decompose it and assign subtasks.
2. Wait for all agents to complete their work.
3. Do NOT output the word "TERMINATE". Only SummaryAgent outputs TERMINATE.
"""


def _build_summary_child_schema(source: str) -> str:
    """Build schema snippet for a single evidence source.

    Args:
        source: Evidence source name.

    Returns:
        JSON schema snippet string.
    """
    if source == "PubMed":
        return """{
        "type": "evidence_analysis",
        "source": "PubMed",
        "thought": string,
        "action": string,
        "observation": string,
        "score": float,
        "weight": float,
        "pmc_ids": [string]
      }"""
    return f"""{{\n        "type": "evidence_analysis",\n        "source": "{source}",\n        "thought": string,\n        "action": string,\n        "observation": string,\n        "score": float,\n        "weight": float\n      }}"""


def _build_summary_schema(enabled_sources: List[str]) -> str:
    """Build the reasoning tree schema based on enabled sources.

    Args:
        enabled_sources: Evidence sources to include.

    Returns:
        JSON schema string.
    """
    children = ",\n      ".join(
        _build_summary_child_schema(source) for source in enabled_sources
    )
    return f"""```json
{{
  "type": "reasoning_tree",
  "drug": string,
  "target": string,
  "root": {{
    "type": "comparison",
    "children": [
      {children}
    ],
    "final_score": float,
    "summary_reasoning": string
  }}
}}
```
"""


def _build_summary_system_message(enabled_sources: List[str]) -> str:
    """Build summary agent system message for enabled sources.

    Args:
        enabled_sources: Evidence sources to include.

    Returns:
        Summary agent system message string.
    """
    source_lines = []
    if "ML" in enabled_sources:
        source_lines.append("- ML-based prediction scores and reasoning")
    if "KG" in enabled_sources:
        source_lines.append("- Knowledge Graph (KG) evidence and reasoning")
    if "PubMed" in enabled_sources:
        source_lines.append("- PubMed search results and reasoning")
    sources_text = "\n".join(source_lines)
    schema_text = _build_summary_schema(enabled_sources)
    input_format = """You will receive a JSON object with some of the following keys (only include keys for available sources):
{
  "ml_evidence": {"drug": string, "target": string, "score": float, "reason": string},
  "kg_evidence": {"drug": string, "target": string, "score": float, "reason": string, "label": string?},
  "pubmed_evidence": {"drug": string, "target": string, "score": float, "reason": string, "label": string?, "pmc_ids": [string]}
}"""

    return f"""
Task:
You will synthesize evidence from the provided sources regarding drug-target interactions:
{sources_text}

Your role is to:
1. Analyze each evidence source independently and summarize its reasoning using a structured format.
2. Compare and contrast the results, identifying agreements and conflicts.
3. Assess the biological plausibility and reliability of each source's reasoning.
4. Use provided categorical labels from KG and PubMed evidence when available.
5. Do NOT compute the final decision; it is computed deterministically downstream.

You must output a symbolic reasoning tree in valid JSON format.

### Output requirements (MUST follow strictly):

- Output **ONLY a valid JSON object**, nothing before or after.
- The JSON must exactly match the schema below. All property names must be **enclosed in double quotes**.
- No markdown, no explanation, no comments, no additional text.
- All float values must be JSON numbers (e.g., 0.75), not strings or formulas.
- The sum of weights in `children` must be exactly 1.0.
- `"final_score"` must be a float based on available evidence (do not use decision rules).
- `"pmc_ids"` must be an array of strings, even if only one.
- After the JSON object, output the word `TERMINATE` **on a new line by itself**.

### Label handling (MUST follow strictly):

- For KG and PubMed, if a `label` is provided in the evidence, use it directly.
- If no label is provided, you may judge evidence strength from the reasoning text.
- In each `observation` field, explicitly include the chosen label.
- For PubMed, always include **all** received PMC IDs in `pmc_ids` with no omissions. If PMC IDs are not provided explicitly, extract all `PMC` IDs from the PubMed reasoning text.
- If a source is missing from the input, omit that source from `children` and renormalize weights to sum to 1.0.

### Label definitions (use only if you must infer a label):

- STRONG: direct physical binding for this exact drug-target pair is shown
  (e.g., Kd/Ki/IC50/EC50, SPR/ITC, competition binding, pull-down, structure/cryo-EM).
- MODERATE: no binding measurement, but explicit direct action for this exact
  drug-target pair is stated (e.g., "drug inhibits/poisons/targets <target>")
  or curated database records a direct action edge (DrugBank/ChEMBL/CTD).
- WEAK: indirect or pathway-level associations, biomarker correlations, multi-hop
  traces (3+ hops), or low-specificity intermediates.
- NONE: no direct or indirect evidence.

### Output schema:

{schema_text}

After this JSON block, output:

TERMINATE

Input format:
{input_format}
Use these values to fill in the corresponding parts of the output JSON. Base your reasoning entirely on this evidence. Preserve PMC IDs from the PubMed evidence.
"""


def build_summary_agent(
    model_client, ablation: str = "full", enabled_sources: List[str] | None = None
):
    """Build the summary agent with an ablation-aware prompt.

    Args:
        model_client: LLM client instance.
        ablation: Ablation mode for enabled sources.
        enabled_sources: Optional explicit list of sources to include.

    Returns:
        Configured SummaryAgent instance.
    """
    if enabled_sources is None:
        enabled_sources = _resolve_enabled_sources(ablation)

    if enabled_sources:
        system_message = _build_summary_system_message(enabled_sources)
    else:
        system_message = """
Task:
You will synthesize evidence from the provided sources regarding drug-target interactions.

Your role is to:
1. Analyze each evidence source independently.
2. Use provided categorical labels when available.
3. Do NOT compute the final decision; it is computed deterministically downstream.

You must output a valid JSON object with the following fields only:

{
  "drug": string,
  "target": string,
  "final_score": float,
  "summary_reasoning": string
}

After this JSON object, output:

TERMINATE

Label handling (MUST follow strictly):
- For KG and PubMed, if a `label` is provided in the evidence, use it directly.
- If no label is provided, you may judge evidence strength from the reasoning text.

Input format:
You will receive inputs as a JSON object containing only the available evidences.
"""
    return AssistantAgent(
        "SummaryAgent",
        description="Synthesize evidence and provide reasoning, format depends on ablation mode.",
        model_client=model_client,
        system_message=system_message,
    )


def get_summary_system_message(
    ablation: str = "full", enabled_sources: List[str] | None = None
) -> str:
    """Return the summary agent system message for a given ablation.

    Args:
        ablation: Ablation mode for enabled sources.
        enabled_sources: Optional explicit list of sources to include.

    Returns:
        Summary agent system message string.
    """
    if enabled_sources is None:
        enabled_sources = _resolve_enabled_sources(ablation)
    if enabled_sources:
        return _build_summary_system_message(enabled_sources)
    return """
Task:
You will synthesize evidence from the provided sources regarding drug-target interactions.

Your role is to:
1. Analyze each evidence source independently.
2. Use provided categorical labels when available.
3. Do NOT compute the final decision; it is computed deterministically downstream.

You must output a valid JSON object with the following fields only:

{
  "drug": string,
  "target": string,
  "final_score": float,
  "summary_reasoning": string
}

After this JSON object, output:

TERMINATE

Label handling (MUST follow strictly):
- For KG and PubMed, if a `label` is provided in the evidence, use it directly.
- If no label is provided, you may judge evidence strength from the reasoning text.

Input format:
You will receive inputs as a JSON object containing only the available evidences.
"""


def build_agents(model_client, ablation: str = "full"):
    agents = {}
    enabled_sources = _resolve_enabled_sources(ablation)
    agents["planning_agent"] = AssistantAgent(
        "PlanningAgent",
        description="Delegates tasks to specialist agents. First to act on any new user request.",
        model_client=model_client,
        system_message=_build_planning_system_message(enabled_sources),
    )

    # Conditionally include others
    if ablation not in ["no_pubmed", "no_rag", "minimal"]:
        agents["pubmed_agent"] = AssistantAgent(
            "PubmedAgent",
            description="Searches for relevant evidence from PubMed based on drug–target pairs.",
            tools=[pubmed_score],
            model_client=model_client,
            system_message="""
        You are the PubmedAgent.

        Role:
        - Search for supporting information about drug–target interactions from PubMed.
        - Use only the `pubmed_score` tool.

        Input Format:
        {
            "drug": <drug_name>,
            "target": <target_name>
        }

        Rules:
        - Make only one RAG call per task.
        - Do not perform calculations or draw conclusions from RAG results.
        - Return raw or summarized information found.
        - This task defines DTI as physical, direct binding only.
        - LABEL=STRONG is allowed only if the evidence shows the drug directly binds the target protein for this exact drug-target pair.
        - Acceptable direct-binding evidence includes quantitative binding/inhibition (Kd/Ki/IC50/EC50), SPR/ITC, competition binding, pull-down, structural determination (crystal/cryo-EM).
        - Do NOT treat biomarker correlations, pathway-level relations (DDR/HRD/BRCAness), or network traces as direct binding evidence.
        - If evidence is indirect or biomarker-only, do not label it STRONG.

        Your responsibility ends after presenting the RAG results.
        """,
        )
    if ablation not in ["no_ml", "minimal"]:
        agents["ml_agent"] = AssistantAgent(
            "MLAgent",
            description="Performs machine learning-based DTI predictions.",
            model_client=model_client,
            tools=[ml_score],
            system_message="""
          You are the MLAgent.

          Role:
          - Use the `ml_score` tool to predict drug–target interaction (DTI) scores based on machine learning models.

          Input Format:
          {
              "drug": <drug_name>,
              "target": <target_name>
          }

          Rules:
          - Use only `ml_score`.
          - Make only one prediction call per task.
          - Do not perform any additional calculations or interpretation.
          - Return the predicted score and any reasoning provided by the model.
          """,
        )
    if ablation not in ["no_kg", "minimal"]:
        agents["kg_agent"] = AssistantAgent(
            "KGAgent",
            description="Gathers DTI information using knowledge graph data.",
            model_client=model_client,
            tools=[kg_score],
            system_message="""
You are the KGAgent.

Role:
- Use the `kg_score` tool to predict drug–target interaction (DTI) scores from knowledge graph data.

Input Format:
{
    "drug": <drug_name>,
    "target": <target_name>
}

Rules:
- Use only `kg_score`.
- Make only one prediction call per task.
- Do not perform any further calculations or interpretation.
- Return the score and any reasoning from the tool.
""",
        )

    agents["summary_agent"] = build_summary_agent(
        model_client, ablation=ablation, enabled_sources=enabled_sources
    )

    return [
        agents["planning_agent"],
        agents.get("pubmed_agent"),
        agents.get("ml_agent"),
        agents.get("kg_agent"),
        agents["summary_agent"],
    ]
