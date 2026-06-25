#!/usr/bin/env python
# coding: utf-8
"""Agent construction utilities for DrugAgent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, _model_info

from drugagent.config import ALL_EVIDENCE_AGENTS, _get_config
from drugagent.summary import get_summary_system_message
from drugagent.tools import kg_score, ml_score, rag_score
from drugagent.utils import normalize_enabled_agents


def build_agents(
    model_client,
    ablation: str = "full",
    use_planning_agent: bool = False,
) -> Dict[str, AssistantAgent]:
    """Build Autogen agents wired to local tool wrappers.

    Args:
        model_client: Autogen model client.
        ablation: Ablation mode.
        use_planning_agent: Whether to include PlanningAgent.

    Returns:
        Mapping of agent names to AssistantAgent instances.
    """
    enabled_sources = normalize_enabled_agents(_resolve_enabled_sources(ablation))

    agents: Dict[str, AssistantAgent] = {}

    if use_planning_agent:
        planning_agent = AssistantAgent(
            "PlanningAgent",
            description="Delegates tasks to specialist agents. First to act on any new user request.",
            model_client=model_client,
            system_message=_build_planning_system_message(enabled_sources),
        )
        agents["planning_agent"] = planning_agent

    if "RAG" in enabled_sources:
        agents["rag_agent"] = AssistantAgent(
            "RAGAgent",
            description="Searches for relevant evidence from literature.",
            tools=[rag_score],
            model_client=model_client,
            system_message="""
        You are the RAGAgent.

        Role:
        - Use the `rag_score` tool to retrieve literature evidence for drug-target interactions.
        - Make only one RAG call per task.
        - Return raw or summarized information found.
        """,
        )

    if "ML" in enabled_sources:
        agents["ml_agent"] = AssistantAgent(
            "MLAgent",
            description="Performs machine learning-based DTI predictions.",
            model_client=model_client,
            tools=[ml_score],
            system_message="""
          You are the MLAgent.

          Role:
          - Use the `ml_score` tool to predict drug-target interaction (DTI) scores.
          - Make only one prediction call per task.
          - Return the predicted score and reasoning.
          """,
        )

    if "KG" in enabled_sources:
        agents["kg_agent"] = AssistantAgent(
            "KGAgent",
            description="Gathers DTI information using knowledge graph data.",
            model_client=model_client,
            tools=[kg_score],
            system_message="""
You are the KGAgent.

Role:
- Use the `kg_score` tool to predict drug-target interaction (DTI) scores from knowledge graph data.
- Make only one prediction call per task.
- Return the score and any reasoning from the tool.
""",
        )

    summary_agent = AssistantAgent(
        "SummaryAgent",
        description="Synthesizes evidence and produces structured reasoning.",
        model_client=model_client,
        system_message=get_summary_system_message(ablation, enabled_sources),
    )

    agents["summary_agent"] = summary_agent
    return agents


def _resolve_enabled_sources(ablation: str) -> List[str]:
    """Resolve enabled evidence sources from ablation flag.

    Args:
        ablation: Ablation mode.

    Returns:
        List of enabled sources.
    """
    mode = str(ablation).strip().lower()
    sources = ALL_EVIDENCE_AGENTS[:]
    if mode == "minimal":
        return []
    if mode == "no_ml" and "ML" in sources:
        sources.remove("ML")
    if mode == "no_kg" and "KG" in sources:
        sources.remove("KG")
    if mode in {"no_rag"} and "RAG" in sources:
        sources.remove("RAG")
    return sources


def _build_planning_system_message(enabled_sources: List[str]) -> str:
    """Build system message for the PlanningAgent.

    Args:
        enabled_sources: Enabled evidence sources.

    Returns:
        Planning agent system message.
    """
    agent_lines = []
    if "RAG" in enabled_sources:
        agent_lines.append("- RAGAgent: Searches for literature evidence.")
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


def create_model_client(model_type: str, model_name: Optional[str] = None):
    """Create an Autogen model client for the requested provider.

    Args:
        model_type: Model provider type.
        model_name: Optional model name override.

    Returns:
        Autogen model client.
    """

    def resolve_model_name(
        preferred_name: Optional[str], deployment: Optional[str]
    ) -> Optional[str]:
        if preferred_name:
            return preferred_name
        return deployment

    def resolve_model_info(model: str) -> Dict[str, Any] | None:
        if not model:
            return None
        try:
            return _model_info.get_info(model)
        except Exception:
            if _model_info._MODEL_INFO:
                return next(iter(_model_info._MODEL_INFO.values()))
            return None

    if model_type == "azure":
        cfg = _get_config()
        model = resolve_model_name(model_name, cfg["deployment_name"])
        client_kwargs = {
            "azure_deployment": cfg["deployment_name"],
            "model": model,
            "api_version": cfg["api_version"],
            "azure_endpoint": cfg["endpoint"],
            "api_key": cfg["api_key"],
            "temperature": 0,
            "seed": 42,
        }
        model_info = resolve_model_info(model)
        if model_info is not None:
            client_kwargs["model_info"] = model_info
        return AzureOpenAIChatCompletionClient(**client_kwargs)
    raise ValueError(f"Unsupported model_type: {model_type}")
