#!/usr/bin/env python
# coding: utf-8
"""CLI entrypoint for DrugAgent."""

from __future__ import annotations

import argparse
import asyncio
from typing import List

from drugagent.agents import build_agents, create_model_client
from drugagent.config import ALL_EVIDENCE_AGENTS
from drugagent.orchestration import chat_with_agents_and_summarize
from drugagent.utils import generate_ablation_configs, normalize_enabled_agents


def get_active_agents(
    ablation: str = "full",
    model_client=None,
    enabled_agents: List[str] | None = None,
    return_enabled: bool = False,
    use_planning_agent: bool = False,
):
    """Create an ordered list of active agents for orchestration.

    Args:
        ablation: Ablation mode.
        model_client: Autogen model client.
        enabled_agents: Enabled evidence sources.
        return_enabled: Whether to return enabled agent names.
        use_planning_agent: Whether to include PlanningAgent.

    Returns:
        Agent list or (agent list, enabled agents).
    """
    agents = build_agents(
        model_client, ablation=ablation, use_planning_agent=use_planning_agent
    )
    planning_agent = agents.get("planning_agent")
    rag_agent = agents.get("rag_agent")
    ml_agent = agents.get("ml_agent")
    kg_agent = agents.get("kg_agent")
    summary_agent = agents["summary_agent"]

    if enabled_agents is None:
        enabled_agents = ALL_EVIDENCE_AGENTS[:]

    mode = str(ablation).strip().lower()
    if mode == "no_rag":
        enabled_agents = [a for a in enabled_agents if a != "RAG"]
    elif mode == "no_ml":
        enabled_agents = [a for a in enabled_agents if a != "ML"]
    elif mode == "no_kg":
        enabled_agents = [a for a in enabled_agents if a != "KG"]
    elif mode == "full":
        pass
    else:
        raise ValueError(f"Unknown ablation type: {ablation}")

    enabled_agents = normalize_enabled_agents(enabled_agents)
    agent_map = {"RAG": rag_agent, "ML": ml_agent, "KG": kg_agent}
    participants = [
        *([planning_agent] if planning_agent is not None else []),
        *[agent_map[a] for a in enabled_agents if agent_map.get(a)],
        summary_agent,
    ]

    if return_enabled:
        return participants, enabled_agents
    return participants


def main() -> None:
    """Run the DrugAgent CLI."""
    parser = argparse.ArgumentParser(description="Drug-Target Interaction CLI Agent")
    parser.add_argument(
        "--drug", type=str, required=True, help="Drug name (e.g., Gefitinib)"
    )
    parser.add_argument(
        "--gene", type=str, required=True, help="Target gene name (e.g., TOP1)"
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        help="Ablation type (e.g., full, no_ml, no_kg, no_rag)",
    )
    parser.add_argument(
        "--enabled_agents",
        type=str,
        default=None,
        help="Comma-separated list of enabled agents (ML,KG,RAG).",
    )
    parser.add_argument(
        "--run_all_ablations",
        action="store_true",
        help="Run all non-empty enabled agent combinations.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="azure",
        choices=["azure"],
        help="Which LLM provider to use",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name to use for the selected LLM provider",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        help="Override reasoning effort (low, medium, high, none).",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable cache usage (skip CSV cache checks and writes).",
    )
    parser.add_argument(
        "--use_planning_agent",
        action="store_true",
        help="Enable PlanningAgent in the multi-agent orchestration (default: disabled).",
    )

    parser.add_argument(
        "--fast_mode",
        action="store_true",
        help="Enable fast mode; use parallel tool execution instead of SelectorGroupChat.",
    )
    parser.add_argument(
        "--no_fast_mode",
        action="store_true",
        help="Deprecated. Use --fast_mode instead.",
    )

    args = parser.parse_args()
    model_client = create_model_client(args.model_type, args.model_name)
    enabled_agents = None
    if args.enabled_agents:
        enabled_agents = [
            part.strip() for part in args.enabled_agents.split(",") if part.strip()
        ]

    reasoning_effort = args.reasoning_effort
    if reasoning_effort and reasoning_effort.strip().lower() == "none":
        reasoning_effort = None

    if args.run_all_ablations:
        for combo in generate_ablation_configs():
            try:
                active_agents, normalized_agents = get_active_agents(
                    ablation=args.ablation,
                    model_client=model_client,
                    enabled_agents=combo,
                    return_enabled=True,
                    use_planning_agent=args.use_planning_agent,
                )
            except ValueError as exc:
                print(str(exc))
                exit(1)

            if not active_agents:
                print("Error: No agents enabled after ablation.")
                exit(1)

            asyncio.run(
                chat_with_agents_and_summarize(
                    args.drug,
                    args.gene,
                    active_agents=active_agents,
                    enabled_agents=normalized_agents,
                    ablation=args.ablation,
                    model_client=model_client,
                    model=None,
                    cache_enabled=not args.no_cache,
                    reasoning_effort=reasoning_effort,
                    fast_mode=args.fast_mode and not args.no_fast_mode,
                )
            )
        return

    try:
        active_agents, normalized_agents = get_active_agents(
            ablation=args.ablation,
            model_client=model_client,
            enabled_agents=enabled_agents,
            return_enabled=True,
            use_planning_agent=args.use_planning_agent,
        )
    except ValueError as exc:
        print(str(exc))
        exit(1)

    if not active_agents:
        print("Error: No agents enabled after ablation.")
        exit(1)

    asyncio.run(
        chat_with_agents_and_summarize(
            args.drug,
            args.gene,
            active_agents=active_agents,
            enabled_agents=normalized_agents,
            ablation=args.ablation,
            model_client=model_client,
            model=None,
            cache_enabled=not args.no_cache,
            reasoning_effort=reasoning_effort,
            fast_mode=args.fast_mode and not args.no_fast_mode,
        )
    )


if __name__ == "__main__":
    main()
