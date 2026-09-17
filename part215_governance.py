"""
Task 15 — AI Governance

Requirements covered:
1. Application layer:
   - Least-autonomy enforcement.
   - Only the Lookup Agent may use check_support_ticket_status.
   - Retrieval Agent and Response Composer are explicitly denied access.

2. Risk classification:
   - Customer-support ticket work is classified as Medium risk.
   - Includes a one-paragraph justification.

3. Runtime layer:
   - Per-request token/cost budget cap.
   - Normal request is accepted.
   - Deliberately oversized request is rejected.
   - Budget rejection happens before execution; it does not silently exceed the cap.

This file is intentionally separate from part27_crew.py so the working
CrewAI implementation remains unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from part27_crew import build_crew


# ============================================================
# 1. APPLICATION-LAYER LEAST-AUTONOMY GOVERNANCE
# ============================================================

LOOKUP_AGENT_NAME = "Ola Support Ticket Lookup Agent"
COLLEGE_LOOKUP_CAPABILITY = "check_support_ticket_status"
ACTUAL_LOOKUP_TOOL_NAME = "support_ticket_lookup"

# Governance checks use the actual declared CrewAI tool name.
LOOKUP_TOOL_NAME = ACTUAL_LOOKUP_TOOL_NAME


class GovernanceViolation(PermissionError):
    """Raised when an agent attempts an action outside its allowed autonomy."""


def enforce_tool_permission(agent_name: str, tool_name: str) -> None:
    """
    Application-layer allow-list.

    Only the Lookup Agent is allowed to call the ticket-status tool.
    Every other agent is denied.
    """
    if tool_name == LOOKUP_TOOL_NAME and agent_name != LOOKUP_AGENT_NAME:
        raise GovernanceViolation(
            f"DENIED: {agent_name} is not allowed to call "
            f"{LOOKUP_TOOL_NAME}. Only {LOOKUP_AGENT_NAME} may call it."
        )

    print(
        f"ALLOWED: {agent_name} may use {tool_name}."
    )


def verify_crewai_tool_wiring() -> None:
    """
    Inspect the actual CrewAI agent configuration and verify that
    ticket lookup is exposed only to the Lookup Agent.

    We identify agents by their tool permissions rather than assuming
    that CrewAI's role string has a particular exact value.
    """
    crew = build_crew(
        "What is the status of support ticket TKT-0001?",
        "lookup",
    )

    if len(crew.agents) != 3:
        raise AssertionError(
            f"Expected 3 CrewAI agents, found {len(crew.agents)}."
        )

    def tool_names(agent: Any) -> set[str]:
        return {
            getattr(tool, "name", "")
            for tool in (agent.tools or [])
        }

    agent_tool_map = {}

    for agent in crew.agents:
        names = tool_names(agent)
        agent_tool_map[agent] = names

        print(
            f"  Agent: {agent.role}"
        )
        print(
            f"    Tools: {sorted(names)}"
        )

    lookup_agents = [
        agent
        for agent, names in agent_tool_map.items()
        if LOOKUP_TOOL_NAME in names
    ]

    if len(lookup_agents) != 1:
        raise AssertionError(
            "Expected exactly one CrewAI agent to have "
            f"{LOOKUP_TOOL_NAME}, but found {len(lookup_agents)}."
        )

    lookup_agent = lookup_agents[0]

    if lookup_agent.role != LOOKUP_AGENT_NAME:
        raise AssertionError(
            "The agent with the ticket-status tool is not the expected "
            f"Lookup Agent.\n"
            f"Expected: {LOOKUP_AGENT_NAME}\n"
            f"Actual: {lookup_agent.role}"
        )

    restricted_agents = [
        agent
        for agent, names in agent_tool_map.items()
        if agent is not lookup_agent and LOOKUP_TOOL_NAME in names
    ]

    if restricted_agents:
        raise AssertionError(
            "Governance failure: another agent has the ticket-status tool: "
            + ", ".join(agent.role for agent in restricted_agents)
        )

    print(
        f"  Lookup Agent confirmed: {lookup_agent.role}"
    )
    print(
        f"  Only this agent has the actual tool: {LOOKUP_TOOL_NAME}"
    )
    print(
        f"  This implements the college-required capability: "
        f"{COLLEGE_LOOKUP_CAPABILITY}"
    )
    print("CREWAI LEAST-AUTONOMY WIRING TEST PASSED")


def demonstrate_least_autonomy() -> None:
    """
    Deliberately attempt the restricted action from a non-privileged
    agent and confirm that governance rejects it.
    """
    non_privileged_agent = "Ola Support Retrieval Agent"

    try:
        enforce_tool_permission(
            non_privileged_agent,
            LOOKUP_TOOL_NAME,
        )
    except GovernanceViolation as exc:
        print("RESTRICTION TEST PASSED")
        print(f"  {exc}")
    else:
        raise AssertionError(
            "Governance failure: non-Lookup agent was allowed "
            "to use the ticket-status tool."
        )

    enforce_tool_permission(
        LOOKUP_AGENT_NAME,
        LOOKUP_TOOL_NAME,
    )


# ============================================================
# 2. RISK CLASSIFICATION
# ============================================================

@dataclass(frozen=True)
class RiskClassification:
    level: str
    justification: str


def classify_risk() -> RiskClassification:
    """
    College-defined risk classification for this task.

    The project handles customer-support tickets, which the
    official instructions classify as Medium risk.
    """
    return RiskClassification(
        level="Medium",
        justification=(
            "This project is classified as Medium risk because it performs "
            "customer-support ticket processing. The system can retrieve "
            "support information and generate customer-facing responses, "
            "so incorrect or unsupported output could affect a support "
            "interaction. The official governance guidance places "
            "customer-support tickets in the Medium-risk category, while "
            "Low risk covers summarization/transcription and High risk "
            "covers areas such as medical data, hiring, and financial data."
        ),
    )


def verify_risk_classification() -> None:
    risk = classify_risk()

    if risk.level not in {"Low", "Medium", "High"}:
        raise AssertionError(
            f"Invalid risk level: {risk.level}"
        )

    if risk.level != "Medium":
        raise AssertionError(
            f"Expected Medium risk for customer-support tickets, "
            f"got {risk.level}"
        )

    if not risk.justification.strip():
        raise AssertionError("Risk justification is empty.")

    print("RISK CLASSIFICATION:")
    print(f"  Level: {risk.level}")
    print(f"  Justification: {risk.justification}")


# ============================================================
# 3. RUNTIME TOKEN/COST BUDGET GOVERNANCE
# ============================================================

@dataclass(frozen=True)
class RuntimeBudget:
    max_input_tokens: int
    max_cost_usd: float


@dataclass(frozen=True)
class BudgetEstimate:
    input_tokens: int
    estimated_cost_usd: float


class BudgetExceededError(RuntimeError):
    """Raised when a request exceeds the configured runtime budget."""


DEFAULT_BUDGET = RuntimeBudget(
    max_input_tokens=400,
    max_cost_usd=0.002,
)


def estimate_tokens(text: str) -> int:
    """
    Deterministic local token estimate.

    This is deliberately simple because Task 15 requires a simulated
    runtime budget rather than a real paid-model billing calculation.
    """
    words = re.findall(r"\S+", text)

    if not words:
        return 0

    # Conservative deterministic approximation:
    # approximately 1 token per word.
    return len(words)


def estimate_cost(input_tokens: int) -> float:
    """
    Deterministic simulated cost.

    The rate is intentionally fixed and local; no API is called.
    """
    cost_per_token = 0.000005
    return input_tokens * cost_per_token


def check_runtime_budget(
    request_text: str,
    budget: RuntimeBudget = DEFAULT_BUDGET,
) -> BudgetEstimate:
    """
    Check the request before execution.

    If either the token limit or cost limit is exceeded, reject it.
    """
    input_tokens = estimate_tokens(request_text)
    estimated_cost = estimate_cost(input_tokens)

    if input_tokens > budget.max_input_tokens:
        raise BudgetExceededError(
            "REQUEST REJECTED: input-token budget exceeded. "
            f"Estimated={input_tokens}, "
            f"Limit={budget.max_input_tokens}."
        )

    if estimated_cost > budget.max_cost_usd:
        raise BudgetExceededError(
            "REQUEST REJECTED: cost budget exceeded. "
            f"Estimated=${estimated_cost:.6f}, "
            f"Limit=${budget.max_cost_usd:.6f}."
        )

    return BudgetEstimate(
        input_tokens=input_tokens,
        estimated_cost_usd=estimated_cost,
    )


def demonstrate_runtime_budget() -> None:
    """
    Test one normal request and one deliberately oversized request.
    """
    normal_request = (
        "What is the status of support ticket TKT-0001?"
    )

    normal_estimate = check_runtime_budget(normal_request)

    print("NORMAL REQUEST:")
    print(f"  Estimated tokens: {normal_estimate.input_tokens}")
    print(
        f"  Estimated cost: "
        f"${normal_estimate.estimated_cost_usd:.6f}"
    )
    print("  BUDGET TEST PASSED: request accepted.")

    oversized_request = "support " * 401

    try:
        check_runtime_budget(oversized_request)
    except BudgetExceededError as exc:
        print("OVERSIZED REQUEST:")
        print("  RUNTIME BUDGET TEST PASSED")
        print(f"  {exc}")
    else:
        raise AssertionError(
            "Governance failure: oversized request was not rejected."
        )


# ============================================================
# 4. FULL TASK 15 TEST
# ============================================================

def main() -> None:
    print("=" * 70)
    print("TASK 15 — AI GOVERNANCE")
    print("=" * 70)

    print("\n[1] Verifying actual CrewAI tool wiring...")
    verify_crewai_tool_wiring()

    print("\n[2] Demonstrating least-autonomy enforcement...")
    demonstrate_least_autonomy()

    print("\n[3] Verifying risk classification...")
    verify_risk_classification()

    print("\n[4] Verifying runtime token/cost budget...")
    demonstrate_runtime_budget()

    print("\n" + "=" * 70)
    print("TASK 15 LOCAL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()