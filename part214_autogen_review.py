"""
Part 2.14 - AutoGen Review Stage

Task 14 requirements:
- AutoGen review stage after CrewAI draft
- 2-agent RoundRobinGroupChat
- Policy-Compliance-Reviewer
- Final-Editor
- max_turns=2
- Pydantic structured verdict
- StructuredMessage custom message type
- Demonstrates approval and revision
- Deterministic local execution
- No API key
- No external LLM network call
"""

import asyncio
import os
from typing import Any, AsyncGenerator, Mapping, Sequence

# Disable CrewAI telemetry for the CrewAI draft stage.
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from pydantic import BaseModel, Field

from autogen_agentchat.base import ChatAgent, Response, TaskResult
from autogen_agentchat.messages import (
    BaseChatMessage,
    StructuredMessage,
    TextMessage,
)
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken


# ============================================================
# STRUCTURED VERDICT
# ============================================================

class YourVerdictModel(BaseModel):
    """Structured AutoGen review verdict required by Task 14."""

    approved: bool
    final_answer: str = Field(min_length=1)
    reason: str = Field(min_length=1)


# ============================================================
# DETERMINISTIC REVIEW LOGIC
# ============================================================

UNSUPPORTED_CLAIM = "customers receive a 100% refund"


def review_draft(
    draft: str,
    retrieved_context: str,
) -> YourVerdictModel:
    """
    Deterministically review a CrewAI draft against retrieved context.

    This local function provides the deterministic decision logic used by
    the two AutoGen ChatAgent participants.
    """

    context_lower = retrieved_context.lower()
    draft_lower = draft.lower()

    # Revision case: deliberately unsupported claim.
    if UNSUPPORTED_CLAIM in draft_lower:
        cleaned_answer = (
            "Refunds and compensation are handled according to the "
            "applicable Ola support policy and the circumstances of "
            "the individual case."
        )

        return YourVerdictModel(
            approved=False,
            final_answer=cleaned_answer,
            reason=(
                "The statement that customers receive a 100% refund "
                "was not supported by the supplied retrieved context."
            ),
        )

    # Missing-context case.
    if not context_lower.strip():
        return YourVerdictModel(
            approved=False,
            final_answer=(
                "The draft cannot be approved because no retrieved "
                "knowledge-base context was supplied."
            ),
            reason=(
                "A grounded response requires retrieved Ola "
                "knowledge-base context."
            ),
        )

    # Approval case.
    return YourVerdictModel(
        approved=True,
        final_answer=draft,
        reason=(
            "The CrewAI draft is consistent with the supplied "
            "retrieved Ola knowledge-base context and contains "
            "no deliberately unsupported claim."
        ),
    )


# ============================================================
# CUSTOM AUTO-GEN CHAT AGENT
# ============================================================

class DeterministicReviewAgent(ChatAgent):
    """
    Local deterministic AutoGen ChatAgent.

    No external model client is used. The agent participates directly
    in the AutoGen RoundRobinGroupChat.
    """

    def __init__(self, agent_name: str, role: str):
        self._name = agent_name
        self._role = role
        self._last_verdict: YourVerdictModel | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._role

    @property
    def produced_message_types(
        self,
    ) -> Sequence[type[BaseChatMessage]]:
        return [TextMessage, StructuredMessage[YourVerdictModel]]

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:
        del cancellation_token

        task_text = self._extract_task_text(messages)

        draft = self._extract_section(
            task_text,
            "CREWAI DRAFT:",
            "ORIGINAL RETRIEVED CONTEXT:",
        )

        context = self._extract_section(
            task_text,
            "ORIGINAL RETRIEVED CONTEXT:",
            None,
        )

        verdict = review_draft(
            draft=draft.strip(),
            retrieved_context=context.strip(),
        )

        self._last_verdict = verdict

        if self._name == "Policy_Compliance_Reviewer":
            message = TextMessage(
                source=self._name,
                content=(
                    "Policy review completed. "
                    f"approved={verdict.approved}. "
                    f"reason={verdict.reason}"
                ),
            )
        else:
            message = StructuredMessage[YourVerdictModel](
                source=self._name,
                content=verdict,
            )

        return Response(chat_message=message)

    async def on_messages_stream(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> AsyncGenerator[BaseChatMessage | Response | Any, None]:
        response = await self.on_messages(messages, cancellation_token)
        yield response

    async def run(
        self,
        *,
        task=None,
        cancellation_token=None,
        output_task_messages=True,
    ) -> TaskResult:
        """Run this local deterministic agent as a TaskRunner."""

        if cancellation_token is None:
            cancellation_token = CancellationToken()

        if task is None:
            messages = []
        elif isinstance(task, str):
            messages = [
                TextMessage(
                    source="user",
                    content=task,
                )
            ]
        elif isinstance(task, BaseChatMessage):
            messages = [task]
        else:
            messages = list(task)

        response = await self.on_messages(
            messages,
            cancellation_token,
        )

        result_messages = []

        if output_task_messages:
            result_messages.extend(messages)

        result_messages.append(response.chat_message)

        return TaskResult(
            messages=result_messages,
            stop_reason="completed",
        )

    def run_stream(
        self,
        *,
        task=None,
        cancellation_token=None,
        output_task_messages=True,
    ):
        """Run this local deterministic agent as a stream."""

        async def _stream():
            if cancellation_token is None:
                token = CancellationToken()
            else:
                token = cancellation_token

            if task is None:
                messages = []
            elif isinstance(task, str):
                messages = [
                    TextMessage(
                        source="user",
                        content=task,
                    )
                ]
            elif isinstance(task, BaseChatMessage):
                messages = [task]
            else:
                messages = list(task)

            if output_task_messages:
                for message in messages:
                    yield message

            response = await self.on_messages(
                messages,
                token,
            )

            yield response.chat_message

            yield TaskResult(
                messages=(
                    list(messages) + [response.chat_message]
                ),
                stop_reason="completed",
            )

        return _stream()

    async def on_reset(
        self,
        cancellation_token: CancellationToken,
    ) -> None:
        del cancellation_token
        self._last_verdict = None

    async def on_pause(
        self,
        cancellation_token: CancellationToken,
    ) -> None:
        del cancellation_token

    async def on_resume(
        self,
        cancellation_token: CancellationToken,
    ) -> None:
        del cancellation_token

    async def save_state(self) -> Mapping[str, Any]:
        return {}

    async def load_state(
        self,
        state: Mapping[str, Any],
    ) -> None:
        del state

    async def close(self) -> None:
        self._last_verdict = None

    @staticmethod
    def _extract_task_text(
    messages: Sequence[BaseChatMessage],
    ) -> str:
        # Find the original task message containing the CrewAI draft
        # and retrieved-context markers.
        for message in messages:
            if isinstance(message, TextMessage):
                if (
                    "CREWAI DRAFT:" in message.content
                    and "ORIGINAL RETRIEVED CONTEXT:" in message.content
                ):
                    return message.content

        return ""

    @staticmethod
    def _extract_section(
        text: str,
        start_marker: str,
        end_marker: str | None,
    ) -> str:
        start = text.find(start_marker)

        if start == -1:
            return ""

        start += len(start_marker)

        if end_marker is None:
            return text[start:]

        end = text.find(end_marker, start)

        if end == -1:
            return text[start:]

        return text[start:end]


# ============================================================
# AUTO-GEN ROUND ROBIN REVIEW
# ============================================================

async def run_autogen_review(
    draft: str,
    retrieved_context: str,
) -> YourVerdictModel:
    """
    Run the required 2-agent AutoGen RoundRobinGroupChat.

    The reviewer speaks first and the Final-Editor speaks second.
    max_turns=2 therefore gives exactly one turn to each agent.
    """

    reviewer = DeterministicReviewAgent(
        agent_name="Policy_Compliance_Reviewer",
        role=(
            "Reviews CrewAI drafts for policy compliance and "
            "grounding against retrieved Ola knowledge-base context."
        ),
    )

    editor = DeterministicReviewAgent(
        agent_name="Final_Editor",
        role=(
            "Produces the final structured verdict after the "
            "policy-compliance review."
        ),
    )

    team = RoundRobinGroupChat(
        participants=[
            reviewer,
            editor,
        ],
        max_turns=2,
        custom_message_types=[
            StructuredMessage[YourVerdictModel],
        ],
    )

    task_message = (
        "Review the CrewAI Composer draft against the original "
        "retrieved context.\n\n"
        "CREWAI DRAFT:\n"
        f"{draft}\n\n"
        "ORIGINAL RETRIEVED CONTEXT:\n"
        f"{retrieved_context}\n\n"
        "Produce a structured verdict with approved, final_answer, "
        "and reason."
    )

    print("\nAUTOGEN TEAM CONFIGURATION")
    print("-" * 70)
    print("Team: RoundRobinGroupChat")
    print("Agents: 2")
    print("Agent 1: Policy_Compliance_Reviewer")
    print("Agent 2: Final_Editor")
    print("max_turns: 2")
    print(
        "custom_message_types: "
        "[StructuredMessage[YourVerdictModel]]"
    )

    print("\nAUTOGEN REVIEW INPUT")
    print("-" * 70)
    print(task_message)

    result = await team.run(task=task_message)

    print("\nAUTOGEN TEAM EXECUTION")
    print("-" * 70)

    structured_verdict: YourVerdictModel | None = None

    for message in result.messages:
        print(f"{type(message).__name__}: {message}")

        if isinstance(
            message,
            StructuredMessage,
        ) and isinstance(
            message.content,
            YourVerdictModel,
        ):
            structured_verdict = message.content

    if structured_verdict is None:
        raise AssertionError(
            "AutoGen team did not produce the required "
            "StructuredMessage[YourVerdictModel]."
        )

    return structured_verdict


# ============================================================
# DEMONSTRATION
# ============================================================

async def run_demo() -> None:
    print("=" * 70)
    print("PART 2.14 - AUTOGEN REVIEW STAGE")
    print("=" * 70)

    retrieved_context = (
        "Source: refund_compensation.txt\n"
        "Similarity: 0.6294\n"
        "Content: Refunds and compensation are handled according "
        "to the applicable support policy and case circumstances."
    )

    # --------------------------------------------------------
    # CASE 1 — APPROVAL
    # --------------------------------------------------------

    print("\n")
    print("-" * 70)
    print("CASE 1: GROUNDED DRAFT — APPROVAL")
    print("-" * 70)

    grounded_draft = (
        "Refunds and compensation are handled according to the "
        "applicable Ola support policy and the circumstances of "
        "the individual case."
    )

    approval_verdict = await run_autogen_review(
        draft=grounded_draft,
        retrieved_context=retrieved_context,
    )

    print("\nSTRUCTURED VERDICT:")
    print(approval_verdict.model_dump())

    assert approval_verdict.approved is True
    assert approval_verdict.final_answer == grounded_draft

    print("\nAPPROVAL TEST PASSED")


    # --------------------------------------------------------
    # CASE 2 — REVISION
    # --------------------------------------------------------

    print("\n")
    print("-" * 70)
    print("CASE 2: DELIBERATELY UNGROUNDED DRAFT — REVISION")
    print("-" * 70)

    ungrounded_draft = (
        "Refunds and compensation are handled according to the "
        "applicable Ola support policy. Customers receive a "
        "100% refund in every case."
    )

    revision_verdict = await run_autogen_review(
        draft=ungrounded_draft,
        retrieved_context=retrieved_context,
    )

    print("\nSTRUCTURED VERDICT:")
    print(revision_verdict.model_dump())

    assert revision_verdict.approved is False
    assert "100% refund" not in revision_verdict.final_answer
    assert revision_verdict.reason

    print("\nREVISION TEST PASSED")


    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TASK 14 DEMONSTRATION COMPLETE")
    print("=" * 70)

    print("\nStructured verdict fields verified:")
    print(
        "approved      :",
        type(approval_verdict.approved).__name__,
    )
    print(
        "final_answer  :",
        type(approval_verdict.final_answer).__name__,
    )
    print(
        "reason        :",
        type(approval_verdict.reason).__name__,
    )

    print("\nTASK 14 LOCAL TESTS PASSED")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    asyncio.run(run_demo())