"""
Part 2.8 - Session Memory

Demonstrates in-process, session-scoped LangChain conversation memory alongside
the existing Part 2.7 CrewAI orchestration.  No API key, remote LLM, or network
call is used: follow-up ticket questions reuse the deterministic local ticket
lookup component that the Task 2.7 lookup tool already wraps.
"""

import os
import re

# Keep the CrewAI demonstration local and deterministic.
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from ticket_lookup import check_support_ticket_status


# This dictionary intentionally lives only in this Python process.  Restarting
# the program creates a new, empty store.
SESSION_HISTORIES: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return the in-memory history for one session, creating it if needed."""

    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = InMemoryChatMessageHistory()

    return SESSION_HISTORIES[session_id]


def _last_human_message(messages: list[BaseMessage]) -> str:
    """Get the current turn's user text from the message sequence."""

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)

    return ""


def _ticket_id_from_messages(messages: list[BaseMessage]) -> str | None:
    """Find the most recent ticket ID mentioned in this session's history."""

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            match = re.search(r"\bTKT-\d{4}\b", str(message.content))
            if match:
                return match.group(0)

    return None


def _run_ticket_lookup(ticket_id: str) -> str:
    """Use the same deterministic local lookup used by the Task 2.7 tool."""

    result = check_support_ticket_status(ticket_id)

    if "error" in result:
        return result["error"]

    return (
        f"Ticket {result['record_id']} is {result['status']}. "
        f"Its recorded resolution time is "
        f"{result['resolution_time_hours']} hours and its escalation score is "
        f"{result['escalation_score']}."
    )


def _respond(messages: list[BaseMessage]) -> AIMessage:
    """Create a deterministic response from the current session history."""

    current_question = _last_human_message(messages)
    ticket_id = _ticket_id_from_messages(messages)

    if re.search(r"\bremember\b", current_question, re.IGNORECASE):
        if ticket_id:
            return AIMessage(
                content=(
                    f"Session context saved: I will remember ticket "
                    f"{ticket_id} for this conversation."
                )
            )

        return AIMessage(
            content="I can remember a ticket ID when you provide one."
        )

    if re.search(r"\b(its|that ticket|the ticket)\b", current_question, re.IGNORECASE):
        if not ticket_id:
            return AIMessage(
                content=(
                    "I do not have a ticket ID in this session yet. "
                    "Please provide a ticket ID such as TKT-0002."
                )
            )

        crew_answer = _run_ticket_lookup(ticket_id)
        return AIMessage(
            content=(
                f"Using this session's remembered ticket ID ({ticket_id}):\n"
                f"{crew_answer}"
            )
        )

    return AIMessage(
        content=(
            "Please provide a ticket ID, or ask about the ticket remembered "
            "in this session."
        )
    )


def build_session_memory_runnable() -> RunnableWithMessageHistory:
    """Build the LangChain runnable that adds history by ``session_id``."""

    return RunnableWithMessageHistory(
        RunnableLambda(_respond),
        get_session_history,
    )


def ask(
    runnable: RunnableWithMessageHistory,
    session_id: str,
    question: str,
) -> str:
    """Submit one turn to a named session and return its text response."""

    response = runnable.invoke(
        [HumanMessage(content=question)],
        config={"configurable": {"session_id": session_id}},
    )

    return str(response.content)


def run_demo() -> None:
    """Execute the required same-session and fresh-session demonstration."""

    SESSION_HISTORIES.clear()
    memory_runnable = build_session_memory_runnable()

    session_a = "ola-session-a"
    session_b = "ola-session-b"

    print("=" * 70)
    print("PART 2.8 - LANGCHAIN SESSION MEMORY DEMO")
    print("=" * 70)

    print("\nSESSION A / TURN 1 - establish context")
    answer_a1 = ask(
        memory_runnable,
        session_a,
        "Please remember that my ticket is TKT-0002.",
    )
    print(answer_a1)

    print("\nSESSION A / TURN 2 - use earlier context")
    answer_a2 = ask(
        memory_runnable,
        session_a,
        "What is its status?",
    )
    print(answer_a2)

    print("\nSESSION B / TURN 1 - fresh session")
    answer_b1 = ask(
        memory_runnable,
        session_b,
        "What is its status?",
    )
    print(answer_b1)

    # Assertions make the demo an executable verification of the requirement.
    assert "TKT-0002" in answer_a1
    assert "TKT-0002" in answer_a2
    assert "Escalated" in answer_a2
    assert "do not have a ticket ID" in answer_b1
    assert len(SESSION_HISTORIES[session_a].messages) == 4
    assert len(SESSION_HISTORIES[session_b].messages) == 2

    print("\n" + "=" * 70)
    print("SESSION MEMORY TEST PASSED")
    print("Session A retained context; Session B remained isolated.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
