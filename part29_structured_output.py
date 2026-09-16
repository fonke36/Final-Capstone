"""
Part 2.9 - Structured Output Schema

Defines and enforces a Pydantic response format for a deterministic local
CrewAI support response.  This is a focused Task 9 demonstration; it does not
replace the existing Part 2.7 multi-agent crew or Part 2.8 session memory.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

# Keep this demonstration local and deterministic.
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

# CrewAI records task outputs in a SQLite database.  Direct that transient
# framework data to the system temp folder so this demo does not create
# project files and can run in restricted environments.
import crewai_core.paths

_TASK_OUTPUT_DIRECTORY = Path(tempfile.gettempdir()) / "ola_task9_crewai_output"
_TASK_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
crewai_core.paths.db_storage_path = lambda: str(_TASK_OUTPUT_DIRECTORY)

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from pydantic import BaseModel, Field, ValidationError, model_validator


class OlaSupportResponse(BaseModel):
    """The required structured format for every final Task 9 crew response."""

    response_type: Literal["ticket_lookup", "knowledge_base", "fallback"]
    summary: str = Field(min_length=1)
    ticket_id: str | None = Field(default=None, pattern=r"^TKT-\d{4}$")
    ticket_status: str | None = None
    resolution_time_hours: float | None = Field(default=None, ge=0)
    escalation_score: float | None = Field(default=None, ge=0, le=1)
    sources: list[str] = Field(min_length=1)
    next_action: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ticket_lookup_fields(self) -> "OlaSupportResponse":
        """Require ticket details whenever the response is a ticket lookup."""

        if self.response_type == "ticket_lookup":
            required_values = (
                self.ticket_id,
                self.ticket_status,
                self.resolution_time_hours,
                self.escalation_score,
            )

            if any(value is None for value in required_values):
                raise ValueError(
                    "ticket_lookup responses require ticket ID, status, "
                    "resolution time, and escalation score."
                )

        return self


# The variable name makes the intended CrewAI response format explicit and is
# passed directly to Task.output_pydantic below.
response_format = OlaSupportResponse


class StructuredMockLLM(BaseLLM):
    """Local deterministic mock following the existing Part 2.7 MockLLM pattern."""

    def __init__(self) -> None:
        super().__init__(model="structured-mock-llm", temperature=0)

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        response_model: Any = None,
        **kwargs: Any,
    ) -> str:
        """Return valid JSON for the response schema without any network call."""

        response = {
            "response_type": "ticket_lookup",
            "summary": "Ticket TKT-0002 is currently escalated.",
            "ticket_id": "TKT-0002",
            "ticket_status": "Escalated",
            "resolution_time_hours": 54.5,
            "escalation_score": 0.9867,
            "sources": ["support_ticket_lookup"],
            "next_action": "Continue escalation handling and update the customer.",
        }

        return json.dumps(response)


def build_structured_crew() -> Crew:
    """Build a local CrewAI crew whose final task requires ``response_format``."""

    agent = Agent(
        role="Ola Structured Response Composer",
        goal="Return a valid structured Ola support response.",
        backstory="You produce deterministic local support responses.",
        llm=StructuredMockLLM(),
        verbose=True,
        allow_delegation=False,
    )

    task = Task(
        description=(
            "Return the status for ticket TKT-0002 using the required "
            "structured response format."
        ),
        expected_output="A response conforming to OlaSupportResponse.",
        agent=agent,
        # This connects the Pydantic schema to the actual CrewAI task output.
        output_pydantic=response_format,
    )

    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )


def validate_final_crew_response(crew_output: Any) -> OlaSupportResponse:
    """Validate a final CrewAI response against the required response format."""

    if crew_output.pydantic is None:
        raise ValueError("Crew response did not include Pydantic structured output.")

    # CrewAI validates through output_pydantic; validate again explicitly so
    # every final response is checked in project code as well.
    return response_format.model_validate(crew_output.pydantic.model_dump())


def run_demo() -> None:
    """Run valid and invalid structured-output validation demonstrations."""

    print("=" * 70)
    print("PART 2.9 - STRUCTURED OUTPUT SCHEMA DEMO")
    print("=" * 70)

    print("\nVALID CREW RESPONSE TEST")
    crew = build_structured_crew()
    crew_output = crew.kickoff()
    validated = validate_final_crew_response(crew_output)

    assert isinstance(crew_output.pydantic, response_format)
    assert validated.ticket_id == "TKT-0002"
    assert validated.ticket_status == "Escalated"
    print("Crew task response_format:", response_format.__name__)
    print("Validated response:", validated.model_dump())
    print("VALID RESPONSE PASSED")

    print("\nINTENTIONALLY INVALID RESPONSE TEST")
    invalid_response = {
        "response_type": "ticket_lookup",
        "summary": "Incomplete ticket response.",
        "ticket_id": "INVALID-ID",
        "ticket_status": "Escalated",
        "resolution_time_hours": -1,
        "escalation_score": 1.5,
        "sources": [],
        "next_action": "",
    }

    try:
        response_format.model_validate(invalid_response)
    except ValidationError as error:
        print("INVALID RESPONSE REJECTED")
        print(error)
    else:
        raise AssertionError("Invalid response unexpectedly passed validation.")

    print("\n" + "=" * 70)
    print("STRUCTURED OUTPUT TEST PASSED")
    print("Crew output used response_format and invalid data was rejected.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
