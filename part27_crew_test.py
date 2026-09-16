"""
Part 2.7 - CrewAI Orchestration Test

Small test to verify:
- Agent
- Task
- Crew
- kickoff()
- MOCK_LLM
"""

import os

# Disable telemetry for deterministic local testing.
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Task, Crew, Process
from crewai.llms.base_llm import BaseLLM


# =========================================================
# 1. MOCK LLM
# =========================================================

class MockLLM(BaseLLM):
    """
    Simple deterministic local LLM for testing CrewAI.
    """

    def __init__(self):
        super().__init__(
            model="mock-llm",
            temperature=0
        )

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=False,
        **kwargs
    ):
        return "MOCK_LLM_RESPONSE"


# =========================================================
# 2. CREATE MOCK LLM
# =========================================================

mock_llm = MockLLM()


# =========================================================
# 3. CREATE AGENT
# =========================================================

test_agent = Agent(
    role="Test Agent",
    goal="Complete a simple test task.",
    backstory=(
        "You are a test agent used to verify that the "
        "CrewAI orchestration system is working."
    ),
    llm=mock_llm,
    verbose=True,
    allow_delegation=False,
)


# =========================================================
# 4. CREATE TASK
# =========================================================

test_task = Task(
    description=(
        "Return a simple confirmation that the CrewAI "
        "test task was executed."
    ),
    expected_output="A confirmation response.",
    agent=test_agent,
)


# =========================================================
# 5. CREATE CREW
# =========================================================

crew = Crew(
    agents=[test_agent],
    tasks=[test_task],
    process=Process.sequential,
    verbose=True,
)


# =========================================================
# 6. RUN CREW
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PART 2.7 - CREWAI ORCHESTRATION TEST")
    print("=" * 60)

    result = crew.kickoff()

    print("\n" + "-" * 60)
    print("KICKOFF RESULT")
    print("-" * 60)

    print(result)

    print("\n" + "=" * 60)
    print("CrewAI kickoff() test completed.")
    print("=" * 60)