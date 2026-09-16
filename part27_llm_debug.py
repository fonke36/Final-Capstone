import os

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from typing import Any
from crewai.llms.base_llm import BaseLLM
from crewai import Agent, Task, Crew, Process

from part27_tools import OlaRAGTool


class DebugLLM(BaseLLM):

    def __init__(self):
        super().__init__(
            model="debug-llm",
            temperature=0,
        )

    def call(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict] | None = None,
        callbacks: Any = None,
        available_functions: Any = None,
        **kwargs: Any,
    ) -> str:

        print("\n" + "=" * 70)
        print("MOCK LLM DEBUG")
        print("=" * 70)

        print("\nMESSAGE TYPE:")
        print(type(messages))

        print("\nMESSAGES:")
        print(messages)

        print("\nTOOLS:")
        print(tools)

        print("\nAVAILABLE FUNCTIONS:")
        print(available_functions)

        print("\nKWARGS:")
        print(kwargs)

        print("\n" + "=" * 70)

        return "DEBUG_RESPONSE"


rag_tool = OlaRAGTool()

agent = Agent(
    role="Retrieval Agent",
    goal="Retrieve information from the Ola knowledge base.",
    backstory="You retrieve support policy information.",
    tools=[rag_tool],
    llm=DebugLLM(),
    verbose=True,
)

task = Task(
    description=(
        "Answer this query using the knowledge base tool: "
        "What is the escalation matrix?"
    ),
    expected_output="Retrieved escalation policy information.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff()

print("\nFINAL RESULT:")
print(result)
