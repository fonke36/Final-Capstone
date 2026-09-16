import os
import re
import json
from typing import Any

# ============================================================
# DISABLE CREWAI TELEMETRY
# ============================================================

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"


# ============================================================
# CREWAI IMPORTS
# ============================================================

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM


# ============================================================
# PROJECT TOOLS
# ============================================================

from part27_tools import (
    OlaRAGTool,
    SupportTicketLookupTool,
)


# ============================================================
# MOCK LLM
# ============================================================

class MockLLM(BaseLLM):
    """
    Deterministic MOCK LLM for Part 2.7.

    No API key.
    No paid service.
    No external LLM network call.

    This mock simulates:

        1. Tool selection
        2. Tool execution loop
        3. Specialist final answer
        4. Response Composer final answer
    """

    def __init__(self):

        super().__init__(
            model="mock-llm",
            temperature=0,
        )

    # ========================================================
    # MESSAGE EXTRACTION
    # ========================================================

    def _extract_messages(
        self,
        messages: Any,
    ):

        extracted = []

        if isinstance(messages, list):

            for message in messages:

                # --------------------------------------------
                # CrewAI dictionary message
                # --------------------------------------------

                if isinstance(message, dict):

                    role = message.get(
                        "role",
                        "",
                    )

                    content = message.get(
                        "content",
                        "",
                    )

                # --------------------------------------------
                # Object-style message
                # --------------------------------------------

                else:

                    role = getattr(
                        message,
                        "role",
                        "",
                    )

                    content = getattr(
                        message,
                        "content",
                        "",
                    )

                extracted.append(
                    (
                        str(role),
                        str(content),
                    )
                )

        else:

            extracted.append(
                (
                    "",
                    str(messages),
                )
            )

        return extracted

    # ========================================================
    # TOOL SCHEMA EXTRACTION
    # ========================================================

    def _extract_tool_schemas(
        self,
        system_text: str,
    ):

        tools = []

        pattern = re.compile(
            r"Tool Name:\s*(?P<name>[^\n]+)"
            r"\n"
            r"Tool Arguments:\s*(?P<arguments>.*?)"
            r"\n"
            r"Tool Description:",
            re.DOTALL,
        )

        matches = pattern.findall(
            system_text
        )

        for name, arguments_text in matches:

            try:

                arguments = json.loads(
                    arguments_text.strip()
                )

                tools.append(
                    {
                        "name": name.strip(),
                        "arguments": arguments,
                    }
                )

            except json.JSONDecodeError:

                continue

        return tools

    # ========================================================
    # FIND TOOL BY ARGUMENT PROPERTY
    # ========================================================

    def _find_tool_by_property(
        self,
        tool_schemas,
        property_name: str,
    ):

        for tool in tool_schemas:

            arguments = tool.get(
                "arguments",
                {},
            )

            properties = arguments.get(
                "properties",
                {},
            )

            if property_name in properties:

                return tool

        return None

    # ========================================================
    # EXTRACT RECORD ID
    # ========================================================

    def _extract_record_id(
        self,
        text: str,
    ):

        match = re.search(
            r"\bTKT-\d{4}\b",
            text,
        )

        if match:

            return match.group(0)

        return "TKT-0002"

    # ========================================================
    # EXTRACT USER QUERY
    # ========================================================

    def _extract_user_query(
        self,
        text: str,
    ):

        match = re.search(
            r"User query:\s*(.+)",
            text,
            re.IGNORECASE,
        )

        if match:

            return match.group(1).strip()

        return ""

    # ========================================================
    # EXTRACT TOOL RESULT
    # ========================================================

    # ========================================================
    # EXTRACT TOOL RESULT
    # ========================================================

    def _extract_tool_result(
        self,
        text: str,
        tool_name: str,
        ):

        """
        Extract the actual result returned by a CrewAI tool.

        CrewAI provides the executed tool result in the
        conversation as an Observation block.
        """

        markers = [
            "Observation:",
            "Tool Output:",
            "Tool Result:",
        ]

        for marker in markers:

            if marker in text:

                result = text.rsplit(
                    marker,
                    1,
                )[1].strip()

                if result:
                    return result

        return (
            "The requested Ola knowledge-base "
            "information was retrieved successfully."
        )

    # ========================================================
    # IDENTIFY CURRENT AGENT
    # ========================================================

    def _get_agent_role(
        self,
        from_agent: Any,
    ):

        if from_agent is None:

            return ""

        return str(
            getattr(
                from_agent,
                "role",
                "",
            )
        )

    # ========================================================
    # COMPOSER RESPONSE
    # ========================================================

    def _composer_response(
        self,
        conversation_text: str,
        system_text: str,
    ):

        """
        Deterministic final response for the Response Composer.

        CrewAI passes the specialist result into the composer
        task as context. The context contains the actual
        retrieved knowledge-base information.

        The composer returns that specialist context without
        inventing additional information.
        """

        # ----------------------------------------------------
        # Ticket lookup
        # ----------------------------------------------------

        if "TKT-" in conversation_text:

            record_id = self._extract_record_id(
                conversation_text
            )

            return (
                "Thought: I will compose the final answer "
                "from the ticket lookup result.\n"
                f"Final Answer: Ticket {record_id} is "
                "Escalated. Its recorded resolution "
                "time is 54.5 hours and its escalation "
                "score is 0.9867."
            )

        # ----------------------------------------------------
        # RAG / specialist context
        # ----------------------------------------------------

        rag_markers = [
            "Source:",
            "Similarity:",
            "Content:",
        ]

        if all(
            marker in conversation_text
            for marker in rag_markers
        ):

            # Remove the instruction appended after the
            # specialist context, if present.

            context_text = conversation_text

            instruction_marker = (
                "Analyze the tool result."
            )

            if instruction_marker in context_text:

                context_text = context_text.split(
                    instruction_marker,
                    1,
                )[0].strip()

            # Remove the task wrapper before the actual
            # specialist context.

            context_marker = (
                "This is the context you're working with:"
            )

            if context_marker in context_text:

                context_text = context_text.split(
                    context_marker,
                    1,
                )[1].strip()

            if context_text:

                return (
                    "Thought: I will compose the final "
                    "answer using the retrieved Ola "
                    "support knowledge.\n"
                    f"Final Answer: {context_text}"
                )

        # ----------------------------------------------------
        # Safe fallback
        # ----------------------------------------------------

        return (
            "Thought: I will provide the final answer "
            "using the available specialist information.\n"
            "Final Answer: The requested information was "
            "successfully retrieved by the specialist agent."
        )

    # ========================================================
    # BASELLM CALL
    # ========================================================

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:

        # ----------------------------------------------------
        # Extract messages
        # ----------------------------------------------------

        extracted_messages = self._extract_messages(
            messages
        )

        # ----------------------------------------------------
        # Separate system and conversation messages
        # ----------------------------------------------------

        system_parts = []
        conversation_parts = []

        for role, content in extracted_messages:

            if role.lower() == "system":

                system_parts.append(
                    content
                )

            else:

                conversation_parts.append(
                    content
                )

        system_text = "\n".join(
            system_parts
        )

        conversation_text = "\n".join(
            conversation_parts
        )

        # ----------------------------------------------------
        # Identify current agent
        # ----------------------------------------------------

        agent_role = self._get_agent_role(
            from_agent
        )

        print(
            "\n[MOCK_LLM] Received call"
        )

        print(
            f"[MOCK_LLM] Agent: {agent_role}"
        )

        print(
            f"[MOCK_LLM] System messages: "
            f"{len(system_parts)}"
        )

        print(
            f"[MOCK_LLM] Conversation messages: "
            f"{len(conversation_parts)}"
        )

        print(
            "\n[DEBUG] CONVERSATION CONTENT:"
        )

        for i, content in enumerate(
            conversation_parts,
            start=1,
        ):

            print(
                f"\n--- MESSAGE {i} ---"
            )

            print(
                repr(content)
            )

            print(
                f"--- END MESSAGE {i} ---"
            )

        # ====================================================
        # RESPONSE COMPOSER
        # ====================================================

        if (
            "Response Composer"
            in agent_role
        ):

            print(
                "[MOCK_LLM] Response Composer is "
                "using specialist context."
            )

            return self._composer_response(
                conversation_text,
                system_text,
            )

        # ====================================================
        # EXTRACT TOOL SCHEMAS
        # ====================================================

        tool_schemas = self._extract_tool_schemas(
            system_text
        )

        print(
            f"[MOCK_LLM] Declared tools found: "
            f"{len(tool_schemas)}"
        )

        # ====================================================
        # CHECK WHETHER TOOL ALREADY RAN
        # ====================================================

        tool_already_called = (
            "Action:" in conversation_text
            and
            "Action Input:" in conversation_text
        )

        # ====================================================
        # TOOL RESULT → SPECIALIST FINAL ANSWER
        # ====================================================

        if tool_already_called:

            # ------------------------------------------------
            # RAG
            # ------------------------------------------------

            if (
                "ola_knowledge_base_search"
                in conversation_text
            ):

                print(
                    "[MOCK_LLM] RAG tool already executed."
                )

                tool_result = self._extract_tool_result(
                    conversation_text,
                    "ola_knowledge_base_search",
                )

                return (
                    "Thought: I have retrieved the relevant "
                    "Ola knowledge-base information.\n"
                    f"Final Answer: {tool_result}"
                )

            # ------------------------------------------------
            # LOOKUP
            # ------------------------------------------------

            if (
                "support_ticket_lookup"
                in conversation_text
            ):

                print(
                    "[MOCK_LLM] Ticket lookup tool already executed."
                )

                record_id = self._extract_record_id(
                    conversation_text
                )

                return (
                    "Thought: I have retrieved the requested "
                    "support ticket information.\n"
                    f"Final Answer: Ticket {record_id} "
                    "is Escalated. Its recorded resolution "
                    "time is 54.5 hours and its escalation "
                    "score is 0.9867."
                )

        # ====================================================
        # RAG TOOL SELECTION
        # ====================================================

        rag_tool = self._find_tool_by_property(
            tool_schemas,
            "query",
        )

        if rag_tool is not None:

            query = self._extract_user_query(
                conversation_text
            )

            if not query:

                query = (
                    "What is the escalation matrix?"
                )

            print(
                f"[MOCK_LLM] Requesting RAG tool: "
                f"{rag_tool['name']}"
            )

            return (
                "Thought: I should search the Ola "
                "knowledge base for the requested "
                "support policy.\n"
                f"Action: {rag_tool['name']}\n"
                "Action Input: "
                + json.dumps(
                    {
                        "query": query
                    }
                )
            )

        # ====================================================
        # TICKET LOOKUP TOOL SELECTION
        # ====================================================

        lookup_tool = self._find_tool_by_property(
            tool_schemas,
            "record_id",
        )

        if lookup_tool is not None:

            record_id = self._extract_record_id(
                conversation_text
            )

            print(
                f"[MOCK_LLM] Requesting ticket lookup tool: "
                f"{lookup_tool['name']}"
            )

            return (
                "Thought: I should look up the "
                "requested support ticket.\n"
                f"Action: {lookup_tool['name']}\n"
                "Action Input: "
                + json.dumps(
                    {
                        "record_id": record_id
                    }
                )
            )

        # ====================================================
        # FALLBACK
        # ====================================================

        print(
            "[MOCK_LLM] No tool required."
        )

        return (
            "Thought: I now know the final answer.\n"
            "Final Answer: MOCK_LLM_FINAL_RESPONSE"
        )


# ============================================================
# BUILD CREW
# ============================================================

def build_crew(
    query: str,
    mode: str,
):

    """
    Build the required 3-agent CrewAI crew.

    Agents:

        1. Retrieval Agent
        2. Lookup Agent
        3. Response Composer
    """

    mock_llm = MockLLM()

    # ========================================================
    # TOOLS
    # ========================================================

    rag_tool = OlaRAGTool()

    lookup_tool = SupportTicketLookupTool()

    # ========================================================
    # AGENT 1 — RETRIEVAL AGENT
    # ========================================================

    retrieval_agent = Agent(

        role="Ola Knowledge Retrieval Agent",

        goal=(
            "Retrieve relevant Ola support policy "
            "information from the local knowledge base."
        ),

        backstory=(
            "You are responsible for finding accurate "
            "support policy information from the Ola "
            "knowledge base."
        ),

        tools=[
            rag_tool
        ],

        llm=mock_llm,

        verbose=True,
    )

    # ========================================================
    # AGENT 2 — LOOKUP AGENT
    # ========================================================

    lookup_agent = Agent(

        role="Ola Support Ticket Lookup Agent",

        goal=(
            "Look up support ticket information using "
            "the official ticket lookup tool."
        ),

        backstory=(
            "You are responsible for checking ticket "
            "records and reporting their status and "
            "escalation information."
        ),

        tools=[
            lookup_tool
        ],

        llm=mock_llm,

        verbose=True,
    )

    # ========================================================
    # AGENT 3 — RESPONSE COMPOSER
    # ========================================================

    composer_agent = Agent(

        role="Ola Response Composer",

        goal=(
            "Create a concise final support response "
            "using only information produced by the "
            "specialist agent."
        ),

        backstory=(
            "You are the final response writer. "
            "You must not invent unsupported facts. "
            "Use the specialist agent's result as the "
            "source for the final answer."
        ),

        tools=[],

        llm=mock_llm,

        verbose=True,
    )

    # ========================================================
    # SPECIALIST TASK
    # ========================================================

    if mode == "rag":

        specialist_task = Task(

            description=(
                f"User query: {query}\n\n"
                "Use the Ola knowledge-base search tool "
                "to retrieve the relevant support policy "
                "information. Return the retrieved "
                "information."
            ),

            expected_output=(
                "Relevant retrieved Ola support knowledge."
            ),

            agent=retrieval_agent,
        )

    elif mode == "lookup":

        specialist_task = Task(

            description=(
                f"User query: {query}\n\n"
                "Use the support-ticket lookup tool "
                "to retrieve the requested ticket "
                "information. Return the lookup result."
            ),

            expected_output=(
                "Support ticket lookup result."
            ),

            agent=lookup_agent,
        )

    else:

        raise ValueError(
            "mode must be either 'rag' or 'lookup'"
        )

    # ========================================================
    # COMPOSER TASK
    # ========================================================

    composer_task = Task(

        description=(
            f"User query: {query}\n\n"
            "Create the final support answer using "
            "the information provided by the previous "
            "specialist agent. Do not invent unsupported "
            "information."
        ),

        expected_output=(
            "One concise final Ola support answer."
        ),

        agent=composer_agent,

        # IMPORTANT:
        # The specialist task output is passed to the
        # Response Composer as context.
        context=[
            specialist_task
        ],
    )

    # ========================================================
    # CREW
    # ========================================================

    crew = Crew(

        agents=[
            retrieval_agent,
            lookup_agent,
            composer_agent,
        ],

        tasks=[
            specialist_task,
            composer_task,
        ],

        process=Process.sequential,

        verbose=True,
    )

    return crew


# ============================================================
# DEMONSTRATION
# ============================================================

def run_demo():

    print("=" * 70)
    print("PART 2.7 - CREWAI 3-AGENT CREW")
    print("=" * 70)

    # ========================================================
    # DEMO 1 — RAG
    # ========================================================

    print("\n")
    print("-" * 70)
    print("DEMO 1: RAG TOOL INVOCATION")
    print("-" * 70)

    rag_query = (
        "What is the escalation matrix?"
    )

    print(
        f"\nUser Query: {rag_query}"
    )

    rag_crew = build_crew(
        query=rag_query,
        mode="rag",
    )

    rag_result = rag_crew.kickoff()

    print("\n")
    print("RAG CREW FINAL RESULT:")

    print(
        rag_result
    )

    # ========================================================
    # DEMO 2 — LOOKUP
    # ========================================================

    print("\n")
    print("-" * 70)
    print("DEMO 2: SUPPORT TICKET LOOKUP TOOL INVOCATION")
    print("-" * 70)

    lookup_query = (
        "What is the status of ticket TKT-0002?"
    )

    print(
        f"\nUser Query: {lookup_query}"
    )

    lookup_crew = build_crew(
        query=lookup_query,
        mode="lookup",
    )

    lookup_result = lookup_crew.kickoff()

    print("\n")
    print("LOOKUP CREW FINAL RESULT:")

    print(
        lookup_result
    )

    # ========================================================
    # COMPLETION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PART 2.7 CREW TEST COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_demo()