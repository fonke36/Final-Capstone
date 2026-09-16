"""
Part 2.7 - CrewAI Tools

Two CrewAI tools:
1. Ola Knowledge Base Retrieval Tool
2. Support Ticket Lookup Tool
"""

from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from rag_generation import (
    retrieve_chunks,
    load_rag_system,
)

from ticket_lookup import check_support_ticket_status


# =========================================================
# Load the existing RAG system
# =========================================================

RAG_MODEL, FIXED_COLLECTION, SENTENCE_COLLECTION = load_rag_system()


# =========================================================
# 1. RAG RETRIEVAL TOOL
# =========================================================

class RAGSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="The Ola support question to search for."
    )


class OlaRAGTool(BaseTool):
    name: str = "ola_knowledge_base_search"

    description: str = (
        "Search the local Ola support knowledge base and return "
        "the most relevant support information for a customer question."
    )

    args_schema: Type[BaseModel] = RAGSearchInput

    def _run(self, query: str) -> str:

        # Use the sentence-based collection recommended in Part 1.5
        results = retrieve_chunks(
            query,
            RAG_MODEL,
            SENTENCE_COLLECTION,
            top_k=3
        )

        if not results:
            return (
                "No relevant information was found in the "
                "Ola support knowledge base."
            )

        output = []

        for result in results:

            output.append(
                f"Source: {result['doc_id']}\n"
                f"Similarity: {result['similarity']:.4f}\n"
                f"Content: {result['text']}"
            )

        return "\n\n".join(output)


# =========================================================
# 2. TICKET LOOKUP TOOL
# =========================================================

class TicketLookupInput(BaseModel):
    record_id: str = Field(
        ...,
        description="The support ticket ID, for example TKT-0001."
    )


class SupportTicketLookupTool(BaseTool):
    name: str = "support_ticket_lookup"

    description: str = (
        "Look up an Ola support ticket using its record ID. "
        "Returns the ticket status, resolution time, and escalation score."
    )

    args_schema: Type[BaseModel] = TicketLookupInput

    def _run(self, record_id: str) -> str:

        result = check_support_ticket_status(record_id)

        return str(result)


# =========================================================
# 3. DIRECT TOOL TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PART 2.7 - ACTUAL CREWAI TOOLS TEST")
    print("=" * 60)

    # Create the two CrewAI tools
    rag_tool = OlaRAGTool()
    lookup_tool = SupportTicketLookupTool()

    # -----------------------------------------------------
    # Test RAG tool
    # -----------------------------------------------------

    print("\n" + "-" * 60)
    print("RAG TOOL TEST")
    print("-" * 60)

    rag_result = rag_tool.run(
        query="What is the SLA policy for different severity levels?"
    )

    print(rag_result)

    # -----------------------------------------------------
    # Test Lookup tool
    # -----------------------------------------------------

    print("\n" + "-" * 60)
    print("TICKET LOOKUP TOOL TEST")
    print("-" * 60)

    lookup_result = lookup_tool.run(
        record_id="TKT-0002"
    )

    print(lookup_result)

    # -----------------------------------------------------
    # Finished
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("Both CrewAI tools executed successfully.")
    print("=" * 60)
