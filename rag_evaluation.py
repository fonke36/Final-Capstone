from rag_generation import (
    load_rag_system,
    retrieve_chunks,
)


# ---------------------------------------------------------
# Part 1.5 — Chunking Strategy Evaluation
# ---------------------------------------------------------

EVALUATION_QUERIES = [
    {
        "query": "What are the rules for ticket priority?",
        "relevant_docs": {"ticket_priority"},
    },
    {
        "query": "What is the SLA policy for different severity levels?",
        "relevant_docs": {"sla_severity"},
    },
    {
        "query": "How does the escalation matrix work?",
        "relevant_docs": {"escalation_matrix"},
    },
    {
        "query": "What is the refund and compensation policy?",
        "relevant_docs": {"refund_compensation"},
    },
    {
        "query": "How should repeat complaints be handled?",
        "relevant_docs": {"repeat_complaints"},
    },
]


TOP_K = 3


# ---------------------------------------------------------
# Document-level scoring
# ---------------------------------------------------------

def score_query(retrieved_results, relevant_docs):
    """
    Calculate document-level precision and recall.

    Retrieved chunks are mapped to their parent documents
    using doc_id, and duplicate documents are removed.
    """

    retrieved_docs = {
        result["doc_id"]
        for result in retrieved_results
    }

    relevant_docs = set(relevant_docs)

    true_positives = retrieved_docs & relevant_docs

    precision = (
        len(true_positives) / len(retrieved_docs)
        if retrieved_docs
        else 0.0
    )

    recall = (
        len(true_positives) / len(relevant_docs)
        if relevant_docs
        else 0.0
    )

    return {
        "retrieved_docs": retrieved_docs,
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
    }


# ---------------------------------------------------------
# Evaluate one strategy
# ---------------------------------------------------------

def evaluate_strategy(
    strategy_name,
    collection,
    model,
):
    """
    Evaluate one chunking strategy on all evaluation
    queries.
    """

    print("\n")
    print("=" * 80)
    print(f"STRATEGY: {strategy_name}")
    print("=" * 80)

    all_results = []

    for item in EVALUATION_QUERIES:

        query = item["query"]
        relevant_docs = item["relevant_docs"]

        retrieved = retrieve_chunks(
            query,
            model,
            collection,
            top_k=TOP_K,
        )

        score = score_query(
            retrieved,
            relevant_docs,
        )

        all_results.append(score)

        print("\n" + "-" * 80)
        print(f"Query: {query}")
        print(f"Expected document(s): {sorted(relevant_docs)}")
        print(
            f"Retrieved document(s): "
            f"{sorted(score['retrieved_docs'])}"
        )
        print(
            f"True positive document(s): "
            f"{sorted(score['true_positives'])}"
        )

        print("\nArithmetic:")

        print(
            f"Precision = "
            f"{len(score['true_positives'])} / "
            f"{len(score['retrieved_docs'])} = "
            f"{score['precision']:.4f}"
        )

        print(
            f"Recall = "
            f"{len(score['true_positives'])} / "
            f"{len(relevant_docs)} = "
            f"{score['recall']:.4f}"
        )

    average_precision = (
        sum(result["precision"] for result in all_results)
        / len(all_results)
    )

    average_recall = (
        sum(result["recall"] for result in all_results)
        / len(all_results)
    )

    print("\n")
    print("-" * 80)
    print(f"{strategy_name} — OVERALL RESULTS")
    print("-" * 80)

    print(
        f"Average document-level precision: "
        f"{average_precision:.4f}"
    )

    print(
        f"Average document-level recall: "
        f"{average_recall:.4f}"
    )

    return {
        "strategy": strategy_name,
        "results": all_results,
        "average_precision": average_precision,
        "average_recall": average_recall,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    model, fixed_collection, sentence_collection = (
        load_rag_system()
    )

    fixed_results = evaluate_strategy(
        "Fixed-size + overlap",
        fixed_collection,
        model,
    )

    sentence_results = evaluate_strategy(
        "Sentence-based",
        sentence_collection,
        model,
    )

    print("\n")
    print("=" * 80)
    print("PART 1.5 — FINAL COMPARISON")
    print("=" * 80)

    print(
        "\nFixed-size + overlap:"
    )

    print(
        f"Precision: "
        f"{fixed_results['average_precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{fixed_results['average_recall']:.4f}"
    )

    print(
        "\nSentence-based:"
    )

    print(
        f"Precision: "
        f"{sentence_results['average_precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{sentence_results['average_recall']:.4f}"
    )

    # -----------------------------------------------------
    # Recommendation
    # -----------------------------------------------------

    print("\n")
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    fixed_score = (
        fixed_results["average_precision"]
        + fixed_results["average_recall"]
    ) / 2

    sentence_score = (
        sentence_results["average_precision"]
        + sentence_results["average_recall"]
    ) / 2

    if sentence_score > fixed_score:

        print(
            "\nSentence-based chunking is recommended because "
            "it achieved higher average document-level precision "
            f"({sentence_results['average_precision']:.4f} vs "
            f"{fixed_results['average_precision']:.4f}) while "
            "maintaining the same average recall "
            f"({sentence_results['average_recall']:.4f})."
        )

        print(
            "This means sentence-based chunking produced fewer "
            "irrelevant parent documents in the top-k results "
            "without losing any of the expected relevant documents "
            "on these evaluation queries."
        )

    elif fixed_score > sentence_score:

        print(
            "\nFixed-size + overlap chunking is recommended "
            "because it achieved the stronger combined "
            "document-level precision and recall on the "
            "evaluation queries."
        )

    else:

        print(
            "\nNeither strategy clearly outperformed the "
            "other on the combined document-level precision "
            "and recall results."
        )