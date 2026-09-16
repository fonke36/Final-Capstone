from statistics import mean

from rag_generation import (
    grounded_answer,
    load_rag_system,
    retrieve_chunks,
    run_calibration,
)
from rag_evaluation import score_query
from part27_crew import build_crew
from part213_llm_judge import Task13MockJudge


# ---------------------------------------------------------------------------
# Task 13: 15-query evaluation set
# ---------------------------------------------------------------------------
#
# Each query contains:
#   - query: the user question
#   - relevant_docs: KB documents expected to answer the question
#
# The four per-query metrics are:
#   1. precision
#   2. recall
#   3. top_1_similarity
#   4. grounded
#
# ---------------------------------------------------------------------------

EVALUATION_QUERIES = [
    {
        "query": "How are support tickets prioritized?",
        "relevant_docs": {"ticket_priority"},
    },
    {
        "query": "What SLA applies to a high severity support issue?",
        "relevant_docs": {"sla_severity"},
    },
    {
        "query": "When should a support ticket be escalated?",
        "relevant_docs": {"escalation_matrix"},
    },
    {
        "query": "What refund or compensation can a customer receive?",
        "relevant_docs": {"refund_compensation"},
    },
    {
        "query": "What communication channels can customers use?",
        "relevant_docs": {"communication_channels"},
    },
    {
        "query": "What are the business hours for Ola support?",
        "relevant_docs": {"business_hours"},
    },
    {
        "query": "How should repeat complaints be handled?",
        "relevant_docs": {"repeat_complaints"},
    },
    {
        "query": "When can a customer receive a service credit?",
        "relevant_docs": {"service_credit"},
    },
    {
        "query": "How is customer feedback collected?",
        "relevant_docs": {"feedback_collection"},
    },
    {
        "query": "How are VIP customers handled?",
        "relevant_docs": {"vip_handling"},
    },
    {
        "query": "How should customers be informed during an outage?",
        "relevant_docs": {"outage_communication"},
    },
    {
        "query": "How long is support ticket data retained?",
        "relevant_docs": {"ticket_data_retention"},
    },
    {
        "query": "What is the escalation process for a serious customer issue?",
        "relevant_docs": {"escalation_matrix"},
    },
    {
        "query": "What is the weather forecast for Agra tomorrow?",
        "relevant_docs": set(),
        "out_of_scope": True,
    },
    {
        "query": "Can Ola provide a home loan to customers?",
        "relevant_docs": set(),
        "out_of_scope": True,
    },
]


def evaluate_query(
    query_record,
    model,
    collection,
    threshold,
    judge,
):
    """Evaluate one query using RAG retrieval, Part 2 Crew, and Task 13 judge."""

    query = query_record["query"]
    relevant_docs = set(query_record["relevant_docs"])
    out_of_scope = query_record.get("out_of_scope", False)

    retrieved_results = retrieve_chunks(
        query,
        model,
        collection,
        top_k=3,
    )

    retrieval_scores = score_query(
        retrieved_results,
        relevant_docs,
    )

    if retrieved_results:
        top_1_similarity = retrieved_results[0]["similarity"]
    else:
        top_1_similarity = 0.0

    grounded_result = grounded_answer(
        query,
        model,
        collection,
        threshold,
        top_k=3,
    )

    # ---------------------------------------------------------
    # Run the existing Part 2 Crew under MOCK_LLM.
    # ---------------------------------------------------------
    crew = build_crew(
        query=query,
        mode="rag",
    )

    crew_result = crew.kickoff()
    crew_answer = str(crew_result)

    # ---------------------------------------------------------
    # Build reference context from retrieved KB material.
    # ---------------------------------------------------------
    context_parts = []
    for item in retrieved_results:
        context_parts.append(str(item.get("text", "")))
    context = "\n".join(context_parts)

    # ---------------------------------------------------------
    # Task 13 LLM-as-judge.
    # ---------------------------------------------------------
    judge_result = judge.judge(
        query=query,
        answer=crew_answer,
        context=context,
        expected_topics=sorted(relevant_docs),
        out_of_scope=out_of_scope,
    )

    return {
        "query": query,
        "precision": retrieval_scores["precision"],
        "recall": retrieval_scores["recall"],
        "top_1_similarity": top_1_similarity,
        "grounded": grounded_result["grounded"],
        "retrieved_docs": sorted(
            retrieval_scores["retrieved_docs"]
        ),
        "relevant_docs": sorted(relevant_docs),
        "sources": grounded_result["sources"],
        "crew_answer": crew_answer,
        "judge": judge_result,
    }


def evaluate_all_queries(
    collection_name,
    collection,
    model,
    threshold,
    judge,
    ):

    """Run the 15-query evaluation."""

    print("=" * 80)
    print(f"TASK 13 - {collection_name} EVALUATION")
    print("=" * 80)

    results = []

    for index, query_record in enumerate(EVALUATION_QUERIES, start=1):
        result = evaluate_query(
            query_record,
            model,
            collection,
            threshold,
            judge,
            )

        results.append(result)

        print(f"\nQuery {index:02d}: {result['query']}")
        print(f"  Precision:        {result['precision']:.4f}")
        print(f"  Recall:           {result['recall']:.4f}")
        print(f"  Top-1 similarity: {result['top_1_similarity']:.4f}")
        print(f"  Grounded:         {result['grounded']}")
        print(f"  Accuracy:         {result['judge']['accuracy']:.1f}")
        print(f"  Grounding:        {result['judge']['grounding']:.1f}")
        print(f"  Completeness:     {result['judge']['completeness']:.1f}")
        print(f"  Safety:           {result['judge']['safety']:.1f}")


    average_precision = mean(
        result["precision"] for result in results
    )

    average_recall = mean(
        result["recall"] for result in results
    )

    average_similarity = mean(
        result["top_1_similarity"] for result in results
    )

    grounded_rate = mean(
        1.0 if result["grounded"] else 0.0
        for result in results
    )

    average_accuracy = mean(
        result["judge"]["accuracy"]
        for result in results
    )

    average_grounding = mean(
        result["judge"]["grounding"]
        for result in results
    )

    average_completeness = mean(
        result["judge"]["completeness"]
        for result in results
    )

    average_safety = mean(
        result["judge"]["safety"]
        for result in results
    )

    print(f"Grounded rate:      {grounded_rate:.4f}")
    print("\n" + "=" * 80)
    print(f"{collection_name} SUMMARY")
    print("=" * 80)
    print(f"Queries evaluated:  {len(results)}")
    print(f"Average precision:  {average_precision:.4f}")
    print(f"Average recall:     {average_recall:.4f}")
    print(f"Average similarity: {average_similarity:.4f}")
    print(f"Grounded rate:      {grounded_rate:.4f}")
    print(f"Average accuracy:   {average_accuracy:.4f}")
    print(f"Average grounding:  {average_grounding:.4f}")
    print(f"Average completeness:{average_completeness:.4f}")
    print(f"Average safety:     {average_safety:.4f}")


    return {
        "strategy": collection_name,
        "results": results,
        "average_precision": average_precision,
        "average_recall": average_recall,
        "average_similarity": average_similarity,
        "grounded_rate": grounded_rate,
        "average_accuracy": average_accuracy,
        "average_grounding": average_grounding,
        "average_completeness": average_completeness,
        "average_safety": average_safety,
    }


def main():
    """Run Task 13 evaluation for both RAG strategies."""

    model, fixed_collection, sentence_collection = load_rag_system()
    judge = Task13MockJudge()

    _, _, threshold = run_calibration(
        model,
        fixed_collection,
        sentence_collection,
    )

    if threshold is None:
        raise RuntimeError(
            "Calibration did not produce a usable similarity threshold."
        )

    print(f"\nCalibrated similarity threshold: {threshold:.4f}")

    fixed_results = evaluate_all_queries(
        "Fixed-size",
        fixed_collection,
        model,
        threshold,
        judge,
    )

    sentence_results = evaluate_all_queries(
        "Sentence-based",
        sentence_collection,
        model,
        threshold,
        judge,
    )

    print("\n" + "=" * 80)
    print("TASK 13 FINAL COMPARISON")
    print("=" * 80)

    print(
        f"Fixed-size:   "
        f"precision={fixed_results['average_precision']:.4f}, "
        f"recall={fixed_results['average_recall']:.4f}, "
        f"similarity={fixed_results['average_similarity']:.4f}, "
        f"grounded_rate={fixed_results['grounded_rate']:.4f}"
    )

    print(
        f"Sentence-based: "
        f"precision={sentence_results['average_precision']:.4f}, "
        f"recall={sentence_results['average_recall']:.4f}, "
        f"similarity={sentence_results['average_similarity']:.4f}, "
        f"grounded_rate={sentence_results['grounded_rate']:.4f}"
    )

    print("\nTASK 13 EVALUATION COMPLETED")


if __name__ == "__main__":
    main()