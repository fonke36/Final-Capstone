from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHROMA_PATH = Path("chroma_db")

FIXED_COLLECTION_NAME = "ola_support_fixed_chunks"
SENTENCE_COLLECTION_NAME = "ola_support_sentence_chunks"


# ---------------------------------------------------------
# Load embedding model and ChromaDB
# ---------------------------------------------------------

def load_rag_system():
    """
    Load the local embedding model and both existing
    ChromaDB collections created in Part 1.3.
    """

    print("Loading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("Loading ChromaDB...")

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    fixed_collection = chroma_client.get_collection(
        name=FIXED_COLLECTION_NAME
    )

    sentence_collection = chroma_client.get_collection(
        name=SENTENCE_COLLECTION_NAME
    )

    return (
        model,
        fixed_collection,
        sentence_collection,
    )


# ---------------------------------------------------------
# Retrieve chunks
# ---------------------------------------------------------

def retrieve_chunks(
    query,
    model,
    collection,
    top_k=3
):
    """
    Retrieve the top-k chunks for a query.

    Because the embeddings were normalized during indexing,
    the returned distance can be converted to cosine
    similarity using:

        cosine_similarity = 1 - cosine_distance
    """

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        cosine_similarity = 1 - (distance / 2)

        retrieved.append({
            "text": document,
            "doc_id": metadata["doc_id"],
            "distance": distance,
            "similarity": cosine_similarity,
        })

    return retrieved


# ---------------------------------------------------------
# Display retrieval results
# ---------------------------------------------------------

def display_results(
    query,
    results,
    strategy
):
    """
    Display retrieved chunks and similarity scores.
    """

    print("\n" + "-" * 70)
    print(f"Query: {query}")
    print(f"Strategy: {strategy}")
    print("-" * 70)

    for index, result in enumerate(results, start=1):

        print(
            f"\nResult #{index}"
        )

        print(
            f"Document: {result['doc_id']}"
        )

        print(
            f"Cosine similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"Text: {result['text']}"
        )


# ---------------------------------------------------------
# Calibration queries
# ---------------------------------------------------------

IN_SCOPE_QUERIES = [
    "What are the rules for ticket priority?",
    "What is the SLA policy for different severity levels?",
    "How does the escalation matrix work?",
    "What is the refund and compensation policy?",
    "How should repeat complaints be handled?",
]


OUT_OF_SCOPE_QUERIES = [
    "What is the capital of France?",
    "How do I cook biryani?",
]


# ---------------------------------------------------------
# Run similarity calibration
# ---------------------------------------------------------

def run_calibration(
    model,
    fixed_collection,
    sentence_collection
):
    """
    Measure top-1 similarity for in-scope and
    out-of-scope queries.

    These measurements are used to select an empirical
    groundedness threshold rather than using an arbitrary
    tutorial value.
    """

    print("\n")
    print("=" * 70)
    print("PART 1.4 — SIMILARITY CALIBRATION")
    print("=" * 70)

    print("\nIN-SCOPE QUERIES")
    print("=" * 70)

    in_scope_scores = []

    for query in IN_SCOPE_QUERIES:

        results = retrieve_chunks(
            query,
            model,
            sentence_collection,
            top_k=3,
        )

        top1_score = results[0]["similarity"]

        in_scope_scores.append(top1_score)

        print(
            f"\nQuery: {query}"
        )

        print(
            f"Top-1 cosine similarity: "
            f"{top1_score:.4f}"
        )

    print("\nOUT-OF-SCOPE QUERIES")
    print("=" * 70)

    out_scope_scores = []

    for query in OUT_OF_SCOPE_QUERIES:

        results = retrieve_chunks(
            query,
            model,
            sentence_collection,
            top_k=3,
        )

        top1_score = results[0]["similarity"]

        out_scope_scores.append(top1_score)

        print(
            f"\nQuery: {query}"
        )

        print(
            f"Top-1 cosine similarity: "
            f"{top1_score:.4f}"
        )

    print("\n")
    print("=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)

    print(
        "\nIn-scope scores:",
        [
            round(score, 4)
            for score in in_scope_scores
        ]
    )

    print(
        "Out-of-scope scores:",
        [
            round(score, 4)
            for score in out_scope_scores
        ]
    )

    print(
        f"\nLowest in-scope score: "
        f"{min(in_scope_scores):.4f}"
    )

    print(
        f"Highest out-of-scope score: "
        f"{max(out_scope_scores):.4f}"
    )

    # We only select a threshold automatically when
    # the two observed groups have a clear gap.
    lowest_in_scope = min(in_scope_scores)
    highest_out_scope = max(out_scope_scores)

    if highest_out_scope < lowest_in_scope:

        threshold = (
            highest_out_scope +
            lowest_in_scope
        ) / 2

        print(
            f"\nEmpirical threshold selected: "
            f"{threshold:.4f}"
        )

        print(
            "\nReason:"
            "\nThe threshold is the midpoint between "
            "the highest observed out-of-scope score "
            "and the lowest observed in-scope score."
        )

    else:

        threshold = None

        print(
            "\nWARNING:"
            "\nThe observed score groups overlap."
        )

        print(
            "Do NOT choose an arbitrary threshold."
        )

    return (
        in_scope_scores,
        out_scope_scores,
        threshold,
    )


# ---------------------------------------------------------
# Grounded generation
# ---------------------------------------------------------

def grounded_answer(
    query,
    model,
    collection,
    threshold,
    top_k=3
):
    """
    Generate a grounded answer using only retrieved
    knowledge-base context.

    If the top-1 similarity is below the empirically
    calibrated threshold, return the required fallback.
    """

    results = retrieve_chunks(
        query,
        model,
        collection,
        top_k=top_k,
    )

    top1_similarity = results[0]["similarity"]

    if threshold is None:

        raise RuntimeError(
            "No valid groundedness threshold has been "
            "calibrated."
        )

    if top1_similarity < threshold:

        return {
            "answer": (
                "I don't know based on the available "
                "Ola support knowledge base."
            ),
            "grounded": False,
            "similarity": top1_similarity,
            "sources": [],
        }

    # -----------------------------------------------------
    # MOCK_LLM grounded generation
    # -----------------------------------------------------
    #
    # The project must work without an API key.
    # Therefore, MOCK_LLM returns an answer constructed
    # only from the retrieved context.
    #
    # We do not introduce an external LLM here.
    #

    context_parts = []

    for result in results:

        context_parts.append(
            f"[{result['doc_id']}] "
            f"{result['text']}"
        )

    context = "\n\n".join(context_parts)

    answer = (
        "Based on the Ola support knowledge base:\n\n"
        + context
    )

    return {
        "answer": answer,
        "grounded": True,
        "similarity": top1_similarity,
        "sources": [
            result["doc_id"]
            for result in results
        ],
    }


# ---------------------------------------------------------
# Main test
# ---------------------------------------------------------

if __name__ == "__main__":

    (
        model,
        fixed_collection,
        sentence_collection,
    ) = load_rag_system()

    (
        in_scope_scores,
        out_scope_scores,
        threshold,
    ) = run_calibration(
        model,
        fixed_collection,
        sentence_collection,
    )

    if threshold is None:

        print(
            "\nPart 1.4 cannot continue until "
            "the calibration produces separable "
            "score clusters."
        )

        raise SystemExit(1)

    print("\n")
    print("=" * 70)
    print("PART 1.4 — GROUNDED GENERATION TEST")
    print("=" * 70)

    test_queries = [
        "What are the rules for ticket priority?",
        "What is the SLA policy for different severity levels?",
        "How does the escalation matrix work?",
        "What is the refund and compensation policy?",
        "How should repeat complaints be handled?",
        "What is the best recipe for biryani?",
    ]

    for query in test_queries:

        result = grounded_answer(
            query,
            model,
            sentence_collection,
            threshold,
            top_k=3,
        )

        print("\n" + "-" * 70)

        print(
            f"Query: {query}"
        )

        print(
            f"Top-1 similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"Grounded: "
            f"{result['grounded']}"
        )

        print(
            f"Sources: "
            f"{result['sources']}"
        )

        print(
            f"\nAnswer:\n"
            f"{result['answer']}"
        )