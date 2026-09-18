"""
TASK 16 — IN-MEMORY RESPONSE CACHE

Purpose:
- Cache grounded-generation responses in memory.
- Use normalized query text as the cache key.
- Demonstrate a real cache miss followed by a real cache hit.
- Prove that the second identical query does not call
  the underlying grounded-generation function again.
- Keep the implementation local and compatible with MOCK_LLM
  / zero API keys / zero network.
"""

from typing import Any, Callable


# ---------------------------------------------------------
# In-memory response cache
# ---------------------------------------------------------

class GroundedResponseCache:
    """
    Simple in-memory cache for grounded-generation responses.

    The cache key is created from normalized query text:
    - leading/trailing whitespace removed
    - repeated internal whitespace collapsed
    - converted to lowercase
    """

    def __init__(self):
        self._cache: dict[str, dict[str, Any]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def normalize_query(query: str) -> str:
        """
        Normalize query text so equivalent queries can share
        the same cache entry.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        return " ".join(query.strip().lower().split())

    def get_or_generate(
        self,
        query: str,
        generator: Callable[[str], dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Return a cached response when available.

        On a cache miss, call the supplied grounded-generation
        function exactly once and store its result.
        """

        cache_key = self.normalize_query(query)

        if cache_key in self._cache:

            self.cache_hits += 1

            print("\nCACHE HIT")
            print(f"Normalized key: {cache_key}")

            return self._cache[cache_key]

        self.cache_misses += 1

        print("\nCACHE MISS")
        print(f"Normalized key: {cache_key}")

        response = generator(query)

        self._cache[cache_key] = response

        return response

    def size(self) -> int:
        """Return the number of cached responses."""

        return len(self._cache)


# ---------------------------------------------------------
# Task 16 verification
# ---------------------------------------------------------

def run_task16_test():
    """
    Verify the complete Task 16 requirement against the
    existing grounded-generation implementation.
    """

    from rag_generation import (
        load_rag_system,
        run_calibration,
        grounded_answer,
    )

    print("=" * 70)
    print("TASK 16 — IN-MEMORY RESPONSE CACHE")
    print("=" * 70)

    # -----------------------------------------------------
    # Load the existing local RAG system.
    # -----------------------------------------------------

    (
        model,
        fixed_collection,
        sentence_collection,
    ) = load_rag_system()

    # -----------------------------------------------------
    # Reuse the existing empirical calibration.
    # -----------------------------------------------------

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
        raise RuntimeError(
            "Task 16 cannot run because no valid "
            "groundedness threshold was calibrated."
        )

    print("\nEmpirical threshold:")
    print(f"{threshold:.4f}")

    # -----------------------------------------------------
    # Create cache.
    # -----------------------------------------------------

    cache = GroundedResponseCache()

    # This counter represents calls to the actual existing
    # grounded-generation function.
    generation_calls = {
        "count": 0
    }

    query = "What are the rules for ticket priority?"

    # -----------------------------------------------------
    # Wrapped grounded-generation function.
    # -----------------------------------------------------

    def counted_grounded_generation(
        incoming_query: str,
    ) -> dict[str, Any]:

        generation_calls["count"] += 1

        print(
            "\nUNDERLYING GROUNDED-GENERATION CALL"
        )

        print(
            f"Generation call number: "
            f"{generation_calls['count']}"
        )

        return grounded_answer(
            incoming_query,
            model,
            sentence_collection,
            threshold,
            top_k=3,
        )

    # -----------------------------------------------------
    # BEFORE — first request
    # -----------------------------------------------------

    print("\n")
    print("-" * 70)
    print("BEFORE / FIRST REQUEST")
    print("-" * 70)

    print(f"Original query: {query}")

    first_response = cache.get_or_generate(
        query,
        counted_grounded_generation,
    )

    print(
        f"Generation calls so far: "
        f"{generation_calls['count']}"
    )

    print(
        f"Cache size: "
        f"{cache.size()}"
    )

    # -----------------------------------------------------
    # AFTER — same normalized query
    # -----------------------------------------------------

    print("\n")
    print("-" * 70)
    print("AFTER / SECOND IDENTICAL REQUEST")
    print("-" * 70)

    second_response = cache.get_or_generate(
        query,
        counted_grounded_generation,
    )

    print(
        f"Generation calls after second request: "
        f"{generation_calls['count']}"
    )

    print(
        f"Cache hits: "
        f"{cache.cache_hits}"
    )

    print(
        f"Cache misses: "
        f"{cache.cache_misses}"
    )

    print(
        f"Cache size: "
        f"{cache.size()}"
    )

    # -----------------------------------------------------
    # Verify the responses are identical.
    # -----------------------------------------------------

    if first_response != second_response:
        raise AssertionError(
            "Cache returned a different response."
        )

    # -----------------------------------------------------
    # Verify exactly one underlying generation call.
    # -----------------------------------------------------

    if generation_calls["count"] != 1:
        raise AssertionError(
            "The second identical query caused an "
            "unnecessary grounded-generation call."
        )

    # -----------------------------------------------------
    # Verify one miss and one hit.
    # -----------------------------------------------------

    if cache.cache_misses != 1:
        raise AssertionError(
            "Expected exactly one cache miss."
        )

    if cache.cache_hits != 1:
        raise AssertionError(
            "Expected exactly one cache hit."
        )

    # -----------------------------------------------------
    # Verify normalized query keys.
    # -----------------------------------------------------

    normalized_a = cache.normalize_query(
        "What are the rules for ticket priority?"
    )

    normalized_b = cache.normalize_query(
        "  WHAT   ARE the rules for ticket priority?  "
    )

    if normalized_a != normalized_b:
        raise AssertionError(
            "Query normalization failed."
        )

    # -----------------------------------------------------
    # Final acceptance evidence.
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TASK 16 ACCEPTANCE EVIDENCE")
    print("=" * 70)

    print(
        "\n1. In-memory cache:"
        "\n   PASS"
    )

    print(
        "\n2. Cache key uses normalized query text:"
        "\n   PASS"
    )

    print(
        "\n3. First request was a cache miss:"
        "\n   PASS"
    )

    print(
        "\n4. Second identical request was a real cache hit:"
        "\n   PASS"
    )

    print(
        "\n5. Underlying grounded-generation calls:"
        f"\n   {generation_calls['count']} "
        "(expected 1)"
    )

    print(
        "\n6. Redundant second generation avoided:"
        "\n   PASS"
    )

    print(
        "\n7. Returned response identical:"
        "\n   PASS"
    )

    print(
        "\n8. Query normalization verified:"
        "\n   PASS"
    )

    print("\n")
    print("=" * 70)
    print("TASK 16 LOCAL TESTS PASSED")
    print("=" * 70)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    run_task16_test()