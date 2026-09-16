"""
Part 2.11 - Long-Term Memory

Persistent, local SQLite memory for the Ola support agent.  This is separate
from Part 2.8's per-process InMemoryChatMessageHistory: records in this module
remain available after the Python process exits.
"""

import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parent
MEMORY_DATABASE_PATH = PROJECT_DIRECTORY / "long_term_memory.sqlite3"

DEMO_CUSTOMER_ID = "demo_customer_001"
DEMO_MEMORY_KEY = "support_preference"
DEMO_TICKET_ID = "TKT-0002"
DEMO_FACT = "Customer prefers concise in-app updates for the active ticket."


@dataclass(frozen=True)
class LongTermMemory:
    """One persistent, customer-scoped support memory record."""

    customer_id: str
    memory_key: str
    ticket_id: str | None
    memory_fact: str
    updated_at: str


def _connect() -> sqlite3.Connection:
    """Open the project-local SQLite database and return rows by column name."""

    connection = sqlite3.connect(MEMORY_DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_long_term_memory() -> None:
    """Create the local persistent-memory table if it does not exist."""

    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS long_term_memory (
                customer_id TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                ticket_id TEXT,
                memory_fact TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (customer_id, memory_key)
            )
            """
        )


def upsert_long_term_memory(
    customer_id: str,
    memory_key: str,
    memory_fact: str,
    ticket_id: str | None = None,
) -> LongTermMemory:
    """Store or update a useful customer-specific support memory item."""

    initialize_long_term_memory()

    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO long_term_memory (
                customer_id, memory_key, ticket_id, memory_fact
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(customer_id, memory_key) DO UPDATE SET
                ticket_id = excluded.ticket_id,
                memory_fact = excluded.memory_fact,
                updated_at = CURRENT_TIMESTAMP
            """,
            (customer_id, memory_key, ticket_id, memory_fact),
        )

    memory = retrieve_long_term_memory(customer_id, memory_key)
    if memory is None:
        raise RuntimeError("Stored long-term memory could not be retrieved.")

    return memory


def retrieve_long_term_memory(
    customer_id: str,
    memory_key: str,
) -> LongTermMemory | None:
    """Retrieve one exact customer memory item from persistent local storage."""

    initialize_long_term_memory()

    with _connect() as connection:
        row = connection.execute(
            """
            SELECT customer_id, memory_key, ticket_id, memory_fact, updated_at
            FROM long_term_memory
            WHERE customer_id = ? AND memory_key = ?
            """,
            (customer_id, memory_key),
        ).fetchone()

    return _row_to_memory(row) if row else None


def search_long_term_memory(
    customer_id: str,
    search_text: str | None = None,
) -> list[LongTermMemory]:
    """Search a customer's persisted memory facts, optionally by text."""

    initialize_long_term_memory()

    query = (
        "SELECT customer_id, memory_key, ticket_id, memory_fact, updated_at "
        "FROM long_term_memory WHERE customer_id = ?"
    )
    parameters: list[str] = [customer_id]

    if search_text:
        query += " AND memory_fact LIKE ?"
        parameters.append(f"%{search_text}%")

    query += " ORDER BY updated_at DESC"

    with _connect() as connection:
        rows = connection.execute(query, parameters).fetchall()

    return [_row_to_memory(row) for row in rows]


def delete_long_term_memory(customer_id: str, memory_key: str) -> bool:
    """Delete one memory item; useful for controlled tests or user requests."""

    initialize_long_term_memory()

    with _connect() as connection:
        cursor = connection.execute(
            """
            DELETE FROM long_term_memory
            WHERE customer_id = ? AND memory_key = ?
            """,
            (customer_id, memory_key),
        )

    return cursor.rowcount > 0


def _row_to_memory(row: sqlite3.Row) -> LongTermMemory:
    """Convert a SQLite row to the explicit long-term-memory data type."""

    return LongTermMemory(
        customer_id=row["customer_id"],
        memory_key=row["memory_key"],
        ticket_id=row["ticket_id"],
        memory_fact=row["memory_fact"],
        updated_at=row["updated_at"],
    )


def verify_persisted_memory() -> None:
    """Run in a fresh process and prove that the parent process's record exists."""

    print("RUN 2 - FRESH PYTHON PROCESS")
    print("Persistent database:", MEMORY_DATABASE_PATH)

    memory = retrieve_long_term_memory(DEMO_CUSTOMER_ID, DEMO_MEMORY_KEY)
    assert memory is not None
    assert memory.ticket_id == DEMO_TICKET_ID
    assert memory.memory_fact == DEMO_FACT

    matches = search_long_term_memory(DEMO_CUSTOMER_ID, "in-app")
    assert any(match.memory_key == DEMO_MEMORY_KEY for match in matches)

    print("Retrieved persistent memory:", memory)
    print("Search retrieval count:", len(matches))
    print("RUN 2 PERSISTENCE VERIFICATION PASSED")


def run_demo() -> None:
    """Store a synthetic record, then prove persistence in a child process."""

    initialize_long_term_memory()
    print("=" * 72)
    print("PART 2.11 - LONG-TERM MEMORY DEMO")
    print("=" * 72)
    print("Persistent SQLite database:", MEMORY_DATABASE_PATH)

    # Make the demonstration deterministic while retaining the record after
    # successful completion for later manual retrieval.
    delete_long_term_memory(DEMO_CUSTOMER_ID, DEMO_MEMORY_KEY)
    before_store = retrieve_long_term_memory(DEMO_CUSTOMER_ID, DEMO_MEMORY_KEY)
    assert before_store is None
    print("RUN 1 - memory before storing:", before_store)

    stored_memory = upsert_long_term_memory(
        customer_id=DEMO_CUSTOMER_ID,
        memory_key=DEMO_MEMORY_KEY,
        ticket_id=DEMO_TICKET_ID,
        memory_fact=DEMO_FACT,
    )
    print("RUN 1 - stored synthetic memory:", stored_memory)
    assert MEMORY_DATABASE_PATH.exists()

    print("\nStarting RUN 2 in a separate Python process...")
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--verify"],
        cwd=PROJECT_DIRECTORY,
        check=True,
        capture_output=True,
        text=True,
    )
    print(completed.stdout, end="")

    print("\n" + "=" * 72)
    print("LONG-TERM MEMORY TEST PASSED")
    print("The SQLite record survived the first process and was retrieved by RUN 2.")
    print("=" * 72)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--verify":
        verify_persisted_memory()
    else:
        run_demo()
