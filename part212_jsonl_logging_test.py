import json
from pathlib import Path

from fastapi.testclient import TestClient

from part211_fastapi import app
from part212_jsonl_logging import LOG_PATH


client = TestClient(app)


def clear_log():
    """Start each test with an empty JSONL log."""
    if LOG_PATH.exists():
        LOG_PATH.unlink()


def read_log_records():
    """Read all JSONL records from the log file."""
    if not LOG_PATH.exists():
        return []

    records = []

    with LOG_PATH.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def test_http_request_logging():
    clear_log()

    response = client.post(
        "/support",
        json={
            "message": (
                "Please contact me at customer@example.com "
                "or +91 98765 43210 about my refund."
            )
        },
    )

    assert response.status_code == 200

    records = read_log_records()

    # Exactly one log entry for one HTTP request.
    assert len(records) == 1

    record = records[0]

    # Required Task 12 fields.
    required_keys = {
        "timestamp",
        "trace_id",
        "event_type",
        "duration_ms",
        "payload",
    }

    assert required_keys.issubset(record.keys())

    # Trace ID must exist.
    assert record["trace_id"]

    # Real request must have timing.
    assert isinstance(record["duration_ms"], (int, float))
    assert record["duration_ms"] >= 0

    # Correct event type.
    assert record["event_type"] == "support_request"

    # PII must be masked in the log.
    logged_message = record["payload"]["message"]

    assert "customer@example.com" not in logged_message
    assert "+91 98765 43210" not in logged_message

    assert "[EMAIL_REDACTED]" in logged_message
    assert "[PHONE_REDACTED]" in logged_message


def test_websocket_request_logging():
    clear_log()

    with client.websocket_connect("/ws") as websocket:
        websocket.send_text(
            "I need help with my refund. "
            "My email is rider@example.com."
        )

        response = websocket.receive_json()

        assert response["response_type"] == "support"

    records = read_log_records()

    # Exactly one log entry for one WebSocket message.
    assert len(records) == 1

    record = records[0]

    assert record["event_type"] == "websocket_support_request"
    assert record["trace_id"]
    assert isinstance(record["duration_ms"], (int, float))
    assert record["duration_ms"] >= 0

    logged_message = record["payload"]["message"]

    assert "rider@example.com" not in logged_message
    assert "[EMAIL_REDACTED]" in logged_message


def test_trace_ids_are_unique():
    clear_log()

    client.post(
        "/support",
        json={"message": "First request"},
    )

    client.post(
        "/support",
        json={"message": "Second request"},
    )

    records = read_log_records()

    assert len(records) == 2

    trace_ids = [record["trace_id"] for record in records]

    assert len(set(trace_ids)) == 2


if __name__ == "__main__":
    test_http_request_logging()
    test_websocket_request_logging()
    test_trace_ids_are_unique()

    print("TASK 12 AUTOMATED TESTS PASSED")