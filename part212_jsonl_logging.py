import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from part210_guardrails import mask_pii


LOG_PATH = Path(__file__).resolve().parent / "logs" / "support_agent.jsonl"

VALID_EVENT_TYPES = {
    "support_request",
    "websocket_support_request",
}


def ensure_log_file():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not LOG_PATH.exists():
        LOG_PATH.touch()


def create_trace_id():
    """Create a unique trace ID for one support request."""
    return str(uuid.uuid4())


def log_event(event_type, payload, trace_id=None, started_at=None):
    """
    Write exactly one structured JSONL event.

    The payload is PII-masked before it is written to disk.
    Timing is measured from started_at when supplied.
    """

    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(f"Unsupported event_type: {event_type}")

    ensure_log_file()

    if trace_id is None:
        trace_id = create_trace_id()

    duration_ms = None

    if started_at is not None:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 3)

    masked_payload = payload.copy()

    if "message" in masked_payload:
        masked_payload["message"] = mask_pii(
            str(masked_payload["message"])
        )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "event_type": event_type,
        "duration_ms": duration_ms,
        "payload": masked_payload,
    }

    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

    return record


def validate_jsonl(path=LOG_PATH):
    """Validate that every JSONL record has the required Task 12 fields."""

    ensure_log_file()

    if not path.exists():
        raise FileNotFoundError(f"Log file does not exist: {path}")

    with path.open("r", encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            raw = line.strip()

            if not raw:
                continue

            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Record on line {line_number} is not a JSON object."
                )

            required_keys = {
                "timestamp",
                "trace_id",
                "event_type",
                "duration_ms",
                "payload",
            }

            missing = sorted(required_keys - record.keys())

            if missing:
                raise ValueError(
                    f"Record on line {line_number} is missing keys: "
                    f"{', '.join(missing)}"
                )

            if not record["trace_id"]:
                raise ValueError(
                    f"Record on line {line_number} has an empty trace_id."
                )

            if record["event_type"] not in VALID_EVENT_TYPES:
                raise ValueError(
                    f"Unsupported event_type "
                    f"'{record['event_type']}' on line {line_number}."
                )

            if record["duration_ms"] is not None:
                if not isinstance(
                    record["duration_ms"],
                    (int, float),
                ):
                    raise ValueError(
                        f"duration_ms on line {line_number} "
                        "must be numeric or null."
                    )

    return True


def main():
    ensure_log_file()

    start = time.perf_counter()

    log_event(
        "support_request",
        {
            "message": "Demo request for ticket priority",
            "response_type": "support",
        },
        started_at=start,
    )

    websocket_start = time.perf_counter()

    log_event(
        "websocket_support_request",
        {
            "message": "Demo websocket message",
        },
        started_at=websocket_start,
    )

    validate_jsonl(LOG_PATH)

    print(f"JSONL validation passed for {LOG_PATH}")


if __name__ == "__main__":
    main()