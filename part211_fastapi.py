import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from part212_jsonl_logging import create_trace_id, log_event


app = FastAPI(
    title="Ola Domain Support Agent",
    description="FastAPI deployment for the Ola support agent capstone.",
    version="1.0.0",
)


class SupportRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {
        "service": "Ola Domain Support Agent",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "mock_llm": True,
    }


@app.post("/support")
def support(request: SupportRequest):
    started_at = time.perf_counter()
    trace_id = create_trace_id()

    response = {
        "response_type": "support",
        "message": request.message,
        "response": (
            "Your support request was received. "
            "The Ola support agent will process it."
        ),
    }

    log_event(
        "support_request",
        {
            "message": request.message,
            "response_type": response["response_type"],
            "response": response["response"],
        },
        trace_id=trace_id,
        started_at=started_at,
    )

    return response

@app.websocket("/ws")
async def websocket_support(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            message = await websocket.receive_text()

            started_at = time.perf_counter()
            trace_id = create_trace_id()

            log_event(
                "websocket_support_request",
                {
                    "message": message,
                },
                trace_id=trace_id,
                started_at=started_at,
            )

            await websocket.send_json(
                {
                    "response_type": "support",
                    "message": message,
                    "response": (
                        "Your support request was received. "
                        "The Ola support agent will process it."
                    ),
                }
            )

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")