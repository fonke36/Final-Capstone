from fastapi.testclient import TestClient

from part211_fastapi import app


client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_support():
    response = client.post(
        "/support",
        json={"message": "What is the refund policy?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["response_type"] == "support"
    assert data["message"] == "What is the refund policy?"
    assert "response" in data


def test_websocket():
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text("I need help with my refund.")

        data = websocket.receive_json()

        assert data["response_type"] == "support"
        assert data["message"] == "I need help with my refund."
        assert "response" in data


if __name__ == "__main__":
    test_root()
    test_health()
    test_support()
    test_websocket()

    print("TASK 11 FASTAPI TESTS PASSED")