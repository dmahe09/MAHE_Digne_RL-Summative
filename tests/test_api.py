from fastapi.testclient import TestClient

from environment.api import app

client = TestClient(app)


def test_reset_and_step():
    j = client.post("/reset?seed=1").json()
    assert j["state"]["step"] == 0
    j = client.post("/step", json={"action": 0}).json()
    assert j["state"]["step"] == 1 and "observation" in j["state"]


def test_bad_action():
    assert client.post("/step", json={"action": 99}).status_code == 422


def test_actions_listing():
    assert len(client.get("/actions").json()) == 11
