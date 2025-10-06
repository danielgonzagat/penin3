from fastapi.testclient import TestClient
from darwinacci_omega.api.app import app


def test_status_and_start():
    client = TestClient(app)
    r = client.get('/status')
    assert r.status_code == 200
    r2 = client.post('/start', json={'cycles': 1, 'pop': 8, 'seed': 7})
    assert r2.status_code == 200
    data = r2.json()
    assert 'champion' in data


def test_portfolio_listing():
    client = TestClient(app)
    r = client.get('/portfolio')
    assert r.status_code == 200
    assert 'tasks' in r.json()
