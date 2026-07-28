"""Accounts, tiers, payments surface, and Pro feature tests.

No model calls and no live Paynow calls: payment/SMTP env is blanked so
the endpoints exercise their local validation paths only.

Run:  .venv/bin/python -m pytest tests/ -q
"""
import os
import tempfile
import uuid

os.environ.setdefault("ADMIN_TOKEN", "test-token")
os.environ.setdefault("RATE_LIMIT", "30")
os.environ.setdefault("ANON_IP_DAILY", "1000")
os.environ.setdefault("SIGNUP_IP_DAILY", "1000")
# fresh DB per run: never touch the real data/onegov.db
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="onegov-test-"))
# Force the "not configured" paths even when .env has live keys.
os.environ["PAYNOW_ID"] = ""
os.environ["PAYNOW_KEY"] = ""
os.environ["SMTP_HOST"] = ""

from fastapi.testclient import TestClient          # noqa: E402

import server                                      # noqa: E402

# https base URL so the Secure auth cookie is stored and sent
client = TestClient(server.app, base_url="https://testserver")

EMAIL = f"pytest-{uuid.uuid4().hex[:8]}@example.com"
PW = "pytest-password-1"


def _me():
    return client.get("/api/auth/me").json()


def test_signup_login_me():
    r = client.post("/api/auth/signup",
                    json={"email": EMAIL, "password": PW})
    assert r.status_code == 200 and r.json()["email"] == EMAIL
    # duplicate
    r = client.post("/api/auth/signup",
                    json={"email": EMAIL, "password": PW})
    assert r.status_code == 409
    # wrong password
    r = client.post("/api/auth/login",
                    json={"email": EMAIL, "password": "wrong-password"})
    assert r.status_code == 401
    # right password
    r = client.post("/api/auth/login",
                    json={"email": EMAIL, "password": PW})
    assert r.status_code == 200
    me = _me()
    assert me["email"] == EMAIL and me["plan"] == "free"
    assert me["daily_left"] is not None


def test_signup_validation():
    assert client.post("/api/auth/signup",
                       json={"email": "not-an-email",
                             "password": PW}).status_code == 400
    assert client.post("/api/auth/signup",
                       json={"email": "a@b.co",
                             "password": "short"}).status_code == 400


def test_checkout_paths():
    assert client.post("/api/pay/checkout",
                       json={"plan": "gold"}).status_code == 400
    # Paynow keys blanked -> 503 (never a live call)
    r = client.post("/api/pay/checkout", json={"plan": "plus"})
    assert r.status_code == 503


def test_pro_gating_and_features():
    # free account: paid endpoints refuse
    for path, code in (("/api/watches", "needs_upgrade"),
                       ("/api/keys", "needs_pro")):
        r = client.get(path)
        assert r.status_code == 402
        assert r.json()["detail"]["code"] == code
    # plus tier: watches allowed (capped at 3), API keys still Pro-only
    r = client.post("/api/admin/set-plan",
                    json={"token": "test-token", "email": EMAIL,
                          "plan": "plus"})
    assert r.status_code == 200
    for i in range(3):
        assert client.post("/api/watches",
                           json={"term": f"Test Act {i} name"}
                           ).status_code == 200
    r = client.post("/api/watches", json={"term": "One too many Act"})
    assert r.status_code == 400 and "Pro" in r.json()["detail"]
    for i in range(3):
        client.post("/api/watches/remove",
                    json={"term": f"test act {i} name"})
    assert client.get("/api/keys").status_code == 402
    # bulk ZIP (one-click packaging) is Pro-only
    assert client.get("/api/bulk/act/2024").status_code == 402
    # admin upgrade to pro
    r = client.post("/api/admin/set-plan",
                    json={"token": "test-token", "email": EMAIL,
                          "plan": "pro"})
    assert r.status_code == 200
    me = _me()
    assert me["plan"] == "pro" and me["days_left"] is not None
    # watches
    r = client.post("/api/watches", json={"term": "Labour Act"})
    assert r.status_code == 200
    ws = client.get("/api/watches").json()
    assert ws and ws[0]["term"] == "labour act"
    assert client.post("/api/watches/seen").status_code == 200
    assert client.post("/api/watches/remove",
                       json={"term": "labour act"}).status_code == 200
    # api keys
    key = client.post("/api/keys/create",
                      json={"label": "pytest"}).json()["key"]
    assert key.startswith("ogk_")
    listed = client.get("/api/keys").json()
    assert any(k["key"] == key for k in listed["keys"])
    # v1 ask auth surface (no model call: validation errors only)
    assert client.post("/api/v1/ask",
                       json={"question": "x"}).status_code == 401
    r = client.post("/api/v1/ask", json={"question": "  "},
                    headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 400            # metered, then rejected
    assert client.post("/api/keys/revoke",
                       json={"key": key}).status_code == 200
    r = client.post("/api/v1/ask", json={"question": "x"},
                    headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 401


def test_export():
    # pro plan still active from previous test; make a stored answer
    aid = server._store_answer("Test question?",
                               "# Heading\n\n**Bold** body.\n\n- item",
                               ["test search"])
    assert aid
    r = client.get(f"/api/export/{aid}", params={"fmt": "pdf"})
    assert r.status_code == 200
    assert r.content[:4] == b"%PDF"
    r = client.get(f"/api/export/{aid}", params={"fmt": "docx"})
    assert r.status_code == 200
    assert r.content[:2] == b"PK"           # zip container
    assert client.get(f"/api/export/{aid}",
                      params={"fmt": "exe"}).status_code == 400


def test_forgot_reset_paths():
    # SMTP blanked -> forgot reports 503
    r = client.post("/api/auth/forgot", json={"email": EMAIL})
    assert r.status_code == 503
    # invalid reset token
    r = client.post("/api/auth/reset",
                    json={"token": "nope", "password": "longenough1"})
    assert r.status_code == 400


def test_zz_cleanup_delete_account():
    r = client.post("/api/auth/delete")
    assert r.status_code == 200
    assert _me()["email"] is None
