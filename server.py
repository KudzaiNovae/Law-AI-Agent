"""Web server for the public legislation chat.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000
"""
import datetime
import hashlib
import hmac
import json
import os
import queue
import re
import sqlite3
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import LawAgent, SYSTEM_PROMPT
from config import PROCESSED_DIR, RAW_DIR
from index.retriever import (HybridRetriever, Q_INSTR_RE, _tokenize,
                             normalize_refs)

# Original Gazette PDFs kept beside HTML sources (ZimLII Acts): the
# indexed text comes from the HTML, but the viewer should show the PDF.
ORIGINALS_DIR = RAW_DIR / "originals"


def _display_file(source_file: str) -> str:
    """Swap an HTML source for its original-PDF companion if we have one."""
    if source_file and source_file.endswith(".html"):
        pdf = source_file[:-5] + ".pdf"
        if (ORIGINALS_DIR / pdf).is_file():
            return pdf
    return source_file

app = FastAPI(title="OneGov Info")

_retriever = HybridRetriever()          # loads index once, shared by all
_agents: dict[str, LawAgent] = {}       # session_id -> agent (in-memory)
MAX_SESSIONS = 300
MAX_HISTORY = 30                        # messages kept per session

# ---------------- Persistence (SQLite) ----------------
# DATA_DIR can point at a mounted Azure Files volume so the DB (and logs)
# survive redeploys: az containerapp update ... --set-env-vars DATA_DIR=/data
DATA_DIR = Path(os.getenv("DATA_DIR", str(Path(__file__).parent / "data")))
DB_PATH = DATA_DIR / "onegov.db"


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=5)
    return conn


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db() as c:
        c.execute("CREATE TABLE IF NOT EXISTS sessions("
                  "sid TEXT PRIMARY KEY, messages TEXT, updated REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS answers("
                  "id TEXT PRIMARY KEY, question TEXT, answer TEXT, "
                  "searches TEXT, ts TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS cache("
                  "key TEXT PRIMARY KEY, answer TEXT, searches TEXT, "
                  "ts REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS users("
                  "id TEXT PRIMARY KEY, email TEXT UNIQUE, salt TEXT, "
                  "pw_hash TEXT, created TEXT, plan TEXT DEFAULT 'free')")
        for ddl in ("ALTER TABLE users ADD COLUMN plan TEXT DEFAULT 'free'",
                    "ALTER TABLE users ADD COLUMN paid_until REAL DEFAULT 0"):
            try:    # existing installs: add missing columns
                c.execute(ddl)
            except sqlite3.OperationalError:
                pass
        c.execute("CREATE TABLE IF NOT EXISTS payments("
                  "reference TEXT PRIMARY KEY, user_id TEXT, amount REAL, "
                  "status TEXT, pollurl TEXT, created REAL, "
                  "plan TEXT DEFAULT 'plus')")
        try:
            c.execute("ALTER TABLE payments ADD COLUMN plan TEXT "
                      "DEFAULT 'plus'")
        except sqlite3.OperationalError:
            pass
        c.execute("CREATE TABLE IF NOT EXISTS watches("
                  "user_id TEXT, term TEXT, created REAL, "
                  "seen_n INTEGER DEFAULT 0, PRIMARY KEY(user_id, term))")
        c.execute("CREATE TABLE IF NOT EXISTS api_keys("
                  "key TEXT PRIMARY KEY, user_id TEXT, label TEXT, "
                  "created REAL, revoked INTEGER DEFAULT 0)")
        c.execute("CREATE TABLE IF NOT EXISTS api_usage("
                  "key TEXT, month TEXT, count INTEGER DEFAULT 0, "
                  "PRIMARY KEY(key, month))")
        c.execute("CREATE TABLE IF NOT EXISTS export_usage("
                  "user_id TEXT, month TEXT, count INTEGER DEFAULT 0, "
                  "PRIMARY KEY(user_id, month))")
        c.execute("CREATE TABLE IF NOT EXISTS tokens("
                  "token TEXT PRIMARY KEY, user_id TEXT, created REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS resets("
                  "token TEXT PRIMARY KEY, user_id TEXT, created REAL, "
                  "used INTEGER DEFAULT 0)")
        c.execute("CREATE TABLE IF NOT EXISTS anon_usage("
                  "anon_id TEXT PRIMARY KEY, count INTEGER DEFAULT 0)")
        # Abuse guards: per-IP daily caps so clearing the og_anon cookie
        # or farming accounts does not grant endless free questions.
        c.execute("CREATE TABLE IF NOT EXISTS anon_ip_usage("
                  "day TEXT, ip TEXT, n INTEGER DEFAULT 0, "
                  "PRIMARY KEY(day, ip))")
        c.execute("CREATE TABLE IF NOT EXISTS signup_ip("
                  "day TEXT, ip TEXT, n INTEGER DEFAULT 0, "
                  "PRIMARY KEY(day, ip))")
        c.execute("CREATE TABLE IF NOT EXISTS user_usage("
                  "user_id TEXT, day TEXT, count INTEGER DEFAULT 0, "
                  "PRIMARY KEY(user_id, day))")
        c.execute("CREATE TABLE IF NOT EXISTS visits("
                  "day TEXT, ip_hash TEXT, n INTEGER DEFAULT 0, "
                  "PRIMARY KEY(day, ip_hash))")


_init_db()


def _save_session(sid: str, agent: LawAgent) -> None:
    """Persist chat history so sessions survive restarts/redeploys."""
    try:
        with _db() as c:
            c.execute("INSERT OR REPLACE INTO sessions VALUES(?,?,?)",
                      (sid, json.dumps(agent.messages, ensure_ascii=False),
                       time.time()))
            c.execute("DELETE FROM sessions WHERE updated < ?",
                      (time.time() - 7 * 86400,))
    except Exception:
        pass                            # anthropic messages aren't JSON


def _load_session(sid: str) -> list | None:
    try:
        with _db() as c:
            row = c.execute("SELECT messages FROM sessions WHERE sid=?",
                            (sid,)).fetchone()
        return json.loads(row[0]) if row else None
    except Exception:
        return None


def _store_answer(question: str, answer: str, searches: list) -> str:
    """Save a Q&A pair and return its shareable id."""
    aid = uuid.uuid4().hex[:10]
    try:
        with _db() as c:
            c.execute("INSERT INTO answers VALUES(?,?,?,?,?)",
                      (aid, question[:2000], answer,
                       json.dumps(searches or [], ensure_ascii=False),
                       datetime.datetime.now(datetime.UTC)
                       .isoformat(timespec="seconds")))
    except Exception:
        return ""
    return aid


# ---------------- Rate limiting (per IP) ----------------
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "8"))   # chat requests / min / IP
_hits: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    # Use the LAST X-Forwarded-For hop: earlier entries are supplied by
    # the client and trivially spoofable; the last one is appended by
    # the trusted ingress proxy in front of us.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[-1].strip()
    return request.client.host if request.client else "?"


def _check_rate(request: Request) -> None:
    ip = _client_ip(request)
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(429, "You are asking very fast. Please wait "
                                 "a minute and try again.")
    q.append(now)
    if len(_hits) > 20000:              # bound memory
        _hits.clear()


# ---- extra abuse guards (in-memory, per process) ----
# Login brute force: after LOGIN_MAX_FAILS wrong passwords for the same
# (ip, email) inside LOGIN_WINDOW_S, further attempts are blocked.
# A successful login clears the counter.
LOGIN_MAX_FAILS = int(os.getenv("LOGIN_MAX_FAILS", "8"))
LOGIN_WINDOW_S = 15 * 60
_login_fails: dict[str, deque] = defaultdict(deque)
# Password-reset mail: per email, at most one every 5 minutes and
# FORGOT_DAILY a day. Exceeding it silently skips the send, so the
# response never reveals whether the address exists.
FORGOT_DAILY = int(os.getenv("FORGOT_DAILY", "5"))
_forgot_hits: dict[str, deque] = defaultdict(deque)
# Bulk ZIP is heavy bandwidth: cap per user per day.
BULK_DAILY = int(os.getenv("BULK_DAILY", "10"))
_bulk_hits: dict[str, deque] = defaultdict(deque)
# Exports build PDFs/Word docs server side: cap per user per hour.
EXPORT_HOURLY = int(os.getenv("EXPORT_HOURLY", "60"))
_export_hits: dict[str, deque] = defaultdict(deque)
# scrypt is deliberately expensive: refuse absurd passwords pre-hash.
MAX_PW_LEN = 128


def _purge(q: deque, window_s: float) -> None:
    now = time.time()
    while q and now - q[0] > window_s:
        q.popleft()


def _bound(bucket: dict) -> None:
    if len(bucket) > 20000:             # bound memory
        bucket.clear()


# ---------------- Accounts (email + password) ----------------
# Anonymous visitors get FREE_QUESTIONS total, tracked by a device cookie
# (IP as fallback); after that, chat requires a free account. Free accounts
# get DAILY_FREE questions per day; the 'paid' plan is unlimited.
FREE_QUESTIONS = int(os.getenv("FREE_QUESTIONS", "5"))
DAILY_FREE = int(os.getenv("DAILY_FREE", "5"))
# Anti-abuse: anonymous questions per IP per day (catches cookie
# clearing) and new accounts per IP per day (catches account farming).
ANON_IP_DAILY = int(os.getenv("ANON_IP_DAILY", "15"))
SIGNUP_IP_DAILY = int(os.getenv("SIGNUP_IP_DAILY", "3"))
TOKEN_TTL_S = 180 * 86400              # signed-in for 180 days
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthRequest(BaseModel):
    email: str
    password: str


def _hash_pw(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode(), salt=salt,
                          n=2 ** 14, r=8, p=1, dklen=32).hex()


def _issue_token(user_id: str) -> str:
    token = uuid.uuid4().hex + uuid.uuid4().hex
    with _db() as c:
        c.execute("INSERT INTO tokens VALUES(?,?,?)",
                  (token, user_id, time.time()))
        c.execute("DELETE FROM tokens WHERE created < ?",
                  (time.time() - TOKEN_TTL_S,))
    return token


def _set_auth_cookie(resp: Response, token: str) -> None:
    resp.set_cookie("og_token", token, max_age=TOKEN_TTL_S, httponly=True,
                    samesite="lax", secure=True, path="/")


def _user_from(request: Request) -> dict | None:
    token = request.cookies.get("og_token")
    if not token:
        return None
    with _db() as c:
        row = c.execute(
            "SELECT u.id, u.email, u.plan, u.paid_until "
            "FROM tokens t JOIN users u "
            "ON u.id = t.user_id WHERE t.token=? AND t.created > ?",
            (token, time.time() - TOKEN_TTL_S)).fetchone()
    if not row:
        return None
    plan = row[2] if row[2] in ("plus", "pro") else "plus"  # legacy 'paid'
    if (row[3] or 0) <= time.time():
        plan = "free"
    return {"id": row[0], "email": row[1], "plan": plan,
            "paid_until": row[3] or 0}


def _anon_id(request: Request) -> str:
    return request.cookies.get("og_anon") or ("ip:" + _client_ip(request))


def _anon_count(anon_id: str) -> int:
    with _db() as c:
        row = c.execute("SELECT count FROM anon_usage WHERE anon_id=?",
                        (anon_id,)).fetchone()
    return row[0] if row else 0


def _today() -> str:
    return datetime.date.today().isoformat()


def _daily_count(user_id: str) -> int:
    with _db() as c:
        row = c.execute("SELECT count FROM user_usage WHERE user_id=? "
                        "AND day=?", (user_id, _today())).fetchone()
    return row[0] if row else 0


def _check_gate(request: Request) -> None:
    """Question gate: anonymous users get FREE_QUESTIONS total, then must
    create an account. Free accounts get DAILY_FREE per day; the 'paid'
    plan is unlimited."""
    user = _user_from(request)
    if user:
        if user["plan"] in ("plus", "pro"):
            return
        if _daily_count(user["id"]) >= DAILY_FREE:
            raise HTTPException(402, {
                "code": "needs_upgrade",
                "message": f"You have used your {DAILY_FREE} free "
                           "questions for today. Upgrade for unlimited "
                           "questions, or come back tomorrow."})
        with _db() as c:
            c.execute("INSERT INTO user_usage VALUES(?,?,1) "
                      "ON CONFLICT(user_id, day) "
                      "DO UPDATE SET count = count + 1",
                      (user["id"], _today()))
        return
    aid = _anon_id(request)
    ip = _client_ip(request)
    with _db() as c:
        row = c.execute("SELECT n FROM anon_ip_usage WHERE day=? AND ip=?",
                        (_today(), ip)).fetchone()
    ip_count = row[0] if row else 0
    # Gate on the device cookie AND a per-IP daily cap, so clearing or
    # rotating the cookie does not reset the free allowance.
    if _anon_count(aid) >= FREE_QUESTIONS or ip_count >= ANON_IP_DAILY:
        raise HTTPException(401, {
            "code": "needs_auth",
            "message": f"You have used your {FREE_QUESTIONS} free "
                       "questions. Create a free account to keep asking."})
    with _db() as c:
        c.execute("INSERT INTO anon_usage VALUES(?,1) ON CONFLICT(anon_id) "
                  "DO UPDATE SET count = count + 1", (aid,))
        c.execute("INSERT INTO anon_ip_usage VALUES(?,?,1) "
                  "ON CONFLICT(day, ip) DO UPDATE SET n = n + 1",
                  (_today(), ip))


@app.post("/api/auth/signup")
def auth_signup(body: AuthRequest, request: Request, response: Response):
    _check_rate(request)
    email = body.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(400, "Enter a valid email address.")
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")
    if len(body.password) > MAX_PW_LEN:
        raise HTTPException(400, "Password is too long (max 128 characters).")
    ip = _client_ip(request)
    with _db() as c:
        row = c.execute("SELECT n FROM signup_ip WHERE day=? AND ip=?",
                        (_today(), ip)).fetchone()
    if (row[0] if row else 0) >= SIGNUP_IP_DAILY:
        raise HTTPException(429, "Too many new accounts created from this "
                                 "network today. Please try again tomorrow.")
    salt = os.urandom(16)
    uid = uuid.uuid4().hex
    try:
        with _db() as c:
            c.execute("INSERT INTO users(id, email, salt, pw_hash, created)"
                      " VALUES(?,?,?,?,?)",
                      (uid, email, salt.hex(),
                       _hash_pw(body.password, salt),
                       datetime.datetime.now(datetime.UTC)
                       .isoformat(timespec="seconds")))
    except sqlite3.IntegrityError:
        raise HTTPException(
            409, "An account with this email already exists. Try logging in.")
    with _db() as c:
        c.execute("INSERT INTO signup_ip VALUES(?,?,1) "
                  "ON CONFLICT(day, ip) DO UPDATE SET n = n + 1",
                  (_today(), ip))
    # Anonymous questions already used on this device carry over into
    # today's allowance, so signing up is not a fresh batch of 5. Capped
    # at DAILY_FREE - 1 so a new account always gets at least 1 question
    # today; the full DAILY_FREE resets tomorrow.
    carried = min(_anon_count(_anon_id(request)), DAILY_FREE - 1)
    if carried > 0:
        with _db() as c:
            c.execute("INSERT INTO user_usage VALUES(?,?,?) "
                      "ON CONFLICT(user_id, day) DO UPDATE SET count=?",
                      (uid, _today(), carried, carried))
    _set_auth_cookie(response, _issue_token(uid))
    print(f"SIGNUP {email}", flush=True)
    return {"email": email}


@app.post("/api/auth/login")
def auth_login(body: AuthRequest, request: Request, response: Response):
    _check_rate(request)
    email = body.email.strip().lower()
    if len(body.password) > MAX_PW_LEN:
        raise HTTPException(401, "Wrong email or password.")
    key = f"{_client_ip(request)}|{email}"
    fails = _login_fails[key]
    _purge(fails, LOGIN_WINDOW_S)
    if len(fails) >= LOGIN_MAX_FAILS:
        raise HTTPException(429, "Too many failed sign-in attempts. "
                                 "Please wait 15 minutes and try again.")
    with _db() as c:
        row = c.execute("SELECT id, salt, pw_hash FROM users WHERE email=?",
                        (email,)).fetchone()
    if not row or not hmac.compare_digest(
            _hash_pw(body.password, bytes.fromhex(row[1])), row[2]):
        fails.append(time.time())
        _bound(_login_fails)
        raise HTTPException(401, "Wrong email or password.")
    _login_fails.pop(key, None)
    _set_auth_cookie(response, _issue_token(row[0]))
    return {"email": email}


@app.post("/api/auth/logout")
def auth_logout(request: Request, response: Response):
    token = request.cookies.get("og_token")
    if token:
        with _db() as c:
            c.execute("DELETE FROM tokens WHERE token=?", (token,))
    response.delete_cookie("og_token", path="/")
    return {"ok": True}


@app.post("/api/auth/delete")
def auth_delete(request: Request, response: Response):
    """Delete the signed-in user's account and tokens (privacy policy)."""
    user = _user_from(request)
    if not user:
        raise HTTPException(401, "Not signed in.")
    with _db() as c:
        c.execute("DELETE FROM tokens WHERE user_id=?", (user["id"],))
        c.execute("DELETE FROM users WHERE id=?", (user["id"],))
    response.delete_cookie("og_token", path="/")
    print(f"ACCOUNT_DELETE {user['email']}", flush=True)
    return {"ok": True}


@app.post("/api/account/cancel-plan")
def cancel_plan(request: Request):
    """Cancel a paid pass: immediate downgrade to Free, no refund.
    Passes are one-off 30-day payments, so there is no renewal to stop;
    this simply ends the current pass early at the user's request."""
    user = _user_from(request)
    if not user:
        raise HTTPException(401, {"code": "needs_auth",
                                  "message": "Sign in first."})
    if user["plan"] == "free":
        raise HTTPException(400, "No active paid plan to cancel.")
    with _db() as c:
        c.execute("UPDATE users SET plan='free', paid_until=0 WHERE id=?",
                  (user["id"],))
    print(f"PLAN_CANCELLED {user['email']} was={user['plan']}", flush=True)
    return {"ok": True, "plan": "free"}


@app.get("/api/auth/me")
def auth_me(request: Request, response: Response):
    """Session state for the UI; also assigns the anonymous device cookie."""
    user = _user_from(request)
    if user:
        left = (None if user["plan"] != "free"
                else max(0, DAILY_FREE - _daily_count(user["id"])))
        until = (datetime.date.fromtimestamp(user["paid_until"]).isoformat()
                 if user["plan"] != "free" else None)
        days_left = (max(0, int((user["paid_until"] - time.time()) // 86400))
                     if user["plan"] != "free" else None)
        exports_left = None
        watches_used = watch_limit = None
        if user["plan"] in ("plus", "pro"):
            with _db() as c:
                if user["plan"] == "plus":
                    row = c.execute(
                        "SELECT count FROM export_usage WHERE user_id=? "
                        "AND month=?", (user["id"], _month())).fetchone()
                    exports_left = max(0, EXPORTS_PLUS_MONTHLY
                                       - (row[0] if row else 0))
                watches_used = c.execute(
                    "SELECT COUNT(*) FROM watches WHERE user_id=?",
                    (user["id"],)).fetchone()[0]
            watch_limit = WATCH_LIMITS[user["plan"]]
        return {"email": user["email"], "plan": user["plan"],
                "daily_left": left, "paid_until": until,
                "days_left": days_left, "free_left": None,
                "exports_left": exports_left,
                "watches_used": watches_used, "watch_limit": watch_limit}
    aid = request.cookies.get("og_anon")
    if not aid:
        aid = uuid.uuid4().hex
        response.set_cookie("og_anon", aid, max_age=365 * 86400,
                            httponly=True, samesite="lax", secure=True,
                            path="/")
    return {"email": None,
            "free_left": max(0, FREE_QUESTIONS - _anon_count(aid))}


# ---------------- Password reset (email) ----------------
# Works once SMTP_* env vars are set (any provider: Gmail app password,
# Zoho, Resend SMTP, ...). Until then /api/auth/forgot returns 503.
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "noreply@onegovai.app")
RESET_TTL_S = 3600                      # links valid for 1 hour


def _send_email(to: str, subject: str, text: str) -> None:
    import smtplib
    from email.message import EmailMessage
    msg = EmailMessage()
    msg["From"] = SMTP_FROM
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(text)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
        s.starttls()
        if SMTP_USER:
            s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)


class ForgotRequest(BaseModel):
    email: str = ""


class ResetRequest(BaseModel):
    token: str = ""
    password: str = ""


@app.post("/api/auth/forgot")
def auth_forgot(body: ForgotRequest, request: Request):
    """Send a reset link. Same response whether or not the email exists,
    so addresses cannot be probed."""
    _check_rate(request)
    if not SMTP_HOST:
        raise HTTPException(503, "Password reset by email is not set up "
                                 "yet. Please contact us.")
    email = body.email.strip().lower()
    # Per-email flood guard: 1 mail / 5 min, FORGOT_DAILY / day. Over the
    # limit we skip the send but return the same reply (no probing).
    fq = _forgot_hits[email]
    _purge(fq, 86400)
    if (fq and time.time() - fq[-1] < 300) or len(fq) >= FORGOT_DAILY:
        return {"ok": True, "message": "If that address has an account, a "
                                       "reset link is on its way."}
    fq.append(time.time())
    _bound(_forgot_hits)
    with _db() as c:
        row = c.execute("SELECT id FROM users WHERE email=?",
                        (email,)).fetchone()
    if row:
        token = uuid.uuid4().hex + uuid.uuid4().hex
        with _db() as c:
            c.execute("INSERT INTO resets VALUES(?,?,?,0)",
                      (token, row[0], time.time()))
            c.execute("DELETE FROM resets WHERE created < ?",
                      (time.time() - 7 * 86400,))
        link = f"{BASE_URL}/reset.html?token={token}"
        try:
            _send_email(email, "Reset your OneGov AI password",
                        "Someone asked to reset the password for this "
                        "OneGov AI account.\n\nReset it here (valid for "
                        f"1 hour):\n{link}\n\nIf this was not you, you "
                        "can ignore this email.")
        except Exception as e:
            print(f"RESET_MAIL_FAIL {email} {e}", flush=True)
            raise HTTPException(502, "Could not send the email. Please "
                                     "try again shortly.")
        print(f"RESET_SENT {email}", flush=True)
    return {"ok": True, "message": "If that address has an account, a "
                                   "reset link is on its way."}


@app.post("/api/auth/reset")
def auth_reset(body: ResetRequest, request: Request):
    _check_rate(request)
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")
    if len(body.password) > MAX_PW_LEN:
        raise HTTPException(400, "Password is too long (max 128 characters).")
    with _db() as c:
        row = c.execute("SELECT user_id, created, used FROM resets "
                        "WHERE token=?", (body.token.strip(),)).fetchone()
    if (not row or row[2]
            or time.time() - row[1] > RESET_TTL_S):
        raise HTTPException(400, "This reset link is invalid or has "
                                 "expired. Request a new one.")
    salt = os.urandom(16)
    with _db() as c:
        c.execute("UPDATE users SET salt=?, pw_hash=? WHERE id=?",
                  (salt.hex(), _hash_pw(body.password, salt), row[0]))
        c.execute("UPDATE resets SET used=1 WHERE token=?",
                  (body.token.strip(),))
        c.execute("DELETE FROM tokens WHERE user_id=?", (row[0],))
    print(f"RESET_DONE {row[0][:8]}", flush=True)
    return {"ok": True}


# ---------------- Payments (Paynow) ----------------
# $PAID_PRICE_USD buys PAID_DAYS of unlimited questions. Flow:
#   /api/pay/checkout -> Paynow browserurl (EcoCash / OneMoney / card)
#   -> Paynow POSTs /api/paynow/result -> we verify against the pollurl
#   (never trusting the webhook body) -> extend users.paid_until.
# Works as soon as PAYNOW_ID / PAYNOW_KEY are set in the environment;
# until then /api/pay/checkout returns 503 and the UI falls back to the
# contact-us path. Register at https://www.paynow.co.zw to get keys.
import urllib.parse
import urllib.request

PAYNOW_ID = os.getenv("PAYNOW_ID", "")
PAYNOW_KEY = os.getenv("PAYNOW_KEY", "")
# While the Paynow integration is in test mode, authemail must be the
# merchant's registered email. Set PAYNOW_TEST_EMAIL to that address;
# unset it once Paynow flips the integration to live.
PAYNOW_TEST_EMAIL = os.getenv("PAYNOW_TEST_EMAIL", "")
PAYNOW_URL = "https://www.paynow.co.zw/interface/initiatetransaction"
PAID_DAYS = int(os.getenv("PAID_DAYS", "30"))
BASE_URL = os.getenv("BASE_URL", "https://onegovai.app").rstrip("/")

# Two paid tiers, both PAID_DAYS-day passes:
#   plus — unlimited questions
#   pro  — unlimited + PDF/Word export, amendment watch alerts, API access
PLANS = {
    "plus": {"price": float(os.getenv("PLUS_PRICE_USD", "5")),
             "label": "Plus"},
    "pro": {"price": float(os.getenv("PRO_PRICE_USD", "15")),
            "label": "Pro"},
}


def _paynow_hash(fields: dict) -> str:
    """SHA512 of the concatenated field values + integration key."""
    s = "".join(str(v) for k, v in fields.items() if k.lower() != "hash")
    return hashlib.sha512((s + PAYNOW_KEY).encode()).hexdigest().upper()


def _paynow_post(url: str, fields: dict | None = None) -> dict:
    data = urllib.parse.urlencode(fields).encode() if fields else b""
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=8) as resp:
        return dict(urllib.parse.parse_qsl(resp.read().decode()))


def _paynow_verified(resp: dict) -> bool:
    """Check the SHA512 hash Paynow signs its responses with. A response
    without a valid hash must never activate a plan."""
    got = resp.get("hash", "")
    return bool(got) and _paynow_hash(resp) == got.upper()


def _poll_state(pollurl: str, expected_amount: float) -> str:
    """Authoritative check: poll Paynow. Returns 'paid' only for a
    hash-verified response with a paid status and the right amount,
    'failed' for a terminal non-payment (cancelled/failed/disputed/
    refunded), else 'pending'."""
    resp = _paynow_post(pollurl)
    status = resp.get("status", "").lower()
    if status in PAYNOW_FAILED:
        return "failed"
    if status not in PAYNOW_PAID:
        return "pending"
    if not _paynow_verified(resp):
        print(f"PAYNOW_BAD_HASH {resp.get('reference', '?')}", flush=True)
        return "pending"
    try:
        amount = float(resp.get("amount", "0"))
    except ValueError:
        amount = 0.0
    if amount + 0.01 < expected_amount:
        print(f"PAYNOW_BAD_AMOUNT {resp.get('reference', '?')} "
              f"{amount} < {expected_amount}", flush=True)
        return "pending"
    return "paid"


class CheckoutRequest(BaseModel):
    plan: str = "plus"                  # 'plus' | 'pro'


@app.post("/api/pay/checkout")
def pay_checkout(body: CheckoutRequest, request: Request):
    user = _user_from(request)
    if not user:
        raise HTTPException(401, "Sign in first to upgrade.")
    if body.plan not in PLANS:
        raise HTTPException(400, "Unknown plan.")
    if not (PAYNOW_ID and PAYNOW_KEY):
        raise HTTPException(503, "Online payment is not available yet. "
                                 "Please contact us to upgrade.")
    _check_rate(request)
    price = PLANS[body.plan]["price"]
    ref = "og-" + uuid.uuid4().hex[:12]
    fields = {                          # order matters for the hash
        "resulturl": f"{BASE_URL}/api/paynow/result",
        "returnurl": f"{BASE_URL}/app.html?paid=pending",
        "reference": ref,
        "amount": f"{price:.2f}",
        "id": PAYNOW_ID,
        "additionalinfo": (f"OneGov AI {PLANS[body.plan]['label']} pass "
                           f"({PAID_DAYS} days) — hosting & service fees"),
        "authemail": PAYNOW_TEST_EMAIL or user["email"],
        "status": "Message",
    }
    fields["hash"] = _paynow_hash(fields)
    try:
        resp = _paynow_post(PAYNOW_URL, fields)
    except Exception as e:
        print(f"PAYNOW_INIT_FAIL {e}", flush=True)
        raise HTTPException(502, "The payment service is not responding. "
                                 "Please try again shortly.")
    if resp.get("status", "").lower() != "ok":
        print(f"PAYNOW_INIT_ERR {resp}", flush=True)
        raise HTTPException(502, "Could not start the payment. "
                                 "Please try again shortly.")
    with _db() as c:
        c.execute("INSERT INTO payments"
                  "(reference, user_id, amount, status, pollurl, created, "
                  "plan) VALUES(?,?,?,?,?,?,?)",
                  (ref, user["id"], price, "created",
                   resp.get("pollurl", ""), time.time(), body.plan))
    return {"url": resp["browserurl"], "reference": ref}


def _activate_paid(user_id: str, plan: str = "plus") -> None:
    if plan not in PLANS:
        plan = "plus"
    now = time.time()
    with _db() as c:
        row = c.execute("SELECT paid_until FROM users WHERE id=?",
                        (user_id,)).fetchone()
        base = max(now, row[0] or 0) if row else now
        c.execute("UPDATE users SET plan=?, paid_until=? WHERE id=?",
                  (plan, base + PAID_DAYS * 86400, user_id))


PAYNOW_PAID = {"paid", "awaiting delivery", "delivered"}
PAYNOW_FAILED = {"cancelled", "failed", "disputed", "refunded"}


@app.post("/api/paynow/result")
async def paynow_result(request: Request):
    """Paynow webhook. The body is only used to find our reference; the
    authoritative status always comes from polling Paynow directly."""
    body = (await request.body()).decode()
    data = dict(urllib.parse.parse_qsl(body))
    ref = data.get("reference", "")
    with _db() as c:
        row = c.execute("SELECT user_id, pollurl, status, plan, amount "
                        "FROM payments WHERE reference=?", (ref,)).fetchone()
    if not row or not row[1]:
        raise HTTPException(404, "Unknown reference")
    if row[2] == "paid":                        # idempotent
        return "ok"
    try:
        state = _poll_state(row[1], row[4] or 0)
    except Exception as e:
        print(f"PAYNOW_POLL_FAIL {ref} {e}", flush=True)
        raise HTTPException(502, "poll failed")
    if state == "paid":
        _activate_paid(row[0], row[3] or "plus")
        with _db() as c:
            c.execute("UPDATE payments SET status='paid' "
                      "WHERE reference=?", (ref,))
        print(f"PAYNOW_PAID {ref}", flush=True)
    else:
        new = "failed" if state == "failed" else "unpaid"
        with _db() as c:
            c.execute("UPDATE payments SET status=? WHERE reference=?",
                      (new, ref))
        if new == "failed":
            print(f"PAYNOW_FAILED {ref}", flush=True)
    return "ok"


@app.post("/api/pay/check")
def pay_check(request: Request):
    """Client-driven fallback to the webhook: poll Paynow for the signed-in
    user's recent unpaid payments and activate the plan if any is paid.
    Called by the app while it shows 'Confirming your payment…'."""
    user = _user_from(request)
    if not user:
        raise HTTPException(401, "Sign in first.")
    _check_rate(request)
    with _db() as c:
        rows = c.execute(
            "SELECT reference, pollurl, plan, amount FROM payments "
            "WHERE user_id=? AND status NOT IN ('paid','failed') "
            "AND pollurl != '' AND created > ? "
            "ORDER BY created DESC LIMIT 3",
            (user["id"], time.time() - 86400)).fetchall()
    activated = False
    failed = False
    for ref, pollurl, plan, amount in rows:
        try:
            state = _poll_state(pollurl, amount or 0)
        except Exception as e:
            print(f"PAYNOW_POLL_FAIL {ref} {e}", flush=True)
            continue
        if state == "paid":
            _activate_paid(user["id"], plan or "plus")
            with _db() as c:
                c.execute("UPDATE payments SET status='paid' "
                          "WHERE reference=?", (ref,))
            print(f"PAYNOW_PAID {ref} (client poll)", flush=True)
            activated = True
        elif state == "failed":
            # Terminal: stop polling this payment on future checks.
            with _db() as c:
                c.execute("UPDATE payments SET status='failed' "
                          "WHERE reference=?", (ref,))
            print(f"PAYNOW_FAILED {ref} (client poll)", flush=True)
            failed = True
    fresh = _user_from(request)
    return {"activated": activated,
            "failed": failed and not activated,
            "plan": fresh["plan"] if fresh else user["plan"]}


# ---------------- Paid features (export, watches, API keys) ---------
API_MONTHLY = int(os.getenv("API_MONTHLY", "1000"))   # calls / key / month
MAX_KEYS = 5
# Tier differentiation: Plus gets a taste, Pro gets the full toolset.
WATCH_LIMITS = {"plus": 3, "pro": 20}
EXPORTS_PLUS_MONTHLY = int(os.getenv("EXPORTS_PLUS_MONTHLY", "10"))
MAX_WATCHES = WATCH_LIMITS["pro"]                     # legacy alias


def _require_paid(request: Request) -> dict:
    """An active, unexpired paid pass (Plus or Pro)."""
    user = _user_from(request)
    if not user:
        raise HTTPException(401, {"code": "needs_auth",
                                  "message": "Sign in first."})
    if user["plan"] not in ("plus", "pro") \
            or user["paid_until"] <= time.time():
        raise HTTPException(402, {
            "code": "needs_upgrade",
            "message": "This feature is part of the paid plans."})
    return user


def _require_pro(request: Request) -> dict:
    user = _user_from(request)
    if not user:
        raise HTTPException(401, {"code": "needs_auth",
                                  "message": "Sign in first."})
    # Belt and braces: an active, unexpired Pro pass — never rely on the
    # plan label alone.
    if user["plan"] != "pro" or user["paid_until"] <= time.time():
        raise HTTPException(402, {
            "code": "needs_pro",
            "message": "This feature is part of the Pro plan."})
    return user


_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def _md_blocks(answer: str) -> list[tuple[str, str]]:
    """Very small markdown reader: (kind, text) blocks for export."""
    blocks: list[tuple[str, str]] = []
    for raw in answer.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            blocks.append(("heading", line.lstrip("# ").strip()))
        elif line.startswith(">"):
            blocks.append(("quote", line.lstrip("> ").strip()))
        elif line[:2] in ("- ", "* ") or re.match(r"^\d+\.\s", line):
            blocks.append(("bullet", re.sub(r"^(-|\*|\d+\.)\s+", "", line)))
        else:
            blocks.append(("para", line))
    return blocks


_EXPORT_FOOTER = ("Generated by OneGov AI (onegovai.app) from Zimbabwe's "
                  "gazetted laws. General information, not legal advice.")


def _export_pdf(question: str, answer: str, ts: str) -> bytes:
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer)
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=20 * mm,
                            rightMargin=20 * mm, topMargin=18 * mm,
                            bottomMargin=18 * mm, title=question[:120])
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["Normal"], fontSize=10.5,
                          leading=15, spaceAfter=6)
    quote = ParagraphStyle("quote", parent=body, leftIndent=14,
                           textColor="#3730a3")
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=14,
                            bulletIndent=4)
    small = ParagraphStyle("small", parent=body, fontSize=8.5,
                           textColor="#71717a")

    def esc(s: str) -> str:
        s = (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;"))
        return _MD_BOLD_RE.sub(r"<b>\1</b>", s)

    story = [Paragraph(esc(question), ss["Title"]),
             Paragraph(f"OneGov AI — {ts}", small),
             Spacer(1, 8)]
    for kind, text in _md_blocks(answer):
        if kind == "heading":
            story.append(Paragraph(esc(text), ss["Heading2"]))
        elif kind == "quote":
            story.append(Paragraph(esc(text), quote))
        elif kind == "bullet":
            story.append(Paragraph(esc(text), bullet, bulletText="•"))
        else:
            story.append(Paragraph(esc(text), body))
    story += [Spacer(1, 14), Paragraph(esc(_EXPORT_FOOTER), small)]
    doc.build(story)
    return buf.getvalue()


def _export_docx(question: str, answer: str, ts: str) -> bytes:
    from io import BytesIO
    import docx  # python-docx
    d = docx.Document()
    d.add_heading(question, level=1)
    p = d.add_paragraph(f"OneGov AI — {ts}")
    p.runs[0].font.size = docx.shared.Pt(8)

    def add_runs(par, text):
        pos = 0
        for m in _MD_BOLD_RE.finditer(text):
            if m.start() > pos:
                par.add_run(text[pos:m.start()])
            par.add_run(m.group(1)).bold = True
            pos = m.end()
        if pos < len(text):
            par.add_run(text[pos:])

    for kind, text in _md_blocks(answer):
        if kind == "heading":
            d.add_heading(_MD_BOLD_RE.sub(r"\1", text), level=2)
        elif kind == "quote":
            add_runs(d.add_paragraph(style="Intense Quote"), text)
        elif kind == "bullet":
            add_runs(d.add_paragraph(style="List Bullet"), text)
        else:
            add_runs(d.add_paragraph(), text)
    f = d.add_paragraph(_EXPORT_FOOTER)
    f.runs[0].font.size = docx.shared.Pt(8)
    buf = BytesIO()
    d.save(buf)
    return buf.getvalue()


@app.get("/api/export/{aid}")
def export_answer(aid: str, request: Request, fmt: str = "pdf"):
    """Paid: download a saved answer as PDF or Word, citations included.
    Plus: EXPORTS_PLUS_MONTHLY per month. Pro: unlimited."""
    user = _require_paid(request)
    if fmt not in ("pdf", "docx"):
        raise HTTPException(400, "fmt must be 'pdf' or 'docx'")
    eq = _export_hits[user["id"]]
    _purge(eq, 3600)
    if len(eq) >= EXPORT_HOURLY:
        raise HTTPException(429, "Too many exports. Please wait an hour "
                                 "and try again.")
    eq.append(time.time())
    _bound(_export_hits)
    if user["plan"] == "plus":
        with _db() as c:
            row = c.execute("SELECT count FROM export_usage WHERE "
                            "user_id=? AND month=?",
                            (user["id"], _month())).fetchone()
            if row and row[0] >= EXPORTS_PLUS_MONTHLY:
                raise HTTPException(402, {
                    "code": "needs_pro",
                    "message": f"Plus includes {EXPORTS_PLUS_MONTHLY} "
                               "exports a month — upgrade to Pro for "
                               "unlimited exports."})
            c.execute("INSERT INTO export_usage VALUES(?,?,1) "
                      "ON CONFLICT(user_id, month) "
                      "DO UPDATE SET count = count + 1",
                      (user["id"], _month()))
    with _db() as c:
        row = c.execute("SELECT question, answer, ts FROM answers "
                        "WHERE id=?", (aid,)).fetchone()
    if not row:
        raise HTTPException(404, "No such answer")
    try:
        data = (_export_pdf(*row) if fmt == "pdf" else _export_docx(*row))
    except Exception as e:
        print(f"EXPORT_FAIL {aid} {e}", flush=True)
        raise HTTPException(500, "Could not generate the file.")
    media = ("application/pdf" if fmt == "pdf" else
             "application/vnd.openxmlformats-officedocument."
             "wordprocessingml.document")
    return Response(data, media_type=media, headers={
        "Content-Disposition":
            f'attachment; filename="onegov-answer-{aid}.{fmt}"'})


class WatchRequest(BaseModel):
    term: str = ""


_WATCH_FILLER = {"of", "the", "and", "act", "chapter", "a", "an"}


def _watch_matches(term: str) -> list[dict]:
    """Catalog items whose titles contain every meaningful token of the
    watched term, newest first."""
    toks = set(_tokenize(term)) - _WATCH_FILLER
    if not toks:
        return []
    out = []
    for i in _retriever.catalog():
        tset = set(_tokenize(i["title"]))
        if toks <= tset:
            out.append({"title": i["title"], "doc_type": i["doc_type"],
                        "year": i["year"]})

    def key(item):
        m = _INSTR_NUM_RE.search(item["title"])
        return ((int(m.group(2)), int(m.group(1))) if m
                else (item["year"] or 0, 0))
    return sorted(out, key=key, reverse=True)[:8]


@app.get("/api/watches")
def watches_list(request: Request):
    """Pro: watched instruments with matching gazetted items and a count
    of items that are new since the user last looked."""
    user = _require_paid(request)
    with _db() as c:
        rows = c.execute("SELECT term, seen_n FROM watches WHERE user_id=? "
                         "ORDER BY created", (user["id"],)).fetchall()
    out = []
    for term, seen_n in rows:
        matches = _watch_matches(term)
        out.append({"term": term, "matches": matches,
                    "new": max(0, len(matches) - (seen_n or 0))})
    return out


@app.post("/api/watches")
def watches_add(body: WatchRequest, request: Request):
    user = _require_paid(request)
    term = " ".join(body.term.split())[:120]
    if len(term) < 3:
        raise HTTPException(400, "Enter an instrument name to watch.")
    limit = WATCH_LIMITS.get(user["plan"], WATCH_LIMITS["plus"])
    with _db() as c:
        n = c.execute("SELECT COUNT(*) FROM watches WHERE user_id=?",
                      (user["id"],)).fetchone()[0]
        if n >= limit:
            msg = (f"Limit of {limit} watches on {user['plan'].title()}."
                   + (" Upgrade to Pro for "
                      f"{WATCH_LIMITS['pro']}." if user["plan"] == "plus"
                      else ""))
            raise HTTPException(400, msg)
        c.execute("INSERT OR IGNORE INTO watches VALUES(?,?,?,0)",
                  (user["id"], term.lower(), time.time()))
    return {"term": term.lower()}


@app.post("/api/watches/remove")
def watches_remove(body: WatchRequest, request: Request):
    user = _require_paid(request)
    with _db() as c:
        c.execute("DELETE FROM watches WHERE user_id=? AND term=?",
                  (user["id"], body.term.strip().lower()))
    return {"ok": True}


@app.post("/api/watches/seen")
def watches_seen(request: Request):
    """Mark all watches as viewed (clears the 'new' badge)."""
    user = _require_paid(request)
    with _db() as c:
        rows = c.execute("SELECT term FROM watches WHERE user_id=?",
                         (user["id"],)).fetchall()
        for (term,) in rows:
            c.execute("UPDATE watches SET seen_n=? WHERE user_id=? "
                      "AND term=?",
                      (len(_watch_matches(term)), user["id"], term))
    return {"ok": True}


class KeyRequest(BaseModel):
    label: str = ""
    key: str = ""


def _month() -> str:
    return datetime.date.today().strftime("%Y-%m")


@app.get("/api/keys")
def keys_list(request: Request):
    user = _require_pro(request)
    with _db() as c:
        rows = c.execute(
            "SELECT k.key, k.label, k.created, "
            "COALESCE(u.count, 0) FROM api_keys k "
            "LEFT JOIN api_usage u ON u.key = k.key AND u.month = ? "
            "WHERE k.user_id=? AND k.revoked=0 ORDER BY k.created",
            (_month(), user["id"])).fetchall()
    return {"limit": API_MONTHLY,
            "keys": [{"key": r[0], "label": r[1],
                      "used_this_month": r[3]} for r in rows]}


@app.post("/api/keys/create")
def keys_create(body: KeyRequest, request: Request):
    user = _require_pro(request)
    with _db() as c:
        # Re-verify against the DB inside the transaction: the account
        # must hold a current, unexpired Pro pass to mint a key.
        row = c.execute("SELECT plan, paid_until FROM users WHERE id=?",
                        (user["id"],)).fetchone()
        if not row or row[0] != "pro" or (row[1] or 0) <= time.time():
            raise HTTPException(402, "An active Pro plan is required "
                                     "to create API keys.")
        n = c.execute("SELECT COUNT(*) FROM api_keys WHERE user_id=? "
                      "AND revoked=0", (user["id"],)).fetchone()[0]
        if n >= MAX_KEYS:
            raise HTTPException(400, f"Limit of {MAX_KEYS} active keys.")
        key = "ogk_" + uuid.uuid4().hex
        c.execute("INSERT INTO api_keys VALUES(?,?,?,?,0)",
                  (key, user["id"], body.label.strip()[:60] or "default",
                   time.time()))
    print(f"APIKEY_ISSUED user={user['id'][:8]}", flush=True)
    return {"key": key}


@app.post("/api/keys/revoke")
def keys_revoke(body: KeyRequest, request: Request):
    user = _require_pro(request)
    with _db() as c:
        c.execute("UPDATE api_keys SET revoked=1 WHERE key=? AND "
                  "user_id=?", (body.key.strip(), user["id"]))
    return {"ok": True}


class AskRequest(BaseModel):
    question: str
    lang: str | None = None


def _api_key_auth(request: Request, meter: bool = True) -> str:
    """Validate 'Authorization: Bearer ogk_...' and (optionally) count
    one call against the key's monthly allowance. Returns the key."""
    auth = request.headers.get("authorization", "")
    key = auth.removeprefix("Bearer ").strip()
    if not key.startswith("ogk_"):
        raise HTTPException(401, "Missing API key. Send it as "
                                 "'Authorization: Bearer ogk_...'.")
    with _db() as c:
        row = c.execute(
            "SELECT k.user_id, u.plan, u.paid_until FROM api_keys k "
            "JOIN users u ON u.id = k.user_id "
            "WHERE k.key=? AND k.revoked=0", (key,)).fetchone()
    if not row:
        raise HTTPException(401, "Unknown or revoked API key.")
    if row[1] != "pro" or (row[2] or 0) <= time.time():
        raise HTTPException(402, "The Pro plan on this account has "
                                 "expired. Renew at onegovai.app.")
    if meter:
        with _db() as c:
            used = c.execute("SELECT count FROM api_usage WHERE key=? AND "
                             "month=?", (key, _month())).fetchone()
            if used and used[0] >= API_MONTHLY:
                raise HTTPException(429, f"Monthly limit of {API_MONTHLY} "
                                         "API calls reached.")
            c.execute("INSERT INTO api_usage VALUES(?,?,1) "
                      "ON CONFLICT(key, month) "
                      "DO UPDATE SET count = count + 1", (key, _month()))
    return key


@app.post("/api/v1/ask")
def api_ask(body: AskRequest, request: Request):
    """Metered public API (Pro): one-shot question, JSON answer.
    Auth: 'Authorization: Bearer ogk_...'."""
    _api_key_auth(request)
    _check_rate(request)
    q = body.question.strip()
    if not q or len(q) > 8000:
        raise HTTPException(400, "question must be 1-8000 characters")
    agent = LawAgent(retriever=_retriever)
    prefix = LANG_PREFIX.get(body.lang or "", "")
    t0 = time.time()
    try:
        answer = agent.ask(prefix + q)
    except Exception as e:
        log_chat("api", q, error=str(e), ms=int((time.time() - t0) * 1000))
        code, msg = _friendly_error(e)
        raise HTTPException(code, msg)
    aid = _store_answer(q, answer, agent.searches)
    log_chat("api", q, answer=answer, searches=agent.searches,
             ms=int((time.time() - t0) * 1000))
    return {"answer": answer, "searches": agent.searches,
            "sources": getattr(agent, "sources", []),
            "share_url": f"{BASE_URL}/s/{aid}" if aid else None}


class SearchRequest(BaseModel):
    query: str
    top_k: int = 8                     # 1-20
    doc_type: str | None = None        # constitution|act|si|gn|gazette|policy
    year: int | None = None


@app.post("/api/v1/search")
def api_search(body: SearchRequest, request: Request):
    """Metered public API (Pro): raw hybrid retrieval over the corpus —
    the same search the agent uses, without the LLM. Returns matching
    provisions with citations and scores."""
    _api_key_auth(request)
    _check_rate(request)
    q = body.query.strip()
    if not q or len(q) > 1000:
        raise HTTPException(400, "query must be 1-1000 characters")
    k = max(1, min(body.top_k or 8, 20))
    try:
        results = _retriever.search(q, top_k=k, doc_type=body.doc_type,
                                    year=body.year)
    except Exception as e:
        print(f"API_SEARCH_FAIL {e}", flush=True)
        raise HTTPException(502, "Search failed. Please try again.")
    return {"results": [{
        "instrument": r.get("doc_title", ""),
        "citation": r.get("citation", ""),
        "doc_type": r.get("doc_type"),
        "section": r.get("section"),
        "section_title": r.get("section_title"),
        "text": r.get("text", "")[:2500],
        "score": r.get("score"),
        "download_url": (f"{BASE_URL}/api/download/{r['source_file']}"
                         if r.get("source_file") else None),
    } for r in results]}


@app.get("/api/v1/usage")
def api_usage(request: Request):
    """Free-of-charge usage check for an API key (does not count as a
    call)."""
    key = _api_key_auth(request, meter=False)
    with _db() as c:
        row = c.execute("SELECT count FROM api_usage WHERE key=? AND "
                        "month=?", (key, _month())).fetchone()
    used = row[0] if row else 0
    return {"month": _month(), "used": used, "limit": API_MONTHLY,
            "remaining": max(0, API_MONTHLY - used)}


# ---------------- Answer cache (first question of a session) --------
CACHE_TTL = int(os.getenv("CACHE_TTL", str(24 * 3600)))


def _cache_key(req: "ChatRequest") -> str:
    norm = " ".join(req.message.lower().split())
    return hashlib.sha256(
        ((req.lang or "en") + "|" + norm).encode()).hexdigest()


def _cache_get(key: str):
    try:
        with _db() as c:
            row = c.execute("SELECT answer, searches, ts FROM cache "
                            "WHERE key=?", (key,)).fetchone()
        if row and time.time() - row[2] < CACHE_TTL:
            return row[0], json.loads(row[1])
    except Exception:
        pass
    return None


def _cache_put(key: str, answer: str, searches: list) -> None:
    try:
        with _db() as c:
            c.execute("INSERT OR REPLACE INTO cache VALUES(?,?,?,?)",
                      (key, answer,
                       json.dumps(searches or [], ensure_ascii=False),
                       time.time()))
    except Exception:
        pass


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    lang: str | None = None            # 'en' | 'sn' | 'nd'


LANG_PREFIX = {
    "sn": "Answer in Shona (chiShona), keeping instrument names and "
          "section citations in English. ",
    "nd": "Answer in Ndebele (isiNdebele), keeping instrument names and "
          "section citations in English. ",
}


def _localise(req: ChatRequest) -> str:
    return LANG_PREFIX.get(req.lang or "", "") + req.message.strip()


LOG_PATH = DATA_DIR / "chat_log.jsonl"


def _friendly_error(e: Exception) -> tuple[int, str]:
    """Map provider errors to a message safe to show the public."""
    s = str(e)
    if ("429" in s or "RESOURCE_EXHAUSTED" in s.upper()
            or "quota" in s.lower() or "rate limit" in s.lower()):
        return 503, ("OneGov is receiving a lot of questions right now. "
                     "Please wait a minute and try again.")
    if "503" in s or "overloaded" in s.lower() or "UNAVAILABLE" in s:
        return 503, ("The answering service is briefly unavailable. "
                     "Please try again in a moment.")
    return 502, ("We hit a temporary problem answering this. "
                 "Please try again in a moment.")


def log_chat(sid: str, question: str, answer: str | None = None,
             error: str | None = None, searches: list | None = None,
             ms: int = 0) -> None:
    """One JSON line per Q&A: to stdout (visible in Azure Log Analytics)
    and to data/chat_log.jsonl locally."""
    entry = {"ts": datetime.datetime.now(datetime.UTC)
                     .isoformat(timespec="seconds"),
             "session": sid[:8],
             "question": question,
             "searches": searches or [],
             "answer_chars": len(answer) if answer else 0,
             "error": error,
             "ms": ms}
    line = json.dumps(entry, ensure_ascii=False)
    print("CHATLOG " + line, flush=True)
    try:
        LOG_PATH.parent.mkdir(exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


@app.get("/api/health")
def health():
    return {"ok": True}


def _get_agent(req: ChatRequest) -> tuple[str, LawAgent]:
    msg = req.message.strip()
    if not msg:
        raise HTTPException(400, "Empty message")
    if len(msg) > 8000:
        raise HTTPException(400, "Message too long (max 8000 chars)")

    sid = req.session_id or ""
    agent = _agents.get(sid)
    if agent is None and sid:                     # revive from SQLite
        stored = _load_session(sid)
        if stored is not None:
            agent = LawAgent(retriever=_retriever)
            agent.messages = stored
            if len(_agents) >= MAX_SESSIONS:
                _agents.pop(next(iter(_agents)))
            _agents[sid] = agent
    if agent is None:
        sid = uuid.uuid4().hex
        if len(_agents) >= MAX_SESSIONS:          # drop oldest session
            _agents.pop(next(iter(_agents)))
        agent = _agents[sid] = LawAgent(retriever=_retriever)

    # Bound context growth; keep the system message (openai-format only)
    if len(agent.messages) > MAX_HISTORY:
        agent.messages = [m for m in agent.messages[:1]
                          if m.get("role") == "system"]
    return sid, agent


def _seed_cached(sid: str, agent: LawAgent, question: str,
                 answer: str) -> None:
    """Give a cache-served chat real history so follow-ups have context."""
    if not agent.messages:
        agent.messages.append({"role": "system", "content": SYSTEM_PROMPT})
    agent.messages.append({"role": "user", "content": question})
    agent.messages.append({"role": "assistant", "content": answer})
    _save_session(sid, agent)


class FeedbackRequest(BaseModel):
    session_id: str | None = None
    question: str = ""
    verdict: str = ""                  # 'up' | 'down'


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    if req.verdict not in ("up", "down"):
        raise HTTPException(400, "verdict must be 'up' or 'down'")
    entry = {"ts": datetime.datetime.now(datetime.UTC)
                     .isoformat(timespec="seconds"),
             "session": (req.session_id or "")[:8],
             "verdict": req.verdict,
             "question": req.question[:500]}
    line = json.dumps(entry, ensure_ascii=False)
    print("FEEDBACK " + line, flush=True)
    try:
        path = DATA_DIR / "feedback.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    return {"ok": True}


@app.get("/api/catalog")
def catalog(doc_type: str | None = None, year: int | None = None):
    """Distinct instruments, newest first, with their source files —
    powers the Browse panel."""
    items = _retriever.catalog(doc_type=doc_type, year=year)
    files: dict[str, str] = {}
    for r in _retriever.records:
        files.setdefault(r["doc_title"], r.get("source_file", ""))
    return [{"title": i["title"], "doc_type": i["doc_type"],
             "year": i["year"],
             "file": _display_file(files.get(i["title"], ""))}
            for i in items]


@app.get("/api/bulk/{doc_type}/{year}")
def bulk_download(doc_type: str, year: int, request: Request):
    """Pro: every source document of a given type + year as one ZIP.
    Individual documents stay free for everyone; the paid part is the
    one-click bulk packaging. Rate limit + daily cap protect bandwidth."""
    user = _require_pro(request)
    _check_rate(request)
    bq = _bulk_hits[user["id"]]
    _purge(bq, 86400)
    if len(bq) >= BULK_DAILY:
        raise HTTPException(429, f"Bulk download limit reached "
                                 f"({BULK_DAILY} a day). Try again tomorrow.")
    bq.append(time.time())
    _bound(_bulk_hits)
    items = _retriever.catalog(doc_type=doc_type, year=year)
    files: dict[str, str] = {}
    for r in _retriever.records:
        files.setdefault(r["doc_title"], r.get("source_file", ""))
    names = sorted({_display_file(files.get(i["title"], ""))
                    for i in items} - {""})
    if not names:
        raise HTTPException(404, "No documents for that type and year")
    if len(names) > 300:
        raise HTTPException(400, "Too many documents — narrow the year")
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            if "/" in n or "\\" in n or n.startswith("."):
                continue
            p = RAW_DIR / n
            if not p.is_file():
                p = ORIGINALS_DIR / n
            if p.is_file():
                z.write(p, n)
    data = buf.getvalue()
    if not data:
        raise HTTPException(404, "Source files not available")
    fname = f"onegov-{doc_type}-{year}.zip"
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition":
                             f'attachment; filename="{fname}"'})


@app.get("/api/source")
def resolve_source(title: str):
    """Resolve an instrument title (as cited in an answer) to its source
    document, so the UI can offer it as a download."""
    title = normalize_refs(title)
    # filler words must not count as title overlap ("...Of The Lisbon...")
    _FILLER = {"of", "the", "and", "as", "at", "on", "in", "for", "to",
               "a", "an", "section"}
    q = set(_tokenize(title)) - _FILLER
    if not q:
        raise HTTPException(404, "No such document")
    q_years = {t for t in q if t.isdigit() and len(t) == 4}
    # exact instrument references like "SI 127 of 2021" must match exactly
    refs = {(int(n), y) for n, y in Q_INSTR_RE.findall(title)}
    best_file, best_title, best_score = None, None, 0.0
    seen: set[str] = set()
    for r in _retriever.records:
        t = r.get("doc_title", "")
        if t in seen:
            continue
        seen.add(t)
        tset = set(_tokenize(t)) - _FILLER
        if not tset:
            continue
        if refs:
            t_nums = {int(x) for x in tset if x.isdigit() and len(x) <= 4}
            if not any(n in t_nums and int(y) in t_nums for n, y in refs):
                continue
        overlap = len(q & tset)
        score = overlap + overlap / len(tset)
        # cited year must agree with the document's year
        t_years = {x for x in tset if x.isdigit() and len(x) == 4}
        if q_years:
            if q_years & t_years:
                score += 2
            elif t_years:
                score -= 2
        # "the Constitution" should hit the Constitution itself, not a
        # notice that merely mentions it
        if "constitution" in q and r.get("doc_type") == "constitution":
            score += 3
        # "the Electoral Act" means the principal (consolidated) Act, not
        # an Amendment Act, unless the citation says "amendment"
        if "amendment" in tset and "amendment" not in q:
            score -= 1.5
        if score > best_score:
            best_file = r.get("source_file")
            best_title, best_score = t, score
    if not best_file or best_score < 2:
        raise HTTPException(404, "No such document")
    best_file = _display_file(best_file)
    return {"title": best_title, "file": best_file,
            "url": f"/api/download/{best_file}"}


_CASELAW_CACHE: dict = {"mtime": 0.0, "items": []}
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_HELD_RE = re.compile(r"<p>Held:\s*(.*?)</p>", re.S)
_FLY_RE = re.compile(r"<p>Flynote[^:]*:\s*(.*?)</p>", re.S)


def _court_group(court: str) -> str:
    c = court.lower()
    if "constitutional" in c:
        return "Constitutional Court"
    if "supreme" in c:
        return "Supreme Court"
    if "high court" in c:
        return "High Court"
    if "labour" in c:
        return "Labour Court"
    if "magistr" in c:
        return "Magistrates Court"
    # JSC quirks: "Commercial Court" and "Mutare Court" are High Court
    # divisions
    if "commercial" in c or re.search(
            r"(harare|bulawayo|mutare|masvingo|chinhoyi)\s+court", c):
        return "High Court"
    return court or "Other"


def _norm_cite(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


_JSC_COURT_IDS = {
    "1": "Constitutional Court", "2": "Harare Supreme Court",
    "4": "Harare Labour Court", "5": "Administrative Court",
    "6": "Commercial Court", "7": "Sheriff of the High Court",
    "8": "Magistrate Court", "9": "Harare High Court",
}


def _caselaw_items() -> list[dict]:
    """Judgment entries from data/raw/_manifest.json, cached by mtime.
    Reads the scraper's manifest directly so new cases appear as soon as
    they are downloaded — before ingest/indexing."""
    mp = RAW_DIR / "_manifest.json"
    if not mp.is_file():
        return []
    mt = mp.stat().st_mtime
    if mt == _CASELAW_CACHE["mtime"]:
        return _CASELAW_CACHE["items"]
    try:
        manifest = json.loads(mp.read_text(encoding="utf-8"))
    except ValueError:
        return _CASELAW_CACHE["items"]
    items = []
    for fname, m in manifest.items():
        if m.get("doc_type") != "judgment":
            continue
        if m.get("duplicate_of"):   # same case kept under another source
            continue
        if not (RAW_DIR / fname).is_file():
            continue
        title = m.get("title", fname)
        # "Judgment HMA49-26: ..." — the prefix is the citation
        citation = ""
        cm = re.match(r"^Judgment\s+([\w/-]+):\s*", title)
        if cm:
            citation = cm.group(1)
            title = title[cm.end():]
        elif title.startswith("Judgment: "):   # veritas
            title = title[10:]
        if not citation:
            nm = re.search(r"\[(19|20)\d{2}\]\s+ZW[A-Z]{2,6}\s+\d+", title)
            if nm:
                citation = nm.group(0)
        date = m.get("date", "")
        if date and not re.match(r"\d{4}-\d{2}-\d{2}", date):
            try:   # ZimLII dates are "20 November 2026" — normalise so
                from datetime import datetime as _dt   # sorting works
                date = _dt.strptime(date.strip(),
                                    "%d %B %Y").strftime("%Y-%m-%d")
            except ValueError:
                pass
        ym = _YEAR_RE.search(date) or _YEAR_RE.search(title)
        original = m.get("original", "")
        summary = flynote = ""
        if fname.endswith(".html") and m.get("source") == "zimlii.org":
            try:  # stub header holds Held + flynote — cheap preview
                head = (RAW_DIR / fname).open(
                    encoding="utf-8", errors="replace").read(4096)
                hm = _HELD_RE.search(head)
                fm = _FLY_RE.search(head)
                summary = re.sub(r"\s+", " ", hm.group(1)) if hm else ""
                flynote = re.sub(r"\s+", " ", fm.group(1)) if fm else ""
            except OSError:
                pass
        court = m.get("court", "")
        if court in _JSC_COURT_IDS:  # JSC numeric court ids (running
            fixed = _JSC_COURT_IDS[court]  # scrapers rewrite old entries)
            title = title.replace(f"({court},", f"({fixed},")
            court = fixed
        items.append({
            "title": title,
            "citation": citation,
            "court": court,
            "group": _court_group(court),
            "judge": m.get("judge", ""),
            "date": date,
            "year": int(ym.group(0)) if ym else None,
            "case_number": m.get("case_number", ""),
            "source": m.get("source", ""),
            "full_text": m.get("full_text", ""),
            "summary": summary[:400],
            "flynote": flynote[:400],
            "file": fname,
            "url": f"/api/download/{fname}",
            "original_url": (f"/api/download/{original}"
                             if original else ""),
        })
    # newest first; undated last
    items.sort(key=lambda i: (i["year"] or 0, i["date"]), reverse=True)
    _CASELAW_CACHE.update(mtime=mt, items=items)
    return items


@app.get("/api/caselaw")
def caselaw(q: str | None = None, court: str | None = None,
            group: str | None = None, year: int | None = None,
            source: str | None = None,
            page: int = 1, page_size: int = 25):
    """Browse court judgments. Free, no key. Filters: q (title/parties/
    case number/citation), court, group (e.g. 'High Court'), year,
    source. Paginated."""
    items = _caselaw_items()
    facets = {
        "courts": sorted({i["court"] for i in items if i["court"]}),
        "groups": sorted({i["group"] for i in items if i["group"]}),
        "years": sorted({i["year"] for i in items if i["year"]},
                        reverse=True),
        "sources": sorted({i["source"] for i in items if i["source"]}),
    }
    if court:
        items = [i for i in items if i["court"] == court]
    if group:
        items = [i for i in items if i["group"] == group]
    if year:
        items = [i for i in items if i["year"] == year]
    if source:
        items = [i for i in items if i["source"] == source]
    if q:
        toks = [t for t in _tokenize(q) if t]
        nq = _norm_cite(q)
        cite_q = len(nq) >= 5 and any(c.isdigit() for c in nq)

        def match(i):
            hay = " ".join((i["title"], i["citation"], i["case_number"],
                            i["judge"])).lower()
            if toks and all(t in hay for t in toks):
                return True
            # citation forms: "HH 179/26", "[2026] ZWHHC 179",
            # "HCH 5667/24" all match once punctuation is stripped
            return cite_q and nq in _norm_cite(hay)
        items = [i for i in items if match(i)]
    page = max(1, page)
    page_size = min(max(1, page_size), 100)
    start = (page - 1) * page_size
    return {"total": len(items), "page": page, "page_size": page_size,
            "facets": facets,
            "items": items[start:start + page_size]}


@app.get("/api/download/{filename}")
def download(filename: str):
    """Serve a raw source document (PDF/HTML) from the corpus."""
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(404, "Not found")
    path = RAW_DIR / filename
    if not path.is_file():
        path = ORIGINALS_DIR / filename   # original-PDF companions
    if not path.is_file():
        raise HTTPException(404, "Not found")
    # inline: browsers render the PDF in-page instead of downloading it
    return FileResponse(path, filename=filename,
                        content_disposition_type="inline")


_QUOTA_HINT = re.compile(r"429|529|quota|rate.?limit|credit|insufficient|"
                         r"exhaust|overload", re.I)


def _is_pro(request: Request) -> bool:
    user = _user_from(request)
    return bool(user and user["plan"] == "pro"
                and user["paid_until"] > time.time())


def _agent_ask(agent: LawAgent, req: ChatRequest, pro: bool) -> str:
    """Pro perks: deeper research + one priority retry on quota errors."""
    agent.deep = pro
    try:
        return agent.ask(_localise(req))
    except Exception as e:
        if pro and _QUOTA_HINT.search(str(e)):
            time.sleep(3)
            return agent.retry_after_quota(_localise(req))
        raise


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    _check_rate(request)
    _check_gate(request)
    sid, agent = _get_agent(req)
    ckey = _cache_key(req) if not req.session_id else None
    if ckey and (hit := _cache_get(ckey)):
        answer, searches = hit
        _seed_cached(sid, agent, req.message.strip(), answer)
        aid = _store_answer(req.message.strip(), answer, searches)
        log_chat(sid, req.message.strip(), answer=answer,
                 searches=searches, ms=0)
        return {"session_id": sid, "answer": answer,
                "searches": searches, "answer_id": aid}
    t0 = time.time()
    pro = _is_pro(request)
    try:
        answer = _agent_ask(agent, req, pro)
    except Exception as e:
        log_chat(sid, req.message.strip(), error=str(e),
                 ms=int((time.time() - t0) * 1000))
        code, msg = _friendly_error(e)
        raise HTTPException(code, msg)
    if ckey:
        _cache_put(ckey, answer, agent.searches)
    _save_session(sid, agent)
    aid = _store_answer(req.message.strip(), answer, agent.searches)
    log_chat(sid, req.message.strip(), answer=answer,
             searches=agent.searches, ms=int((time.time() - t0) * 1000))
    return {"session_id": sid, "answer": answer,
            "searches": agent.searches, "answer_id": aid,
            "sources": getattr(agent, "sources", [])}


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest, request: Request):
    """SSE: emits {'type':'search'} events live, then a final 'done'."""
    _check_rate(request)
    _check_gate(request)
    sid, agent = _get_agent(req)

    ckey = _cache_key(req) if not req.session_id else None
    if ckey and (hit := _cache_get(ckey)):
        answer, searches = hit
        _seed_cached(sid, agent, req.message.strip(), answer)
        aid = _store_answer(req.message.strip(), answer, searches)
        log_chat(sid, req.message.strip(), answer=answer,
                 searches=searches, ms=0)

        def cached_gen():
            for s in searches:
                yield "data: " + json.dumps(
                    {"type": "search", "query": s}) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "done", "answer": answer, "session_id": sid,
                 "searches": searches, "answer_id": aid}) + "\n\n"
        return StreamingResponse(cached_gen(),
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})
    q: queue.Queue = queue.Queue()
    agent.on_event = lambda label: q.put(("search", label))
    agent.on_delta = lambda text: q.put(("delta", text))
    pro = _is_pro(request)

    def work():
        t0 = time.time()
        msg = req.message.strip()
        try:
            answer = _agent_ask(agent, req, pro)
            if ckey:
                _cache_put(ckey, answer, agent.searches)
            _save_session(sid, agent)
            aid = _store_answer(msg, answer, agent.searches)
            log_chat(sid, msg, answer=answer, searches=agent.searches,
                     ms=int((time.time() - t0) * 1000))
            q.put(("done", (answer, aid)))
        except Exception as e:
            log_chat(sid, msg, error=str(e),
                     ms=int((time.time() - t0) * 1000))
            q.put(("error", _friendly_error(e)[1]))

    threading.Thread(target=work, daemon=True).start()

    def gen():
        while True:
            kind, payload = q.get()
            if kind == "search":
                yield "data: " + json.dumps(
                    {"type": "search", "query": payload}) + "\n\n"
            elif kind == "delta":
                yield "data: " + json.dumps(
                    {"type": "delta", "text": payload}) + "\n\n"
            elif kind == "done":
                answer, aid = payload
                yield "data: " + json.dumps(
                    {"type": "done", "answer": answer, "session_id": sid,
                     "searches": agent.searches,
                     "answer_id": aid,
                     "sources": getattr(agent, "sources", [])}) + "\n\n"
                return
            else:
                yield "data: " + json.dumps(
                    {"type": "error", "error": payload}) + "\n\n"
                return

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/meta")
def meta():
    """Corpus freshness info for the UI."""
    chunks = PROCESSED_DIR / "chunks.jsonl"
    updated = (datetime.date.fromtimestamp(chunks.stat().st_mtime)
               .isoformat() if chunks.is_file() else None)
    titles = {r.get("doc_title") for r in _retriever.records}
    return {"updated": updated, "instruments": len(titles),
            "chunks": len(_retriever.records)}


_INSTR_NUM_RE = re.compile(r"(\d{1,4})\s+of\s+(\d{4})", re.I)


@app.get("/api/whatsnew")
def whatsnew(n: int = 12):
    """Most recently gazetted instruments, ordered by (year, instrument
    number) parsed from the title — powers 'Recently gazetted' sections."""
    files: dict[str, str] = {}
    for r in _retriever.records:
        files.setdefault(r["doc_title"], r.get("source_file", ""))

    def key(item):
        m = _INSTR_NUM_RE.search(item["title"])
        if m:
            return (int(m.group(2)), int(m.group(1)))
        return (item["year"] or 0, 0)

    items = sorted(_retriever.catalog(), key=key, reverse=True)
    return [{"title": i["title"], "doc_type": i["doc_type"],
             "year": i["year"],
             "file": _display_file(files.get(i["title"], ""))}
            for i in items[:max(1, min(n, 50))]]


@app.get("/api/answer/{aid}")
def get_answer(aid: str):
    with _db() as c:
        row = c.execute("SELECT question, answer, searches, ts FROM "
                        "answers WHERE id=?", (aid,)).fetchone()
    if not row:
        raise HTTPException(404, "No such answer")
    return {"question": row[0], "answer": row[1],
            "searches": json.loads(row[2]), "ts": row[3]}


SHARE_PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} | OneGov AI</title>
<meta property="og:title" content="{title}">
<meta property="og:description" content="Answer from OneGov AI, with exact section references from Zimbabwe's laws.">
<link rel="icon" href="/icon.svg" type="image/svg+xml">
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.1.0/purify.min.js"></script>
<style>
 body{{font-family:Inter,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
  background:#f9f9fb;color:#18181b;margin:0;line-height:1.7}}
 .wrap{{max-width:720px;margin:0 auto;padding:34px 20px 60px}}
 .brand{{display:flex;align-items:center;gap:10px;margin-bottom:26px;
  font-weight:700;font-size:16px}}
 .mark{{width:34px;height:34px;border-radius:10px;color:#fff;display:flex;
  align-items:center;justify-content:center;font-size:12px;font-weight:800;
  background:linear-gradient(135deg,#4f46e5,#8b5cf6)}}
 .q{{background:#f1f1f4;border-radius:16px;padding:13px 18px;font-size:15px;
  font-weight:600;margin-bottom:22px}}
 .a{{font-size:14.5px}}
 .a blockquote{{border-left:3px solid #c7c9f4;background:#eef0fe;
  margin:0 0 13px;padding:10px 15px;border-radius:0 10px 10px 0}}
 .cta{{display:inline-block;margin-top:30px;padding:11px 20px;color:#fff;
  border-radius:12px;text-decoration:none;font-weight:600;font-size:14px;
  background:linear-gradient(135deg,#4f46e5,#8b5cf6)}}
 .foot{{color:#71717a;font-size:12px;margin-top:26px}}
</style></head><body><div class="wrap">
<div class="brand"><div class="mark">ZW</div>OneGov AI</div>
<div class="q" id="q"></div><div class="a" id="a"></div>
<a class="cta" href="/app.html">Ask your own question →</a>
<div class="foot">OneGov provides information from Zimbabwe's Constitution,
Acts of Parliament, statutory instruments, general notices, Government
Gazettes and national policies. It is general information, not legal
advice. <a href="/privacy.html">Privacy policy</a>.</div></div>
<script>
const D={data};
document.getElementById('q').textContent=D.question;
document.getElementById('a').innerHTML=
  DOMPurify.sanitize(marked.parse(D.answer));
document.title=D.question.slice(0,80)+' | OneGov AI';
</script></body></html>"""


@app.get("/s/{aid}")
def share_page(aid: str):
    """Public shareable page for one Q&A."""
    with _db() as c:
        row = c.execute("SELECT question, answer FROM answers WHERE id=?",
                        (aid,)).fetchone()
    if not row:
        raise HTTPException(404, "No such answer")
    title = (row[0][:80]).replace('"', "'")
    data = json.dumps({"question": row[0], "answer": row[1]},
                      ensure_ascii=False).replace("</", "<\\/")
    return HTMLResponse(SHARE_PAGE.format(title=title, data=data))


# ---------------- Admin (token-protected) ----------------
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


def _admin(token: str) -> None:
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(403, "Forbidden")


def _tail_jsonl(path: Path, n: int) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines()[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _user_stats() -> dict:
    """Account/revenue stats straight from the DB (persists across deploys,
    unlike the tailed chat log)."""
    now = time.time()
    month = datetime.date.today().strftime("%Y-%m")
    today = datetime.date.today().isoformat()
    with _db() as c:
        total = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        active = c.execute(
            "SELECT plan, COUNT(*) FROM users "
            "WHERE paid_until > ? GROUP BY plan", (now,)).fetchall()
        plans = {"free": total, "plus": 0, "pro": 0}
        for plan, n in active:
            plan = plan if plan in ("plus", "pro") else "plus"
            plans[plan] += n
            plans["free"] -= n
        signups_7d = c.execute(
            "SELECT COUNT(*) FROM users WHERE created >= ?",
            (datetime.date.fromtimestamp(now - 7 * 86400).isoformat(),)
        ).fetchone()[0]
        signups_30d = c.execute(
            "SELECT COUNT(*) FROM users WHERE created >= ?",
            (datetime.date.fromtimestamp(now - 30 * 86400).isoformat(),)
        ).fetchone()[0]
        pays = c.execute(
            "SELECT COUNT(*), COALESCE(SUM(amount), 0) FROM payments "
            "WHERE status = 'paid'").fetchone()
        pays_30d = c.execute(
            "SELECT COUNT(*), COALESCE(SUM(amount), 0) FROM payments "
            "WHERE status = 'paid' AND created >= ?",
            (now - 30 * 86400,)).fetchone()
        q_today = c.execute(
            "SELECT COALESCE(SUM(count), 0) FROM user_usage WHERE day = ?",
            (today,)).fetchone()[0]
        anon_total = c.execute(
            "SELECT COALESCE(SUM(count), 0) FROM anon_usage").fetchone()[0]
        api_month = c.execute(
            "SELECT COALESCE(SUM(count), 0) FROM api_usage WHERE month = ?",
            (month,)).fetchone()[0]
        keys_live = c.execute(
            "SELECT COUNT(*) FROM api_keys WHERE revoked = 0").fetchone()[0]
        watches = c.execute("SELECT COUNT(*) FROM watches").fetchone()[0]

        def _traffic(days: int) -> dict:
            since = datetime.date.fromtimestamp(
                now - days * 86400).isoformat()
            u, v = c.execute(
                "SELECT COUNT(DISTINCT ip_hash), COALESCE(SUM(n), 0) "
                "FROM visits WHERE day >= ?", (since,)).fetchone()
            return {"visitors": u, "pageviews": v}

        traffic = {"today": _traffic(0), "last_7d": _traffic(7),
                   "last_30d": _traffic(30)}
    return {
        "traffic": traffic,
        "users_total": total,
        "plans": plans,
        "signups": {"last_7d": signups_7d, "last_30d": signups_30d},
        "revenue_usd": {"all_time": round(pays[1], 2), "count": pays[0],
                        "last_30d": round(pays_30d[1], 2),
                        "count_30d": pays_30d[0]},
        "questions_today_logged_in": q_today,
        "questions_anonymous_total": anon_total,
        "api": {"calls_this_month": api_month, "active_keys": keys_live},
        "watches_total": watches,
    }


@app.get("/api/admin/stats")
def admin_stats(token: str = ""):
    _admin(token)
    chats = _tail_jsonl(LOG_PATH, 2000)
    fb = _tail_jsonl(LOG_PATH.parent / "feedback.jsonl", 2000)
    errors = [c for c in chats if c.get("error")]
    answered = [c for c in chats if not c.get("error")]
    top = Counter(" ".join(c.get("question", "").lower().split())
                  for c in chats).most_common(20)
    return {
        "users": _user_stats(),
        "chats": len(chats),
        "errors": len(errors),
        "avg_ms": (sum(c.get("ms", 0) for c in answered)
                   // max(len(answered), 1)),
        "feedback": {"up": sum(1 for f in fb if f.get("verdict") == "up"),
                     "down": sum(1 for f in fb
                                 if f.get("verdict") == "down")},
        "top_questions": [{"q": q, "n": n} for q, n in top],
        "recent_errors": errors[-10:],
        "recent_feedback": fb[-10:],
    }


@app.get("/api/admin/log")
def admin_log(token: str = "", n: int = 200):
    _admin(token)
    return _tail_jsonl(LOG_PATH, min(n, 1000))


# Daily DB snapshots (kept alongside the DB, so they persist whenever
# DATA_DIR is on a mounted volume). scripts/backup_db.sh can additionally
# push them off-machine.
BACKUP_DIR = DATA_DIR / "backups"
BACKUP_KEEP = 14


def _snapshot_db() -> Path:
    """Consistent SQLite snapshot (safe against concurrent writes)."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    out = BACKUP_DIR / f"onegov-{datetime.date.today().isoformat()}.db"
    src = sqlite3.connect(DB_PATH)
    dst = sqlite3.connect(out)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()
    old = sorted(BACKUP_DIR.glob("onegov-*.db"))[:-BACKUP_KEEP]
    for p in old:
        p.unlink(missing_ok=True)
    return out


def _backup_loop() -> None:
    while True:
        try:
            out = BACKUP_DIR / (f"onegov-{datetime.date.today().isoformat()}"
                                ".db")
            if not out.exists():
                _snapshot_db()
                print(f"DB_BACKUP {out.name}", flush=True)
        except Exception as e:
            print(f"DB_BACKUP_FAIL {e}", flush=True)
        time.sleep(6 * 3600)            # check 4x a day


threading.Thread(target=_backup_loop, daemon=True).start()


@app.get("/api/admin/backup")
def admin_backup(token: str = ""):
    """Download a fresh snapshot of the whole database."""
    _admin(token)
    try:
        out = _snapshot_db()
    except Exception as e:
        raise HTTPException(500, f"Backup failed: {e}")
    return FileResponse(out, filename=out.name,
                        media_type="application/octet-stream")


class PlanChange(BaseModel):
    token: str = ""
    email: str = ""
    plan: str = "plus"                  # 'free' | 'plus' | 'pro'


@app.post("/api/admin/set-plan")
def admin_set_plan(body: PlanChange):
    """Manual plan activation (fallback to the Paynow flow)."""
    _admin(body.token)
    plan = "plus" if body.plan == "paid" else body.plan   # legacy alias
    if plan not in ("free", "plus", "pro"):
        raise HTTPException(400, "plan must be 'free', 'plus' or 'pro'")
    until = time.time() + PAID_DAYS * 86400 if plan != "free" else 0
    with _db() as c:
        cur = c.execute("UPDATE users SET plan=?, paid_until=? "
                        "WHERE email=?",
                        (plan, until, body.email.strip().lower()))
    if cur.rowcount == 0:
        raise HTTPException(404, "No user with that email")
    return {"email": body.email.strip().lower(), "plan": plan}


@app.get("/api/admin/users")
def admin_users(token: str = "", n: int = 200):
    _admin(token)
    with _db() as c:
        rows = c.execute("SELECT email, plan, created FROM users "
                         "ORDER BY created DESC LIMIT ?",
                         (min(n, 1000),)).fetchall()
    return [{"email": r[0], "plan": r[1], "created": r[2]} for r in rows]


# Scraper credentials settable from the admin page. Whitelisted keys only —
# never allow arbitrary env writes (ADMIN_TOKEN, API keys stay file-only).
SCRAPER_ENV_KEYS = ("PARLZIM_COOKIE", "PARLZIM_UA",
                    "VERITAS_COOKIE", "VERITAS_UA",
                    "LAWSAFRICA_TOKEN")
ENV_PATH = Path(__file__).parent / ".env"
_env_lock = threading.Lock()


class ScraperCreds(BaseModel):
    token: str = ""
    values: dict[str, str] = {}


def _write_env(updates: dict[str, str]) -> None:
    """Update keys in .env in place, preserving everything else."""
    with _env_lock:
        lines = (ENV_PATH.read_text(encoding="utf-8").splitlines()
                 if ENV_PATH.exists() else [])
        done = set()
        for i, line in enumerate(lines):
            m = re.match(r"\s*([A-Z_][A-Z0-9_]*)\s*=", line)
            if m and m.group(1) in updates:
                k = m.group(1)
                lines[i] = f"{k}={json.dumps(updates[k])}"
                done.add(k)
        for k, v in updates.items():
            if k not in done:
                lines.append(f"{k}={json.dumps(v)}")
        ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.get("/api/admin/scraper-creds")
def scraper_creds_status(token: str = ""):
    _admin(token)
    return {k: bool(os.getenv(k)) for k in SCRAPER_ENV_KEYS}


@app.post("/api/admin/scraper-creds")
def scraper_creds_set(body: ScraperCreds):
    _admin(body.token)
    updates = {k: v.strip() for k, v in body.values.items()
               if k in SCRAPER_ENV_KEYS and v.strip()}
    if not updates:
        raise HTTPException(400, "No valid keys given")
    _write_env(updates)
    os.environ.update(updates)  # scrapers re-read .env; this aids this proc
    return {"saved": sorted(updates)}


# ---------------- Visitor tracking ----------------
# Counts page loads (not API calls) per day per anonymised visitor.
# IPs are salted+hashed and the hash is only ever used for uniqueness,
# so no raw addresses are stored.
_VISIT_SALT = os.getenv("VISIT_SALT", "onegov-visits")
_PAGE_RE = re.compile(r"^(/|/[^/]+\.html|/s/[A-Za-z0-9_-]+)$")


def _record_visit(day: str, ip_hash: str) -> None:
    try:
        with _db() as c:
            c.execute("INSERT INTO visits(day, ip_hash, n) VALUES(?,?,1) "
                      "ON CONFLICT(day, ip_hash) DO UPDATE SET n = n + 1",
                      (day, ip_hash))
    except sqlite3.Error:
        pass                                    # never break a page load


@app.middleware("http")
async def _track_visits(request: Request, call_next):
    response = await call_next(request)
    if (request.method == "GET" and response.status_code == 200
            and _PAGE_RE.match(request.url.path)):
        ip_hash = hashlib.sha256(
            (_VISIT_SALT + _client_ip(request)).encode()).hexdigest()[:16]
        _record_visit(datetime.date.today().isoformat(), ip_hash)
    return response


# Serve the chat UI (must be mounted last)
app.mount("/", StaticFiles(directory=Path(__file__).parent / "web",
                           html=True), name="web")
