"""Shared test env: server config is read at import time, so these must
be set before any test module imports `server`. Raises the free-question
gates so validation tests never trip them mid-suite."""
import os
import tempfile

os.environ.setdefault("ADMIN_TOKEN", "test-token")
os.environ.setdefault("RATE_LIMIT", "1000")
os.environ.setdefault("ANON_IP_DAILY", "10000")
os.environ.setdefault("SIGNUP_IP_DAILY", "10000")
os.environ.setdefault("FREE_QUESTIONS", "10000")
os.environ.setdefault("DAILY_FREE", "10000")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="onegov-test-"))
os.environ["PAYNOW_ID"] = ""
os.environ["PAYNOW_KEY"] = ""
os.environ["SMTP_HOST"] = ""
