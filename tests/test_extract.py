"""Getting text out of the formats Veritas publishes judgments in.

The older Supreme Court and Constitutional Court judgments are Word
files, and until these were handled 23 of them — Zuva Petroleum among
them — were dropped on the floor at ingest with a one-line warning.
"""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.extract import extract_text                  # noqa: E402

JUDGMENT = "Nyamande & Anor v Zuva Petroleum (Pvt) Ltd SC 43/15"
HAS_SOFFICE = bool(shutil.which("soffice") or shutil.which("libreoffice"))


@pytest.fixture
def docx_bytes(tmp_path):
    docx = pytest.importorskip("docx")
    p = tmp_path / "src.docx"
    d = docx.Document()
    d.add_paragraph(JUDGMENT)
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text = "Order"
    t.rows[0].cells[1].text = "Appeal dismissed with costs."
    d.save(str(p))
    return p.read_bytes()


def test_docx_text_is_extracted(tmp_path, docx_bytes):
    p = tmp_path / "judgment.docx"
    p.write_bytes(docx_bytes)
    assert JUDGMENT in extract_text(p)


def test_a_table_is_not_dropped(tmp_path, docx_bytes):
    """Judgments put the order — the part that decides the case — in a
    table often enough that skipping tables loses the holding."""
    p = tmp_path / "judgment.docx"
    p.write_bytes(docx_bytes)
    assert "Appeal dismissed with costs." in extract_text(p)


def test_a_docx_named_doc_is_still_read(tmp_path, docx_bytes):
    """Several of these files are OOXML or RTF behind a ".doc" name, so
    dispatch is on the leading bytes. Trusting the extension sends them
    to LibreOffice as binary Word and loses the judgment."""
    p = tmp_path / "judgment.doc"
    p.write_bytes(docx_bytes)
    assert JUDGMENT in extract_text(p)


@pytest.mark.skipif(not HAS_SOFFICE, reason="LibreOffice not installed")
def test_rtf_named_doc_is_read(tmp_path):
    p = tmp_path / "judgment.doc"
    p.write_bytes(rb"{\rtf1\ansi " + JUDGMENT.encode() + rb"\par }")
    assert "Zuva Petroleum" in extract_text(p)


def test_an_unknown_format_still_raises(tmp_path):
    p = tmp_path / "judgment.xyz"
    p.write_text("whatever")
    with pytest.raises(ValueError):
        extract_text(p)


def test_a_missing_libreoffice_is_reported_clearly(tmp_path, monkeypatch):
    """Ingest keeps going past one failed file, so the message is the
    only thing that says the corpus is short a judgment."""
    monkeypatch.setattr(shutil, "which", lambda _: None)
    p = tmp_path / "judgment.doc"
    p.write_bytes(b"\xd0\xcf\x11\xe0" + b"\x00" * 16)   # OLE2 header
    with pytest.raises(RuntimeError, match="LibreOffice"):
        extract_text(p)
