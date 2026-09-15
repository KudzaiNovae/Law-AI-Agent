"""What gets kept out of the index.

A register of named individuals is both useless to answer from and
personal data we should not be serving, so it is dropped at ingest. The
risk in any such rule is that it eats real law instead, which is what
these tests guard.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RAW_DIR                                # noqa: E402
from ingest.run import (                                  # noqa: E402
    DIRECTORY_FILES, is_personal_register, infer_meta)


def _rows(n, template):
    return "\n".join(template.format(i=i) for i in range(n))


# Gazette CIV/45: a deceased-voters roll, and pdfplumber reads its
# table right-to-left, so the names come out reversed too.
REGISTER = _rows(400, "{i} D-729{i:04d}-31 tsenoH ebuD 1401 aerA gnimraF M")
# The same rows as they read when the table is the right way round.
REGISTER_LTR = _rows(400, "{i} Honest Dube 63-172{i:04d}-K 12/3/1949 M")


def test_a_roll_of_named_individuals_is_a_register():
    assert is_personal_register(REGISTER)


def test_the_rule_does_not_depend_on_the_text_reading_forwards():
    """These tables are extracted reversed as often as not, so a rule
    that only recognised readable names would miss the worst file."""
    assert is_personal_register(REGISTER_LTR)


def test_an_act_is_not_a_register():
    act = _rows(400, "{i}. Interpretation. In this Act, unless the "
                     "context otherwise requires, 'employee' means ...")
    assert not is_personal_register(act)


def test_a_wage_table_is_not_a_register():
    """Collective bargaining agreements are lists of short rows with
    numbers in them, which is the shape a register also has — and they
    carry the minimum wage, which people actually ask about."""
    cba = _rows(400, "Grade {i} Nurse Aide ... 350,00 per month")
    assert not is_personal_register(cba)


def test_a_judgment_naming_a_party_s_id_survives():
    """A handful of ID numbers is ordinary in a judgment or a deed
    transfer. Only a bulk roll is a register."""
    judgment = ("The applicant, Farai Moyo (ID 63-544855 T 63), sought "
                "an order ... " + "The court considered the papers. " * 200)
    assert not is_personal_register(judgment)


def test_an_election_petition_annexing_voter_ids_survives():
    """The Constitutional Court presidential election petition annexes
    voter affidavits: 112 ID numbers across 124 000 characters. A rule
    that only counted IDs dropped it, which is how a landmark judgment
    went missing. Density is what tells the two apart — the registers
    run an ID every ~150 characters, this one every ~1 100.
    """
    petition = ""
    for i in range(112):
        petition += (f"AND the deponent {i}, whose national registration "
                     f"number is 63-{i:06d}-K, states that he attended the "
                     f"polling station and was turned away, and the Court "
                     f"has considered the affidavit in its entirety. "
                     + "The applicant further contends as follows. " * 12)
    assert len(petition) / 112 > 500          # the shape of the real file
    assert not is_personal_register(petition)


def test_a_dense_roll_is_still_a_register_at_any_length():
    """The margin cuts both ways: density has to catch a short roll too,
    not just the 6 000-row Gazette issues."""
    short_roll = _rows(150, "{i} Honest Dube 63-172{i:04d}-K 12/3/1949 M")
    assert is_personal_register(short_roll)


def test_every_named_directory_still_exists():
    """The list excludes by exact filename, so a re-scrape that renames
    one would quietly put 5 000 rows of names back in the index. Fail
    here instead, where the cause is obvious."""
    missing = [n for n in DIRECTORY_FILES if not (RAW_DIR / n).exists()]
    if missing and not any(RAW_DIR.iterdir()):
        pytest.skip("no raw corpus checked out")
    assert not missing


# ---------------- doc_type / title inference ----------------

@pytest.mark.parametrize("name, doc_type", [
    ("gn_gn_2021_1178_appointment_of_senator.pdf", "gn"),
    ("si_si_2024_110_finance_amendment.pdf", "si"),
    ("act_2002_12_value_added_tax_act.html", "act"),
    ("judgment_jsc_hh_37_14.pdf", "judgment"),
])
def test_scraper_prefix_decides_the_type(name, doc_type):
    assert infer_meta(Path(name))["doc_type"] == doc_type


def test_the_doubled_scraper_prefix_still_yields_a_clean_title():
    """The scrapers write "gn_" in front of a name that already starts
    with "gn_", and only one prefix is stripped — the rest has to be
    rewritten into a citable number rather than left as "Gn 2021 1178"."""
    title = infer_meta(Path("gn_gn_2021_1178_appointment_of_senator.pdf"))
    assert title["title"].startswith("GN 2021-1178")
