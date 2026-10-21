"""The pinned verifier reads French listings: the spot check that motivated stop 75."""

from __future__ import annotations

from legal_rag_verifier.premise import Premise
from legal_rag_verifier.verifier import Verdict, Verifier

LISTING = "Prix : 279,99 € — Autonomie : 30 heures — Garantie : 2 ans — livré le 5 novembre 2026"


def test_a_french_listing_grounds_its_english_twin() -> None:
    answer = (
        "It costs €279.99, lasts 30 hours, has a 2-year warranty and arrives on 5 November 2026."
    )
    verdict = Verifier().check_sentence(Premise.from_text(LISTING), answer)
    assert verdict.verdict is Verdict.EMIT


def test_a_wrong_price_against_a_french_listing_is_caught() -> None:
    verdict = Verifier().check_sentence(Premise.from_text(LISTING), "It costs €289.99.")
    assert verdict.verdict is Verdict.ROLLBACK
