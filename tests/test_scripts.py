"""The scripts' own logic, without the network, the router or data/."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from product_answer_checks import battery as bt
from scripts import answer, check, judge, router

ROOT = Path(__file__).resolve().parent.parent


def test_a_checked_answer_keeps_only_emitted_sentences() -> None:
    out = check.checked({"qid": "A01-fr"}, ["One.", "Two.", "Three."], ["EMIT", "ROLLBACK", "EMIT"])
    assert out["text"] == "One. Three."
    assert out["verdicts"] == ["EMIT", "ROLLBACK", "EMIT"]


def test_the_answer_prompt_is_the_sealed_prompt_and_the_question() -> None:
    q = bt.Question("A01-fr", 1, "A", "fr", "Where?", [1], [], [], {}, premise="LISTING")
    system, user = answer.messages(q)
    assert system["content"] == bt.PROMPT.format(premise="LISTING")
    assert user == {"role": "user", "content": "Where?"}


def test_opus_55_is_capped_at_the_spend_go() -> None:
    assert router.CAPS[router.OPUS55] == 10.0
    assert set(answer.MODELS.values()) == {router.GEMINI, router.OPUS46}


def test_the_judge_reads_a_label_and_maps_it_to_passages() -> None:
    assert judge.parse_reply('{"label": "Supported", "passage": 2}')["label"] == "supported"
    assert [p.entailment for p in judge.one_hot("supported", 2, 3)] == [0.0, 1.0, 0.0]


def test_the_claim_check_runner_reads_a_french_listing() -> None:
    row = {
        "qid": "A01-fr",
        "premise": "Prix : 279,99 €",
        "query": "Where?",
        "sentences": ["It costs €279.99.", "It costs €289.99."],
    }
    out = subprocess.run(  # noqa: S603
        [sys.executable, str(ROOT / "scripts/claimcheck.py")],
        input=json.dumps(row) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(out.stdout)["verdicts"] == ["EMIT", "ROLLBACK"]


def test_the_draw_is_read_up_to_its_first_gap() -> None:
    from scripts import fetch_icecat  # noqa: PLC0415

    rows = {n: {"n": n, "id": n, "status": "category"} for n in (0, 1, 2, 4)}
    prefix, selection = fetch_icecat.stopping_point(rows)
    assert sorted(prefix) == [0, 1, 2]
    assert not selection.complete


def test_the_claim_check_is_the_released_build() -> None:
    from importlib.metadata import version  # noqa: PLC0415

    assert version("legal-rag-verifier") == "0.3.0"
    assert check.PINNED == "66450a6"


def test_the_report_counts_wrong_kept_and_right_removed() -> None:
    from scripts import report  # noqa: PLC0415

    rows = [("A01-fr", ["Right.", "Wrong.", "Odd."], [False, True, True], 0.5)]
    grades = {("A01-fr", "Right."): "yes", ("A01-fr", "Wrong."): "no"}
    out = report.tally(rows, grades)
    assert (out["right_removed"], out["wrong_kept"], out["unsettled_kept"], out["empty"]) == (
        1,
        1,
        1,
        0,
    )
    assert report.is_price("It costs €279.99.")
    assert not report.is_price("It weighs 2 kg.")


def test_the_reading_rule() -> None:
    from scripts import report  # noqa: PLC0415

    one = {"wrong_kept": 3, "right_removed": 20}
    assert report.reading(
        {"1 other model": one, "3 claim check then 1B": {"wrong_kept": 5, "right_removed": 19}}
    )
    assert not report.reading(
        {"1 other model": one, "3 claim check then 1B": {"wrong_kept": 6, "right_removed": 1}}
    )
    assert not report.reading(
        {"1 other model": one, "3 claim check then 1B": {"wrong_kept": 3, "right_removed": 20}}
    )


def test_the_graders_random_check_is_seeded_and_per_generator() -> None:
    from scripts import grade  # noqa: PLC0415

    key = {f"Q{i:03d}A1": {"answers": ["flash" if i % 2 else "opus46"]} for i in range(200)}
    first = grade.checks(key)
    assert first == grade.checks(key)
    assert len(first) == 2 * grade.CHECK_PER_GENERATOR


def test_the_totals_flag_reaches_the_config() -> None:
    # The installed 0.3.0 has no ``totals``: the flag must fail there, not be ignored.
    row = {"qid": "A01-en", "premise": "Price: €279.99", "query": "Where?", "sentences": ["Hi."]}
    out = subprocess.run(  # noqa: S603
        [sys.executable, str(ROOT / "scripts/claimcheck.py"), "--totals"],
        input=json.dumps(row) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode != 0
    assert "totals" in out.stderr
    assert check.TOTALS == "9ab6603"
    assert check.TOTALS_RELEASE == "legal-rag-verifier==0.4.0"


def test_euro_amounts_are_read_without_the_checker() -> None:
    from scripts import report  # noqa: PLC0415

    assert report.euro_cents("at **€908.98** (€903.99 + €4.99)") == {90898, 90399, 499}
    assert report.euro_cents("a total of 60,98 € (50,99 € + 9,99 €)") == {6098, 5099, 999}
    assert report.euro_cents("1 299,99 €, EUR 12.50, €1,299") == {129999, 1250, 129900}
    assert report.euro_cents("by November 2026 120 € in all") == {12000}
    assert report.euro_cents("Returns within 30 days.") == set()


def test_offer_totals_leave_out_free_shipping() -> None:
    offers = [{"price": 90399, "shipping": 499}, {"price": 24799, "shipping": 0}]
    assert check.offer_totals(offers) == {90898}
    assert check.only_totals_failed(["EUR:908.98"], {90898})
    assert not check.only_totals_failed(["EUR:908.98", "5"], {90898})
    assert not check.only_totals_failed(["EUR:316.99"], {90898})
    assert not check.only_totals_failed([], {90898})


def test_the_dev_check_compares_changed_with_expected() -> None:
    offers = {"D09-en": [{"price": 31100, "shipping": 499}]}
    reference = {
        "D09-en": {
            "verdicts": ["ROLLBACK", "ROLLBACK", "EMIT"],
            "failed": [["EUR:315.99"], ["EUR:316.99"], []],
        }
    }
    after = {"D09-en": {"verdicts": ["EMIT", "ROLLBACK", "EMIT"]}}
    diff = check.changes(reference, after, offers)
    assert diff == {"changed": [("D09-en", 0)], "expected": [("D09-en", 0)]}


def test_the_reading_rule_for_arm_3t_reads_its_own_row() -> None:
    from scripts import report  # noqa: PLC0415

    arms = {
        "1 other model": {"wrong_kept": 0, "right_removed": 94},
        "3 claim check then 1B": {"wrong_kept": 5, "right_removed": 148},
        "3t claim check with totals then 1B": {"wrong_kept": 2, "right_removed": 60},
    }
    assert not report.reading(arms)
    assert report.reading(arms, "3t claim check with totals then 1B")
