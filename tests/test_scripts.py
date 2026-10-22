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


def test_no_opus_55_call_before_the_spend_go() -> None:
    assert router.CAPS[router.OPUS55] == 0.0
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
