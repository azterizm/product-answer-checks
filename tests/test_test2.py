"""Test 2's own logic (docs/METHOD2.md): the made-up products, the recorded prompts, the targets,
the cost rule and the ledger's stage. No network, no router, no Modal."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from legal_rag_verifier.verifier import Verifier
from product_answer_checks import battery as bt
from product_answer_checks import sheets as sh
from product_answer_checks import synthetic as sy
from product_answer_checks.selection import SEED
from scripts import judge, modal_ledger, record2, test2


def test_the_products_are_made_up_in_twelve_categories_and_split_by_product() -> None:
    assert len(sy.IDS) == 120
    assert Counter(sy.split().values()) == {"train": 108, "validation": 12}
    assert sy.split() == sy.split()
    for pid in sy.IDS:
        en, fr = sy.sheet(pid, "en"), sy.sheet(pid, "fr")
        assert sh.brand(en) in sy.BRANDS
        assert len(sh.features(en)) == len(sh.features(fr)) >= sh.MIN_FEATURES
        assert sh.feature(en, sh.WEIGHT) is not None
        assert sh.local_language(fr) == "FR"
        assert (sh.energy_features(en) is not None) == sy.category_of(pid).energy
    assert sum(sy.category_of(p).energy for p in sy.IDS) == 50


def test_french_values_use_the_decimal_comma_and_french_units() -> None:
    laptop = 1
    weight_fr = sh.feature(sy.sheet(laptop, "fr"), sh.WEIGHT)
    weight_en = sh.feature(sy.sheet(laptop, "en"), sh.WEIGHT)
    assert weight_fr is not None
    assert weight_en is not None
    assert weight_en.value.replace(".", ",") == weight_fr.value
    memory = [f for f in sh.features(sy.sheet(laptop, "fr")) if f.name == "Mémoire interne"]
    assert memory[0].value.endswith(" Go")


def test_every_question_comes_from_test_1s_code_with_its_own_seed() -> None:
    questions = sy.questions()
    assert len({q.qid for q in questions}) == len(questions)
    types = Counter(q.type for q in questions)
    assert types["A"] == types["C"] == types["D"] == 240
    assert types["E"] == 100
    where = sy.split()
    for q in questions:
        assert len({where[p] for p in q.products}) == 1  # a triple never crosses the split
    a, c = (next(q for q in questions if q.qid == f"T001{t}-en") for t in "AC")
    assert a.offers != c.offers  # each type draws its own offers


def test_the_seed_argument_leaves_test_1s_questions_as_they_were() -> None:
    def sheets(pid: int, lang: str) -> dict[str, object]:
        return sy.sheet(1, lang)

    default = bt.single_questions("A01", "A", 7, sheets)
    explicit = bt.single_questions("A01", "A", 7, sheets, seed=SEED + 7)
    assert [q.record() for q in default] == [q.record() for q in explicit]


PREMISE = (
    "Acme Kettle\nPower: 2200 W\n\nOffers as of Monday 2 November 2026:\n"
    "Shop A — Price: €59.99 — Shipping: free — In stock\n"
    "Shop B — Price: €54.99 — Shipping: €9.99 — In stock"
)


def test_the_recorder_keeps_each_prompt_and_answers_as_the_grade_says() -> None:
    scorer = record2.Recorder()
    claims, judged = Verifier(), Verifier(scorer, judge.STRICT_FIGURES)
    row = {
        "qid": "T001A-en",
        "generator": "qwen7b",
        "premise": PREMISE,
        "query": "Where is it cheapest?",
        "sentences": [
            "Shop A sells it for €59.99 with free shipping.",
            "It costs €61.00 at Shop C.",
            "Shop B is the cheapest option.",
        ],
        "grades": ["yes", "no", "no"],
    }
    out = record2.record(row, claims, judged, scorer)
    assert out["claims"] == ["EMIT", "ROLLBACK", "EMIT"]
    assert out["verdicts"] == ["EMIT", "ROLLBACK", "ROLLBACK"]
    assert out["prompts"][1] == []
    first, third = out["prompts"][0], out["prompts"][2]
    assert len(first) == len(third) == 1
    assert "(none)" in first[0]
    assert "Shop A sells it for €59.99 with free shipping." in third[0].split("Sentence to")[0]


def _row(prompts: list[list[str]], grades: list[str | None], claims: list[str]) -> dict[str, Any]:
    return {"qid": "T001A-en", "generator": "flash", "prompts": prompts, "grades": grades,
            "claims": claims}  # fmt: skip


def test_targets_follow_the_grade_and_leave_out_what_it_cannot_place() -> None:
    rows = [
        _row(
            [["p1", "p2"], ["p3"], ["p4", "p5"], ["p6"], []],
            ["yes", "no", "no", "unsure", "yes"],
            ["EMIT", "EMIT", "EMIT", "EMIT", "ROLLBACK"],
        ),
        _row([["p1", "p2"]], ["yes"], ["EMIT"]),
    ]
    out, stats = test2.examples(rows, {1: "train"}, {"T001A-en": 1})
    targets = [(e["messages"][1]["content"], e["messages"][2]["content"]) for e in out["train"]]
    assert targets == [
        ("p1", test2.reply("supported")),
        ("p2", test2.reply("supported")),
        ("p3", test2.reply("contradicted")),
    ]
    assert json.loads(test2.reply("contradicted")) == {"label": "contradicted", "passage": 0}
    assert stats["left out: wrong, several clauses"] == 1
    assert stats["left out: grade unsure"] == 1
    assert stats["sentences the claim check rolls back"] == 1
    assert stats["repeated prompt"] == 2
    assert out["train"][0]["messages"][0]["content"] == judge.SYSTEM


def test_the_cost_rule_drops_the_doubling_then_an_epoch() -> None:
    legal = 76_478_800
    light, heavy, heavier = (test2.plan(legal, p) for p in (10_000_000, 25_000_000, 60_000_000))
    assert (light["epochs"], light["product_copies"]) == (2, 2)
    assert (heavy["epochs"], heavy["product_copies"]) == (2, 1)
    assert (heavier["epochs"], heavier["product_copies"]) == (1, 1)
    assert light["estimate_usd"] <= test2.LIMIT_USD
    assert test2.plan(legal, 0)["stop_seconds"] * test2.H100_PER_SECOND <= test2.STOP_USD


def test_test_2_spends_under_its_own_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(modal_ledger, "LEDGER", tmp_path / "ledger.json")
    modal_ledger.record("arm 3 battery flash", "L40S", 3600)
    modal_ledger.record("test2 train run3", "H100", 1800)
    assert modal_ledger.spent() == modal_ledger.usd("L40S", 3600)
    assert modal_ledger.spent("test2") == modal_ledger.usd("H100", 1800)
    modal_ledger.check("H100", 1800, "test2")
    with pytest.raises(SystemExit, match="test2"):
        modal_ledger.check("H100", 2 * 1800, "test2")
