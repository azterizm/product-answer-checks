from __future__ import annotations

from decimal import Decimal

from product_answer_checks import battery as bt
from product_answer_checks.selection import Selection
from tests.fakes import sheet

FEATURES = [
    ("94", "Weight & dimensions", "Weight", "2.4 kg"),
    ("1", "Battery", "Battery capacity", "80 Wh"),
    ("2705", "Power", "Energy efficiency class", "E"),
    ("42849", "Power", "Energy consumption per 1000 h", "97 kWh"),
]


def sheets(pid: int, lang: str) -> dict[str, object]:
    capacity = ("1", "Battery", "Battery capacity", f"{pid} Wh")
    return sheet(
        features=[*FEATURES[:1], capacity, *FEATURES[2:]], lang=lang.upper(), title=f"P{pid} {lang}"
    )


def test_each_question_is_written_once_in_each_language_with_the_same_offers() -> None:
    sel = Selection(dev=[1, 2, 3])
    sel.types = {"A": [10], "B": [20], "C": [30], "D": [40], "E": [50]}
    sel.companions = {20: [21, 22]}
    sel.b_feature = {20: "1"}
    battery, dev = bt.questions(sel, sheets)
    assert [q.qid for q in battery] == [f"{t}01-{lang}" for t in "ABCDE" for lang in ("fr", "en")]
    assert [q.type for q in dev[::2]] == ["A", "C", "D"]
    for fr, en in zip(battery[::2], battery[1::2], strict=True):
        assert fr.question == en.question
        assert fr.offers == en.offers
        assert fr.premise != en.premise
        assert "Prix :" in fr.premise
        assert "Price:" in en.premise


def test_the_budget_question_names_the_second_highest_value() -> None:
    sel = Selection()
    sel.types = {"A": [], "B": [20], "C": [], "D": [], "E": []}
    sel.companions = {20: [30, 10]}
    sel.b_feature = {20: "1"}
    battery, _ = bt.questions(sel, sheets)
    q = battery[1]
    assert "battery capacity of at least 20 Wh" in q.question
    assert q.premise.count("Product ") == 3
    under = q.products[q.intended["product"] - 1]
    assert Decimal(under) >= 20


def test_the_committed_record_hashes_the_premise_instead_of_holding_it() -> None:
    sel = Selection()
    sel.types = {"A": [10], "B": [], "C": [], "D": [], "E": []}
    battery, _ = bt.questions(sel, sheets)
    record = battery[0].record()
    assert "premise" not in record
    assert record["premise_sha256"] == bt.sha256(battery[0].premise)


def test_the_census_counts_the_unread_forms() -> None:
    out = bt.census(
        {"a": "Batterie : 5 000 mAh", "b": "Prix 1 299,99 € et 1,234 kg", "c": "2,4 kg"}
    )
    assert out["plain-space groups"]["count"] == 1
    assert out["comma before three digits"]["count"] == 1
