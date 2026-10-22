from __future__ import annotations

from decimal import Decimal

from product_answer_checks.selection import DEV, PER_TYPE, Candidate, b_feature, select
from product_answer_checks.sheets import Quantity


def cand(i: int, category: str = "Laptops", *, energy: bool = False, weight: bool = True,
         value: int | None = None, brand: str | None = None) -> Candidate:  # fmt: skip
    numeric = {"1": Quantity(Decimal(value if value is not None else i), "Wh")}
    return Candidate(i, category, brand or f"brand{i}", energy, weight, numeric)


def test_the_spec_feature_needs_two_above_the_third() -> None:
    assert b_feature([cand(1, value=80), cand(2, value=80), cand(3, value=50)], 0) == "1"
    assert b_feature([cand(1, value=80), cand(2, value=50), cand(3, value=50)], 0) is None


def test_selection_fills_dev_then_each_type_in_draw_order() -> None:
    pool = [cand(i) for i in range(DEV)]
    pool += [cand(100 + i, "TVs", energy=True) for i in range(PER_TYPE)]
    pool += [cand(200 + i, "Laptops") for i in range(PER_TYPE * 3)]
    pool += [cand(400 + i, "Monitors") for i in range(PER_TYPE * 3)]
    out = select(pool)
    assert out.dev == list(range(DEV))
    assert out.types["E"] == [100 + i for i in range(PER_TYPE)]
    assert len(out.types["B"]) == PER_TYPE
    used = [i for ids in out.types.values() for i in ids] + [
        c for cs in out.companions.values() for c in cs
    ]
    assert len(used) == len(set(used))
    assert out.complete


def test_at_most_two_per_brand_and_category() -> None:
    out = select([cand(i, brand="same") for i in range(5)])
    assert out.dev == [0, 1]


def test_a_short_draw_is_not_complete() -> None:
    assert not select([cand(i) for i in range(20)]).complete
