from __future__ import annotations

import random
from datetime import date

import pytest

from product_answer_checks import offers as of


def test_money_and_dates_in_both_languages() -> None:
    assert of.money_en(129999) == "€1,299.99"
    assert of.money_fr(129999) == "1 299,99 €"
    assert of.date_en(date(2026, 11, 5)) == "Thursday 5 November 2026"
    assert of.date_fr(date(2026, 12, 1)) == "mardi 1er décembre 2026"
    assert of.working_days_after(of.AS_OF, 5) == date(2026, 11, 9)


@pytest.mark.parametrize("seed", range(200))
def test_type_a_holds_all_three_traps_and_one_best_delivered_price(seed: int) -> None:
    offers = of.single("A", (19, 1999), random.Random(seed))
    assert len(offers) == 4
    roles = {o.role: o for o in offers}
    in_stock = [o for o in offers if o.in_stock]
    best = min(in_stock, key=lambda o: o.total)
    assert best is roles["best"]
    assert sum(o.total == best.total for o in in_stock) == 1
    assert min(offers, key=lambda o: o.price) is roles["oos"]
    assert roles["shipping"].price < best.price < roles["shipping"].total
    struck = roles["struck"]
    assert struck.old_price is not None
    assert struck.old_price > struck.price


@pytest.mark.parametrize("seed", range(200))
def test_type_d_has_exactly_one_offer_meeting_both(seed: int) -> None:
    offers = of.single("D", (19, 1999), random.Random(seed))
    due = of.deadline(offers)
    meets = [
        o for o in offers if o.in_stock and o.delivery <= due and o.returns >= of.RETURNS_WANTED
    ]
    assert len(meets) == 1
    assert any(o.in_stock and o.delivery <= due and o.returns < of.RETURNS_WANTED for o in offers)
    assert any(o.in_stock and o.delivery > due and o.returns >= of.RETURNS_WANTED for o in offers)


@pytest.mark.parametrize("seed", range(200))
def test_type_b_budget_roles(seed: int) -> None:
    budget, (under, over, cheap) = of.budget_three(
        (19, 1999), ("under", "over", "cheap"), random.Random(seed)
    )
    assert under.total < budget
    assert over.price < budget < over.total
    assert cheap.price < min(under.price, over.price)
    assert sum(o.old_price is not None for o in (under, over, cheap)) == 1


@pytest.mark.parametrize("type_", ["C", "E"])
def test_other_types_have_the_struck_price_and_another_trap(type_: str) -> None:
    for seed in range(100):
        offers = of.single(type_, (49, 1999), random.Random(seed))
        roles = {o.role for o in offers}
        assert "struck" in roles
        assert "best" in roles
        assert roles & {"shipping", "oos"}
        assert len(offers) in {3, 4}


def test_french_and_english_lines_hold_the_same_values() -> None:
    o = of.Offer(27999, 0, date(2026, 11, 5), 30, old_price=32999)
    assert of.line_en("A", o) == (
        "Shop A — Price: €279.99 (was €329.99) — Shipping: free — Estimated delivery Thursday 5 "
        "November 2026 — Returns within 30 days — In stock"
    )
    assert of.line_fr("A", o) == (
        "Boutique A — Prix : 279,99 € (au lieu de 329,99 €) — Livraison : gratuite — "
        "Livraison estimée le jeudi 5 novembre 2026 — Retours sous 30 jours — En stock"
    )
