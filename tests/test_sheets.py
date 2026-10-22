from __future__ import annotations

from decimal import Decimal

import pytest

from product_answer_checks import sheets as sh
from tests.fakes import sheet


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("Laptops", True),
        ("PCs/Workstations", True),
        ("TVs", True),
        ("Hair dryers", True),
        ("Office chairs", False),
        ("Laptop Spare Parts", False),
        ("Mobile Phone Cases", False),
        ("Warranty & Support Extensions", False),
        ("Socket-Outlets", False),
        ("Dishwashers", True),
    ],
)
def test_category_words_match_at_the_start_of_a_word(category: str, expected: bool) -> None:
    assert sh.in_scope(category) is expected


def test_price_bands_by_category() -> None:
    assert sh.price_band("Electric kettles") == (19, 199)
    assert sh.price_band("Dishwashers") == (249, 1299)
    assert sh.price_band("Laptops") == (49, 1999)


def test_eligibility_needs_both_sheets_in_scope_and_in_french() -> None:
    en, fr = sheet(), sheet(lang="FR")
    assert sh.ineligible(en, fr) is None
    assert sh.ineligible(None, fr) == "no English sheet"
    assert sh.ineligible(sheet("Office chairs"), fr) == "category"
    assert sh.ineligible(en, None) == "no French sheet"
    assert sh.ineligible(en, sheet(lang="EN")) == "French title not in French"
    assert sh.ineligible(sheet(extra=10), fr) == "few features"


def test_numeric_features_leave_out_weight_packaging_and_yes_no() -> None:
    s = sheet(
        features=[
            ("94", "Weight & dimensions", "Weight", "2.4 kg"),
            ("762", "Packaging data", "Package weight", "3 kg"),
            ("1", "Battery", "Battery capacity", "80 Wh"),
            ("2", "Display", "Resolution", "2560 x 1600 pixels"),
            ("3", "Ports", "USB ports quantity", "2"),
        ]
    )
    assert sh.numeric_features(s) == {
        "1": sh.Quantity(Decimal("80"), "Wh"),
        "3": sh.Quantity(Decimal("2"), ""),
    }


def test_energy_features_need_a_class_and_a_consumption() -> None:
    klass = ("2705", "Power", "Energy efficiency class", "E")
    use = ("42849", "Power", "Energy consumption per 100 cycles", "97 kWh")
    found = sh.energy_features(sheet(features=[klass, use]))
    assert found is not None
    assert [f.value for f in found] == ["E", "97 kWh"]
    assert sh.energy_features(sheet(features=[klass])) is None


def test_the_listing_is_title_summary_then_features() -> None:
    s = sheet(features=[("1", "Battery", "Battery capacity", "80 Wh")], extra=0)
    assert (
        sh.listing(s)
        == "Brand Model 16\nBrand Model 16, summary\nBattery — Battery capacity: 80 Wh"
    )
