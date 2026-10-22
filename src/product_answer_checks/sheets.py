"""Open Icecat sheets: which products are in scope, their features, and the listing a model reads.

METHOD.md §1 (eligibility, numeric features) and §2 (the listing).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

Sheet = Mapping[str, Any]

INCLUDE = (
    "laptop",
    "notebook",
    "tablet",
    "pc",
    "workstation",
    "monitor",
    "smartphone",
    "mobile phone",
    "smartwatch",
    "headphone",
    "headset",
    "earphone",
    "speaker",
    "soundbar",
    "tv",
    "television",
    "projector",
    "camera",
    "e-book",
    "power bank",
    "console",
    "printer",
    "keyboard",
    "mouse",
    "mice",
    "router",
    "hard drive",
    "solid state drive",
    "flash drive",
    "vacuum",
    "coffee",
    "kettle",
    "toaster",
    "blender",
    "food processor",
    "microwave",
    "hair",
    "shaver",
    "toothbrush",
    "iron",
    "fryer",
    "fan",
    "air purifier",
    "heater",
    "dishwasher",
    "washing machine",
    "refrigerator",
    "fridge",
    "freezer",
    "oven",
    "dryer",
    "hob",
    "cooker",
)
EXCLUDE = (
    "spare part",
    "accessor",
    "case",
    "cover",
    "cable",
    "mount",
    "bag",
    "part",
    "screen protector",
    "warranty",
    "replacement",
    "filter",
    "stand",
    "dock",
    "adapter",
    "ink",
    "toner",
    "licence",
    "license",
)
SMALL_APPLIANCES = (
    "vacuum",
    "coffee",
    "kettle",
    "toaster",
    "blender",
    "food processor",
    "microwave",
    "hair",
    "shaver",
    "toothbrush",
    "iron",
    "fryer",
    "fan",
    "air purifier",
    "heater",
)
LARGE_APPLIANCES = (
    "dishwasher",
    "washing machine",
    "refrigerator",
    "fridge",
    "freezer",
    "oven",
    "dryer",
    "hob",
    "cooker",
)
MIN_FEATURES = 15
WEIGHT = "94"
ENERGY_CLASS = "2705"
ENERGY_PREFIXES = ("energy consumption", "power consumption")
PACKAGING = "packaging"
_NUMERIC = re.compile(r"(?P<num>\d+(?:\.\d+)?)(?:\s?(?P<unit>[A-Za-z%°µ/²³]+))?")
_YES_NO = {"yes", "no"}


def _starts_a_word(words: Sequence[str], name: str) -> bool:
    return any(re.search(rf"(?<![\w-]){re.escape(w)}", name) for w in words)


def in_scope(category: str) -> bool:
    """A word of the category starts with an included word and none with an excluded one."""
    name = category.lower()
    return _starts_a_word(INCLUDE, name) and not _starts_a_word(EXCLUDE, name)


def price_band(category: str) -> tuple[int, int]:
    """The base-price band in euros for a category (METHOD §2)."""
    name = category.lower()
    if _starts_a_word(SMALL_APPLIANCES, name):
        return 19, 199
    if _starts_a_word(LARGE_APPLIANCES, name):
        return 249, 1299
    return 49, 1999


@dataclass(frozen=True)
class Feature:
    id: str
    group: str
    name: str
    value: str


def info(sheet: Sheet) -> Mapping[str, Any]:
    general: Mapping[str, Any] = sheet["data"]["GeneralInfo"]
    return general


def category(sheet: Sheet) -> str:
    return str(info(sheet)["Category"]["Name"]["Value"])


def brand(sheet: Sheet) -> str:
    return str(info(sheet)["Brand"])


def title(sheet: Sheet) -> str:
    return str(info(sheet)["Title"])


def local_language(sheet: Sheet) -> str:
    return str(info(sheet)["TitleInfo"]["GeneratedLocalTitle"]["Language"])


def summary(sheet: Sheet) -> str:
    return str((info(sheet).get("SummaryDescription") or {}).get("ShortSummaryDescription") or "")


def features(sheet: Sheet) -> list[Feature]:
    """Every feature with a value, in sheet order."""
    out = []
    for group in sheet["data"].get("FeaturesGroups") or []:
        group_name = group["FeatureGroup"]["Name"]["Value"]
        for f in group["Features"]:
            value = (f.get("PresentationValue") or "").strip()
            if value:
                out.append(
                    Feature(
                        str(f["Feature"]["ID"]), group_name, f["Feature"]["Name"]["Value"], value
                    )
                )
    return out


def ok(sheet: Sheet | None) -> bool:
    return bool(sheet) and sheet.get("msg") == "OK" and isinstance(sheet.get("data"), Mapping)  # type: ignore[union-attr]


def ineligible(en: Sheet | None, fr: Sheet | None) -> str | None:
    """Why a product is not eligible (METHOD §1, rules 1–3), or None; rule 4 is for selection."""
    reason = None
    if en is None or not ok(en):
        reason = "no English sheet"
    elif not in_scope(category(en)):
        reason = "category"
    elif fr is None or not ok(fr):
        reason = "no French sheet" if len(features(en)) >= MIN_FEATURES else "few features"
    elif local_language(fr) != "FR":
        reason = "French title not in French"
    elif min(len(features(en)), len(features(fr))) < MIN_FEATURES:
        reason = "few features"
    return reason


@dataclass(frozen=True)
class Quantity:
    value: Decimal
    unit: str


def quantity(value: str) -> Quantity | None:
    """A number with at most one unit ("80 Wh", "2.4 kg", "16"), else None."""
    m = _NUMERIC.fullmatch(value.strip())
    if m is None:
        return None
    return Quantity(Decimal(m["num"]), m["unit"] or "")


def numeric_features(sheet: Sheet) -> dict[str, Quantity]:
    """Numeric features by id (METHOD §1): not weight, not packaging, not yes/no."""
    out: dict[str, Quantity] = {}
    for f in features(sheet):
        if f.id == WEIGHT or PACKAGING in f.group.lower() or f.value.lower() in _YES_NO:
            continue
        q = quantity(f.value)
        if q is not None:
            out.setdefault(f.id, q)
    return out


def feature(sheet: Sheet, feature_id: str) -> Feature | None:
    return next((f for f in features(sheet) if f.id == feature_id), None)


def energy_features(sheet: Sheet) -> tuple[Feature, Feature] | None:
    """The energy class and the first energy or power consumption feature (type E), or None."""
    klass = feature(sheet, ENERGY_CLASS)
    use = next((f for f in features(sheet) if f.name.lower().startswith(ENERGY_PREFIXES)), None)
    return None if klass is None or use is None else (klass, use)


def listing(sheet: Sheet) -> str:
    """Title, short summary, then every feature as ``Group — Name: Value`` (METHOD §2)."""
    lines = [title(sheet)]
    if text := summary(sheet):
        lines.append(text)
    lines += [f"{f.group} — {f.name}: {f.value}" for f in features(sheet)]
    return "\n".join(lines)
