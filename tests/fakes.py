"""Small Icecat-shaped sheets for the tests (no network, no data/)."""

from __future__ import annotations

from typing import Any


def sheet(
    category: str = "Laptops",
    features: list[tuple[str, str, str, str]] | None = None,
    *,
    lang: str = "EN",
    title: str = "Brand Model 16",
    brand: str = "Brand",
    extra: int = 15,
) -> dict[str, Any]:
    """A sheet with the given (id, group, name, value) features plus ``extra`` filler ones."""
    rows = list(features or [])
    rows += [(f"9{i:03d}", "Other", f"Filler {i}", "Yes") for i in range(extra)]
    groups: dict[str, list[dict[str, Any]]] = {}
    for fid, group, name, value in rows:
        groups.setdefault(group, []).append(
            {"Feature": {"ID": fid, "Name": {"Value": name}}, "PresentationValue": value}
        )
    return {
        "msg": "OK",
        "data": {
            "GeneralInfo": {
                "Title": title,
                "Brand": brand,
                "Category": {"Name": {"Value": category}},
                "TitleInfo": {"GeneratedLocalTitle": {"Value": title, "Language": lang}},
                "SummaryDescription": {"ShortSummaryDescription": f"{title}, summary"},
            },
            "FeaturesGroups": [
                {"FeatureGroup": {"Name": {"Value": g}}, "Features": fs} for g, fs in groups.items()
            ],
        },
    }
