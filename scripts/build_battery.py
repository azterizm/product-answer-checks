"""Build and seal test 1's battery from the draw (docs/METHOD.md §1–§4).

  uv run python scripts/build_battery.py build   # → battery/ and the census, not yet sealed
  uv run python scripts/build_battery.py seal    # → battery/seal.json, before any generator call

``battery/`` holds what may be committed: IDs, categories, brands, types, offers, questions, the
prompt and hashes. Listings (sheet text) are rebuilt from ``data/`` and never written there.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import fetch_icecat as fi  # type: ignore[import-not-found]  # noqa: E402

from product_answer_checks import battery as bt  # noqa: E402
from product_answer_checks import sheets as sh  # noqa: E402
from product_answer_checks.selection import MIN_BATTERY, Selection  # noqa: E402

BATTERY = ROOT / "battery"
PRODUCTS = BATTERY / "products.jsonl"
QUESTIONS = BATTERY / "questions.jsonl"
DEV = BATTERY / "dev.jsonl"
PROMPT = BATTERY / "prompt.txt"
CENSUS = BATTERY / "census.json"
SEAL = fi.SEAL


def selection() -> tuple[Selection, dict[str, Any]]:
    prefix, out = fi.stopping_point(fi.read_draw())
    counts = Counter(r["status"] for r in prefix.values())
    return out, {"drawn": len(prefix), "statuses": dict(sorted(counts.items()))}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(bt.dumps(r) + "\n" for r in rows), encoding="utf-8")


def product_rows(sel: Selection) -> list[dict[str, Any]]:
    role: dict[int, tuple[str, str]] = dict.fromkeys(sel.dev, ("dev", ""))
    for type_, ids in sel.types.items():
        for pid in ids:
            role[pid] = ("battery", type_)
    for pid, companions in sel.companions.items():
        for c in companions:
            role[c] = ("companion", f"B of {pid}")
    rows = []
    for pid, (set_, type_) in role.items():
        en = fi.load(pid, "en")
        rows.append(
            {
                "id": pid,
                "set": set_,
                "type": type_,
                "category": sh.category(en),
                "brand": sh.brand(en),
            }
        )
    return rows


def build() -> None:
    sel, draw = selection()
    if not sel.complete and sel.battery < MIN_BATTERY:
        raise SystemExit(f"only {sel.battery} battery products; the draw is not finished")
    battery, dev = bt.questions(sel, fi.load)
    BATTERY.mkdir(exist_ok=True)
    write_jsonl(PRODUCTS, product_rows(sel))
    write_jsonl(QUESTIONS, [q.record() for q in battery])
    write_jsonl(DEV, [q.record() for q in dev])
    PROMPT.write_text(bt.PROMPT + "\n", encoding="utf-8")
    french = {f"{pid}": sh.listing(fi.load(pid, "fr")) for pid in sorted(_all_products(sel))}
    report = {"draw": draw, "complete": sel.complete, "census": bt.census(french)}
    CENSUS.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    sizes = {t: len(ids) for t, ids in sel.types.items()}
    print(
        json.dumps({"questions": len(battery), "dev": len(dev), "types": sizes, **report}, indent=1)
    )


def _all_products(sel: Selection) -> set[int]:
    ids = set(sel.dev) | {pid for ids in sel.types.values() for pid in ids}
    return ids | {c for cs in sel.companions.values() for c in cs}


def seal() -> None:
    sel, draw = selection()
    battery, dev = bt.questions(sel, fi.load)
    stored = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()]
    if stored != [json.loads(bt.dumps(q.record())) for q in battery]:
        raise SystemExit("battery/questions.jsonl does not match a fresh build; run build first")
    ids = sorted(_all_products(sel))
    out = {
        "sealed": datetime.now(UTC).isoformat(timespec="seconds"),
        "fetched": _fetched(ids),
        "draw": draw,
        "prompt": bt.sha256(PROMPT.read_bytes()),
        "files": {p.name: bt.sha256(p.read_bytes()) for p in (PRODUCTS, QUESTIONS, DEV, CENSUS)},
        "raw": {
            f"{pid}.{lang}": bt.sha256(fi.path(pid, lang).read_bytes())
            for pid in ids
            for lang in bt.LANGS
        },
        "listings": {
            f"{pid}.{lang}": bt.sha256(sh.listing(fi.load(pid, lang)))
            for pid in ids
            for lang in bt.LANGS
        },
        "premises": {q.qid: bt.sha256(q.premise) for q in [*battery, *dev]},
    }
    SEAL.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"sealed {len(battery)} questions, {len(dev)} dev, {len(ids)} products → {SEAL}")


def _fetched(ids: list[int]) -> str:
    times = [fi.path(pid, lang).stat().st_mtime for pid in ids for lang in bt.LANGS]
    first, last = (
        datetime.fromtimestamp(t, UTC).date().isoformat() for t in (min(times), max(times))
    )
    return first if first == last else f"{first} to {last}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "seal"])
    args = parser.parse_args()
    build() if args.command == "build" else seal()


if __name__ == "__main__":
    main()
