"""Fetch test 1's products from Open Icecat (docs/METHOD.md §1).

  uv run python scripts/fetch_icecat.py draw      # the seeded draw, resumable → data/icecat/
  uv run python scripts/fetch_icecat.py rebuild   # a reader's copy of the sealed sheets, by hash

Needs an Open Icecat username in ``ICECAT_USERNAME`` (or ``.env``). Sheets stay in ``data/``
(git-ignored): the Open Icecat licence's terms for republishing sheet text are not assumed, so
only IDs and hashes are committed (``battery/``).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from product_answer_checks import sheets as sh  # noqa: E402
from product_answer_checks.selection import SEED, Candidate, Selection, select  # noqa: E402

API = "https://live.icecat.biz/api"
DATA = ROOT / "data/icecat"
DRAW = DATA / "draw.jsonl"
SEAL = ROOT / "battery/seal.json"
LOW, HIGH = 90_000_000, 135_000_000
MAX_DRAWS = 40_000
WORKERS = 8
PAUSE = 0.2  # seconds per request per worker: with the replies, about 10 requests a second in all
BATCH = 400
FINAL = {
    "eligible",
    "no English sheet",
    "category",
    "few features",
    "no French sheet",
    "French title not in French",
}
_lock = threading.Lock()


def username() -> str:
    name = os.environ.get("ICECAT_USERNAME")
    env = ROOT / ".env"
    if not name and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("ICECAT_USERNAME="):
                name = line.split("=", 1)[1].strip()
    if not name:
        raise SystemExit("set ICECAT_USERNAME (an Open Icecat account's username)")
    return name


def fetch(icecat_id: int, lang: str, user: str) -> tuple[dict[str, Any] | None, bytes]:
    """One sheet: the parsed JSON (None when not an Open Icecat product) and the raw bytes."""
    query = urllib.parse.urlencode({"UserName": user, "Language": lang, "icecat_id": icecat_id})
    for attempt in range(4):
        time.sleep(PAUSE)
        try:
            with urllib.request.urlopen(f"{API}?{query}", timeout=60) as response:  # noqa: S310
                raw = response.read()
            return json.loads(raw), raw
        except urllib.error.HTTPError as exc:
            if exc.code in {400, 403, 404, 410}:
                return None, b""
            error: Exception = exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            error = exc
        time.sleep(2**attempt)
    raise RuntimeError(f"Icecat {icecat_id} {lang}: {error!r}")


def path(icecat_id: int, lang: str) -> Path:
    return DATA / f"{icecat_id}.{lang}.json"


def load(icecat_id: int, lang: str) -> sh.Sheet:
    sheet: sh.Sheet = json.loads(path(icecat_id, lang).read_bytes())
    return sheet


def draws() -> Iterator[tuple[int, int]]:
    rng = random.Random(SEED)
    for n in range(MAX_DRAWS):
        yield n, rng.randrange(LOW, HIGH)


def summary(icecat_id: int) -> dict[str, Any]:
    en = load(icecat_id, "en")
    numeric = {fid: [str(q.value), q.unit] for fid, q in sh.numeric_features(en).items()}
    return {
        "category": sh.category(en),
        "brand": sh.brand(en),
        "energy": sh.energy_features(en) is not None,
        "weight": sh.feature(en, sh.WEIGHT) is not None,
        "numeric": numeric,
    }


def check(n: int, icecat_id: int, user: str) -> dict[str, Any]:
    """Fetch and judge one drawn ID; sheets that pass are kept with their French twin."""
    en, raw_en = fetch(icecat_id, "en", user)
    record: dict[str, Any] = {"n": n, "id": icecat_id}
    if (
        en is not None
        and sh.ok(en)
        and sh.in_scope(sh.category(en))
        and len(sh.features(en)) >= sh.MIN_FEATURES
    ):
        fr, raw_fr = fetch(icecat_id, "fr", user)
        reason = sh.ineligible(en, fr)
        if reason is None:
            path(icecat_id, "en").write_bytes(raw_en)
            path(icecat_id, "fr").write_bytes(raw_fr)
    else:
        reason = sh.ineligible(en, None)
    record["status"] = reason or "eligible"
    if en is not None and sh.ok(en):
        record["category"] = sh.category(en)
    if reason is None:
        record |= summary(icecat_id)
    return record


def read_draw() -> dict[int, dict[str, Any]]:
    if not DRAW.exists():
        return {}
    rows = [json.loads(line) for line in DRAW.read_text(encoding="utf-8").splitlines() if line]
    return {r["n"]: r for r in rows if r["status"] in FINAL}


def candidates(records: Mapping[int, Mapping[str, Any]]) -> list[Candidate]:
    out = []
    for n in sorted(records):
        r = records[n]
        if r["status"] != "eligible":
            continue
        numeric = {fid: sh.Quantity(Decimal(v), unit) for fid, (v, unit) in r["numeric"].items()}
        out.append(Candidate(r["id"], r["category"], r["brand"], r["energy"], r["weight"], numeric))
    return out


def progress(selection: Selection) -> str:
    sizes = " ".join(f"{t}{len(ids)}" for t, ids in selection.types.items())
    companions = sum(map(len, selection.companions.values()))
    return f"dev {len(selection.dev)} {sizes} companions {companions}"


def draw() -> Selection:
    """Draw in order until the battery is full or MAX_DRAWS IDs are drawn (METHOD §1)."""
    user = username()
    DATA.mkdir(parents=True, exist_ok=True)
    records = read_draw()
    todo = [(n, i) for n, i in draws() if n not in records]
    selection = select(candidates(records))
    with ThreadPoolExecutor(WORKERS) as pool, DRAW.open("a", encoding="utf-8") as log:
        for start in range(0, len(todo), BATCH):
            if selection.complete:
                break
            batch = todo[start : start + BATCH]
            for record in pool.map(lambda job: check(job[0], job[1], user), batch):
                with _lock:
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")
                records[record["n"]] = record
            log.flush()
            # Selection reads the draw in order, so only a complete prefix counts.
            prefix = {}
            for n in range(len(records) + 1):
                if n not in records:
                    break
                prefix[n] = records[n]
            selection = select(candidates(prefix))
            counts = Counter(r["status"] for r in prefix.values())
            print(
                f"{len(prefix)} drawn, {counts['eligible']} eligible: {progress(selection)}",
                flush=True,
            )
    last = max(records) if records else -1
    print(f"done at draw {last + 1}: {progress(selection)}, complete={selection.complete}")
    return selection


def rebuild() -> None:
    """Fetch every sealed sheet again with the reader's username and compare its listing hash."""
    from product_answer_checks.battery import sha256  # noqa: PLC0415

    user = username()
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    DATA.mkdir(parents=True, exist_ok=True)
    changed = []
    for key, expected in seal["listings"].items():
        icecat_id, lang = key.split(".")
        sheet, raw = fetch(int(icecat_id), lang, user)
        if sheet is None:
            changed.append(f"{key}: no longer in Open Icecat")
            continue
        path(int(icecat_id), lang).write_bytes(raw)
        if sha256(sh.listing(sheet)) != expected:
            changed.append(f"{key}: listing changed since {seal['fetched']}")
    print(f"{len(seal['listings'])} sheets fetched; {len(changed)} differ from the seal")
    for line in changed:
        print(" ", line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["draw", "rebuild"])
    args = parser.parse_args()
    if args.command == "draw":
        started = datetime.now(UTC).isoformat(timespec="seconds")
        draw()
        print(f"started {started}")
    else:
        rebuild()


if __name__ == "__main__":
    main()
