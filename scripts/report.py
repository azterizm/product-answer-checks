"""Test 1's report (docs/METHOD.md §8): per generator, half and arm, the wrong sentences kept, the
right sentences removed, empty answers and seconds added; price sentences apart; the reading rule.

  uv run python scripts/report.py     # → results/report.json (counts only)
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import answer as an  # type: ignore[import-not-found]  # noqa: E402
import check as ck  # type: ignore[import-not-found]  # noqa: E402
import grade as gr  # type: ignore[import-not-found]  # noqa: E402

REPORT = ROOT / "results/report.json"
ARMS = {
    "0 no check": None,
    "1 other model": "judge",
    "2 claim check": "claims",
    "2b claim check 0.2.0": "claims-020",
    "3 claim check then 1B": "judge1b",
}
ARM1_JUDGE = {"opus46": "gemini-3.8-flash-high (router)", "flash": "qwen27-ar4 (one L40S)"}
HARDWARE = {
    "1 other model": "Opus 4.6's answers: Flash on the router; Flash's: Qwen 27B on one L40S",
    "2 claim check": "CPU (Apple Mac)",
    "2b claim check 0.2.0": "CPU (Apple Mac)",
    "3 claim check then 1B": "CPU, then one L40S",
}
PRICE = re.compile(r"€|\bEUR\b|euro", re.IGNORECASE)
MARGIN = 2  # the reading rule: at most 2 more wrong sentences kept than arm 1


def is_price(sentence: str) -> bool:
    """A price sentence (METHOD, Changes): it contains "€", "EUR" or "euro"."""
    return bool(PRICE.search(sentence))


Row = tuple[str, Sequence[str], Sequence[bool], float | None]


def tally(rows: Iterable[Row], grades: Mapping[tuple[str, str], str]) -> dict[str, Any]:
    """Counts over answers given as (question id, sentences, kept flags, seconds added)."""
    names = ("answers", "sentences", "wrong_kept", "right_removed", "wrong_removed", "right_kept")
    out: dict[str, Any] = dict.fromkeys((*names, "unsettled_kept", "unsettled_removed", "empty"), 0)
    seconds = []
    for qid, said, kept, added in rows:
        out["answers"] += 1
        out["sentences"] += len(said)
        out["empty"] += int(not any(kept))
        if added is not None:
            seconds.append(added)
        for s, k in zip(said, kept, strict=True):
            g = grades.get((qid, s))
            if g not in {"yes", "no"}:  # the graders never agreed: read the counts as a range
                out["unsettled_kept" if k else "unsettled_removed"] += 1
            elif g == "no":
                out["wrong_kept" if k else "wrong_removed"] += 1
            else:
                out["right_kept" if k else "right_removed"] += 1
    out["seconds_added_median"] = round(statistics.median(seconds), 4) if seconds else None
    return out


def reading(arms: Mapping[str, Mapping[str, Any]]) -> bool:
    """Arm 3 "costs less for the same safety": at most MARGIN more wrong kept than arm 1, and fewer
    right removed."""
    one, three = arms["1 other model"], arms["3 claim check then 1B"]
    return bool(
        three["wrong_kept"] <= one["wrong_kept"] + MARGIN
        and three["right_removed"] < one["right_removed"]
    )


def final_grades() -> dict[tuple[str, str], str]:
    """(question id, sentence) → yes / no / unsure: the graders' agreement, else yours."""
    key = json.loads((gr.OUT / "key.json").read_text(encoding="utf-8"))
    settled = gr.consensus()
    yours: dict[str, str] = {}
    file = gr.OUT / "leftover.csv"
    if file.exists():
        with file.open(encoding="utf-8") as f:
            yours = {r["item"]: r["correct"].strip().lower() for r in csv.DictReader(f)}
    out = {}
    for code, item in key.items():
        g = settled[code].get("grade") if settled[code] else yours.get(code, "")
        out[(item["qid"], item["text"])] = g if g in gr.GRADES else ""
    return out


def arm_rows(gen: str, arm: str | None, lang: str, *, price_only: bool = False) -> list[Row]:
    if arm is None:
        answers = an.done("battery", gen)
        said = {qid: gr.sentences(a["text"]) for qid, a in answers.items()}
        out: list[Row] = [(qid, ss, [True] * len(ss), 0.0) for qid, ss in said.items()]
    else:
        file = ck.path("battery", arm, gen)
        if not file.exists():
            return []
        out = []
        for r in map(json.loads, file.read_text(encoding="utf-8").splitlines()):
            added = r.get("check_seconds")
            if arm == "judge1b":
                added = float(r["claim_seconds"]) + float(r["judge_seconds"])
            out.append((r["qid"], r["sentences"], [v == "EMIT" for v in r["verdicts"]], added))
    out = [o for o in out if o[0].endswith(f"-{lang}")]
    if price_only:
        priced: list[Row] = []
        for qid, ss, kept, t in out:
            pairs = [(s, k) for s, k in zip(ss, kept, strict=True) if is_price(s)]
            priced.append((qid, [s for s, _ in pairs], [k for _, k in pairs], t))
        out = priced
    return out


def graders() -> dict[str, Any]:
    """How the grades were settled, the graders' first-pass agreement per generator, and yours on
    the random checks where you graded them."""
    key = json.loads((gr.OUT / "key.json").read_text(encoding="utf-8"))
    settled = gr.consensus()
    first = {g: gr.done(g) for g in gr.GRADERS}
    yours: dict[str, str] = {}
    file = gr.OUT / "leftover.csv"
    if file.exists():
        with file.open(encoding="utf-8") as f:
            yours = {r["item"]: r["correct"].strip().lower() for r in csv.DictReader(f)}
    out: dict[str, Any] = {
        "graders": list(gr.GRADERS),
        "settled": {"first": 0, "again": 0, "unsettled": 0},
    }
    for v in settled.values():
        out["settled"][v["by"] if v else "unsettled"] += 1
    for gen in gr.GENERATORS:
        codes = [k for k, v in key.items() if gen in v["answers"]]
        sure = [k for k in codes if all(first[g].get(k) in {"yes", "no"} for g in gr.GRADERS)]
        same = sum(len({first[g][k] for g in gr.GRADERS}) == 1 for k in sure)
        checked = [k for k in codes if yours.get(k) in {"yes", "no"} and settled[k]]
        out[gen] = {
            "sentences": len(codes),
            "first_pass_agree": same,
            "first_pass_both_sure": len(sure),
            "your_checks": len(checked),
            "your_checks_agree": sum(yours[k] == settled[k]["grade"] for k in checked),
        }
    return out


def report() -> dict[str, Any]:
    grades = final_grades()
    out: dict[str, Any] = {"hardware": HARDWARE, "arm1_judge": ARM1_JUDGE, "grading": graders(),
                           "generators": {}}  # fmt: skip
    for gen in gr.GENERATORS:
        table: dict[str, Any] = {}
        for lang in ("fr", "en"):
            for scope, price_only in (("all", False), ("price", True)):
                arms = {}
                for name, arm in ARMS.items():
                    rows = arm_rows(gen, arm, lang, price_only=price_only)
                    if rows:
                        arms[name] = tally(rows, grades)
                        if price_only:  # an answer with no price sentence is not an empty answer
                            del arms[name]["empty"]
                table[f"{lang} {scope}"] = arms
        both = {}
        for name, arm in ARMS.items():
            if arm == "claims-020":  # the French half only: not comparable across both halves
                continue
            rows = arm_rows(gen, arm, "fr") + arm_rows(gen, arm, "en")
            if rows:
                both[name] = tally(rows, grades)
        table["both"] = both
        if {"1 other model", "3 claim check then 1B"} <= set(both):
            table["reading_costs_less_for_same_safety"] = reading(both)
        out["generators"][gen] = table
    REPORT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(report()["generators"], indent=1)[:4000])
