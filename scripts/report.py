"""Test 1's report (docs/METHOD.md §8): per generator, half and arm, the wrong sentences kept, the
right sentences removed, empty answers and seconds added; price sentences apart; the reading rule.
Test 2's arms 4 and 4t (docs/METHOD2.md §6) are read on the same answers and grades: R1 and R2.

  uv run python scripts/report.py     # → results/report.json (counts only)
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
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
    "2t claim check with totals": "claims76",
    "3 claim check then 1B": "judge1b",
    "3t claim check with totals then 1B": "judge1b76",
    "4 claim check then product-trained 1B": "judge1b-run3",
    "4t claim check with totals then product-trained 1B": "judge1b76-run3",
}
ARM1_JUDGE = {"opus46": "gemini-3.8-flash-high (router)", "flash": "qwen27-ar4 (one L40S)"}
HARDWARE = {
    "1 other model": "Opus 4.6's answers: Flash on the router; Flash's: Qwen 27B on one L40S",
    "2 claim check": "CPU (Apple Mac)",
    "2b claim check 0.2.0": "CPU (Apple Mac)",
    "2t claim check with totals": "CPU (Apple Mac)",
    "3 claim check then 1B": "CPU, then one L40S",
    "3t claim check with totals then 1B": "CPU, then one L40S",
    "4 claim check then product-trained 1B": "CPU, then one L40S",
    "4t claim check with totals then product-trained 1B": "CPU, then one L40S",
}
ARM_4T = "4t claim check with totals then product-trained 1B"
PRICE = re.compile(r"€|\bEUR\b|euro", re.IGNORECASE)
MARGIN = 2  # the reading rule: at most 2 more wrong sentences kept than arm 1
# A euro amount as the answers write it, read here and not by the checker (METHOD, Changes
# 2026-10-23): "€1,234.56", "€ 908.98", "EUR 60.98", "1 234,56 €", "60,98 €", "567.98 €".
_AMOUNT = (
    r"\d{1,3}(?:,\d{3})+(?:\.\d{2})?"
    r"|\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:,\d{2})?"
    r"|\d+(?:[.,]\d{2})?"
)
EURO = re.compile(
    rf"(?:€|\bEUR)\s?(?P<lead>{_AMOUNT})|(?<![\d.,])(?P<trail>{_AMOUNT})\s?(?:€|EUR\b)"
)


def is_price(sentence: str) -> bool:
    """A price sentence (METHOD, Changes): it contains "€", "EUR" or "euro"."""
    return bool(PRICE.search(sentence))


def euro_cents(sentence: str) -> set[int]:
    """The euro amounts a sentence states, in cents."""
    out = set()
    for m in EURO.finditer(sentence):
        text = re.sub(r"[ \u00a0\u202f]", "", m["lead"] or m["trail"])
        text = text.replace(",", ".") if re.fullmatch(r"\d+,\d{2}", text) else text.replace(",", "")
        out.add(int(Decimal(text) * 100))
    return out


def offer_totals(offers: Iterable[Mapping[str, Any]]) -> set[int]:
    """Each offer's price plus its shipping, in cents, where shipping is not free."""
    return {o["price"] + o["shipping"] for o in offers if o["shipping"]}


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


def reading(arms: Mapping[str, Mapping[str, Any]], arm: str = "3 claim check then 1B") -> bool:
    """Arm 3 "costs less for the same safety": at most MARGIN more wrong kept than arm 1, and fewer
    right removed. For arm 3t it is reported as information only (METHOD, Changes 2026-10-23)."""
    one, three = arms["1 other model"], arms[arm]
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
            if arm.startswith("judge1b"):
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


def read_checked(arm: str, gen: str) -> dict[str, dict[str, Any]]:
    file = ck.path("battery", arm, gen)
    return {r["qid"]: r for r in map(json.loads, file.read_text(encoding="utf-8").splitlines())}


def totals_block(gen: str, grades: Mapping[tuple[str, str], str]) -> dict[str, Any] | None:
    """Arm 2t against arm 2 (METHOD, Changes 2026-10-23): the sums gap and what came back, and
    the wrong sentences arm 2t keeps that arm 2 removed."""
    if not ck.path("battery", "claims76", gen).exists():
        return None
    rows = (ROOT / "battery/questions.jsonl").read_text(encoding="utf-8").splitlines()
    offers = {q["qid"]: q["offers"] for q in map(json.loads, rows)}
    two, two_t = read_checked("claims", gen), read_checked("claims76", gen)
    gap, back, not_back, wrong_kept, right_lost = [], [], [], [], []
    for qid, r in sorted(two.items()):
        totals = offer_totals(offers[qid])
        for i, s in enumerate(r["sentences"]):
            before, after = r["verdicts"][i] == "EMIT", two_t[qid]["verdicts"][i] == "EMIT"
            grade = grades.get((qid, s))
            if grade == "yes" and not before and euro_cents(s) & totals:
                gap.append(f"{qid}#{i}")
                if after:
                    back.append(f"{qid}#{i}")
                else:
                    not_back.append({"sentence": f"{qid}#{i}",
                                     "reasons": two_t[qid]["reasons"][i],
                                     "failed": two_t[qid]["failed"][i]})  # fmt: skip
            if grade == "no" and after and not before:
                wrong_kept.append(f"{qid}#{i}")
            if grade == "yes" and before and not after:
                right_lost.append(f"{qid}#{i}")
    return {
        "sums_gap": len(gap),
        "recovered": len(back),
        "not_recovered": not_back,
        "wrong_newly_kept": wrong_kept,
        "right_newly_removed": right_lost,
        "verdicts_changed": sum(
            a != b
            for qid, r in two.items()
            for a, b in zip(r["verdicts"], two_t[qid]["verdicts"], strict=True)
        ),
    }


def sign_test(only_a: int, only_b: int) -> float:
    """Exact two-sided sign test on the sentences only one of two arms gets right."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2.0**n
    return round(min(1.0, 2 * tail), 4)


def against_3t(gen: str, grades: Mapping[tuple[str, str], str]) -> dict[str, Any] | None:
    """R2 (METHOD2 §6): 4t against 3t, the right sentences only one of them removes."""
    if not ck.path("battery", "judge1b76-run3", gen).exists():
        return None
    four, three = read_checked("judge1b76-run3", gen), read_checked("judge1b76", gen)
    only_4t, only_3t = [], []
    for qid, r in sorted(three.items()):
        for i, s in enumerate(r["sentences"]):
            if grades.get((qid, s)) != "yes":
                continue
            kept_3t, kept_4t = r["verdicts"][i] == "EMIT", four[qid]["verdicts"][i] == "EMIT"
            if kept_3t and not kept_4t:
                only_4t.append(f"{qid}#{i}")
            if kept_4t and not kept_3t:
                only_3t.append(f"{qid}#{i}")
    return {
        "right_removed_only_by_4t": only_4t,
        "right_removed_only_by_3t": len(only_3t),
        "sign_test_p": sign_test(len(only_4t), len(only_3t)),
    }


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
        if {"1 other model", "3t claim check with totals then 1B"} <= set(both):
            table["reading_for_3t_information_only"] = reading(
                both, "3t claim check with totals then 1B"
            )
        if {"1 other model", ARM_4T} <= set(both):
            table["test2_R1_4t_costs_less_for_same_safety"] = reading(both, ARM_4T)
        r2 = against_3t(gen, grades)
        if r2 is not None:
            table["test2_R2_4t_against_3t"] = r2
        block = totals_block(gen, grades)
        if block is not None:
            table["totals"] = block
        out["generators"][gen] = table
    REPORT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(report()["generators"], indent=1)[:4000])
