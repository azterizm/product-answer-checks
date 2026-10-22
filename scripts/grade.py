"""Test 1's grading (docs/METHOD.md §7 and Changes): the verifier's ``live68_grade.py`` flow,
copied at da40159 with the sheet built from this battery. Two graders; then what they still differ
on, and 30 random sentences per generator, go to you.

  uv run python scripts/grade.py sheet            # the blind sheet of arm 0's sentences
  uv run python scripts/grade.py grade opus55     # first pass, one call per question
  uv run python scripts/grade.py grade flash
  uv run python scripts/grade.py grade opus55 --again   # items they differ on, reason first
  uv run python scripts/grade.py grade flash --again
  uv run python scripts/grade.py leftover         # → results/raw/grading/leftover.md and .csv

The sheet quotes the listings, so it stays in ``results/raw/grading/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import answer as an  # type: ignore[import-not-found]  # noqa: E402
import judge as jd  # type: ignore[import-not-found]  # noqa: E402
import router  # type: ignore[import-not-found]  # noqa: E402

from legal_rag_verifier.segmenter import split_sentences  # noqa: E402

OUT = ROOT / "results/raw/grading"
GRADES = ("yes", "no", "unsure")
# Opus 5.5 was down at the router (HTTP 503): Flash and Qwen 27B grade, by your choice (METHOD).
# Qwen 27B runs on Modal (scripts/qwen27.py), with this file's prompts and parsing.
GRADERS: tuple[str, ...] = ("flash", "qwen27")
MODELS = {"opus55": router.OPUS55, "flash": router.GEMINI, "qwen27": "qwen27-ar4 (Modal L40S)"}
GENERATORS = ("flash", "opus46")
SHEET_SEED = 20261022
CHECK_SEED = 20261030  # the 30 random sentences per generator you grade (METHOD §7)
CHECK_PER_GENERATOR = 30
WORKERS = 4
RULE = (
    'For each numbered sentence, grade it against the product listings and offers shown: "yes" if '
    'every claim in the sentence matches them, "no" if any claim is not supported by them or '
    'contradicts them, "unsure" otherwise. A struck-through price ("was …" or "au lieu de …") is '
    "not the current price. A total that leaves out shipping when the question asks for the "
    'delivered price is "no". A sentence with no claim is "yes" unless it misstates the listings.'
)
SYSTEM = "You grade sentences of answers against a source. Reply with a single JSON object."
USER = (
    "{rule}\n\n{block}\n\nReturn a JSON object with one key per numbered sentence (for example "
    '"Q001A1") and the value "yes", "no" or "unsure".'
)
AGAIN = (
    "{rule}\n\n{block}\n\nGrade only these sentences: {items}.\n\nReturn a JSON object with one "
    'key per listed sentence. Each value is an object with "reason" (one sentence, written before '
    'you decide) and "grade" ("yes", "no" or "unsure").'
)
_ITEM = re.compile(r"\*\*(Q\d+[A-Z]\d+)\.\*\*")
_lock = threading.Lock()


def sentences(text: str) -> list[str]:
    return [s.text.strip() for s in split_sentences(text) if s.text.strip()]


# ---------------------------------------------------------------------- the sheet


def build() -> dict[str, Any]:
    """The blind sheet: per question, the generators' answers lettered at random; each distinct
    sentence numbered once (a repeat points to its first number)."""
    questions = {q.qid: q for q in an.sealed("battery")}
    answers = {g: an.done("battery", g) for g in GENERATORS}
    rng = random.Random(SHEET_SEED)
    order = sorted(questions)
    rng.shuffle(order)
    lines = ["# Test 1: sentences to grade", "", RULE, ""]
    key: dict[str, Any] = {}
    counts = {"questions": len(order), "items": 0, "repeats": 0}
    for n, qid in enumerate(order, 1):
        q = questions[qid]
        texts: dict[str, list[str]] = {}
        for g in GENERATORS:
            if qid in answers[g]:
                texts.setdefault(answers[g][qid]["text"].strip(), []).append(g)
        group = sorted(texts)
        rng.shuffle(group)
        block = [f"## Q{n:03d}. {q.question}", "", "Source:", "", "```", q.premise, "```", ""]
        first: dict[str, str] = {}
        for letter, text in zip("ABCD", group, strict=False):
            block += [f"**Answer {letter}**", ""]
            for k, s in enumerate(sentences(text), 1):
                code = f"Q{n:03d}{letter}{k}"
                if s in first:
                    block.append(f"- (= {first[s]}) {s}")
                    counts["repeats"] += 1
                else:
                    first[s] = code
                    block.append(f"- **{code}.** {s}")
                    key[code] = {"qid": qid, "text": s, "answers": texts[text]}
                    counts["items"] += 1
            block.append("")
        lines += block
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sheet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False) + "\n", "utf-8")
    return counts


def blocks() -> dict[str, str]:
    text = (OUT / "sheet.md").read_text(encoding="utf-8")
    parts = re.split(r"(?m)^## (Q\d+)\. ", text)[1:]
    return {q: f"{q}. {body.strip()}" for q, body in zip(parts[::2], parts[1::2], strict=True)}


# ---------------------------------------------------------------------- graders


def path(grader: str, *, again: bool = False) -> Path:
    return OUT / f"grader-{grader}{'.again' if again else ''}.jsonl"


def done(grader: str, *, again: bool = False) -> dict[str, str]:
    out: dict[str, str] = {}
    file = path(grader, again=again)
    if file.exists():
        for row in map(json.loads, file.read_text(encoding="utf-8").splitlines()):
            out |= row["grades"]
    return out


def open_items() -> list[str]:
    if len(GRADERS) == 1:  # one grader: what it marks unsure goes to you, not to a second pass
        return []
    first = [done(g) for g in GRADERS]
    return [k for k in first[0] if not (first[0][k] == first[1].get(k) != "unsure")]


def parse(reply: Mapping[str, Any], items: Sequence[str], *, again: bool) -> dict[str, str]:
    out = {}
    for k in items:
        value = reply.get(k)
        grade = value.get("grade", "") if again and isinstance(value, Mapping) else value
        grade = str(grade or "").strip().lower()
        if grade not in GRADES:
            raise ValueError(f"{k} has no valid grade")
        out[k] = grade
    return out


def todo(grader: str, *, again: bool = False) -> list[tuple[str, list[str], list[dict[str, str]]]]:
    """The questions ``grader`` still has to grade, each with its items and its messages."""
    have = done(grader, again=again)
    wanted = set(open_items()) if again else None
    out = []
    for q, block in blocks().items():
        items = [k for k in _ITEM.findall(block) if wanted is None or k in wanted]
        if items and not set(items) <= set(have):
            user = (
                AGAIN.format(rule=RULE, block=block, items=", ".join(items))
                if again
                else USER.format(rule=RULE, block=block)
            )
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
            out.append((q, items, messages))
    return out


def save(grader: str, q: str, items: list[str], text: str, *, again: bool = False) -> bool:
    """Parse one reply and append its grades; False when it is not a valid reply."""
    try:
        reply = jd.extract_json(text)
        grades = parse(reply, items, again=again)
    except (ValueError, json.JSONDecodeError):
        return False
    row: dict[str, Any] = {"q": q, "grades": grades, "model": MODELS[grader]}
    if again:
        row["reasons"] = {k: str((reply.get(k) or {}).get("reason", ""))[:400] for k in items}
    with _lock, path(grader, again=again).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


def grade(grader: str, *, again: bool = False) -> None:
    """``grader``'s share through the router, one call per question, appended as each returns. A
    rate limit past the router's retries ends the run cleanly; a rerun asks what is left."""
    if grader == "qwen27":
        raise SystemExit("Qwen 27B grades on Modal: scripts/qwen27.py")
    an.env()
    model = MODELS[grader]
    jobs = todo(grader, again=again)
    print(f"{grader} ({model}){' again' if again else ''}: {len(jobs)} questions", flush=True)
    stopped: list[str] = []

    def one(job: tuple[str, list[str], list[dict[str, str]]]) -> None:
        q, items, messages = job
        for _ in range(3):
            if stopped:
                return
            try:
                text = router.chat(model, messages)["content"]
            except (router.BudgetExceededError, router.RouterError) as exc:
                stopped.append(repr(exc)[:300])
                return
            if save(grader, q, items, text, again=again):
                return
        print(f"{q}: no valid reply, asked again on the next run", flush=True)

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(one, jobs))
    left = len(todo(grader, again=again))
    if stopped:
        print(f"STOPPED: {stopped[0]}; {left} questions left", flush=True)
    print(json.dumps({"left": left, "ledger": router.ledger().get(model, {})}))


def consensus() -> dict[str, dict[str, Any]]:
    """Agreed grades: the first pass where both say the same (not unsure), else the second. With
    one grader (as the verifier's stop 71), its grade unless unsure."""
    first = [done(g) for g in GRADERS]
    second = [done(g, again=True) for g in GRADERS]
    key = json.loads((OUT / "key.json").read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for k in key:
        if len(GRADERS) == 1:
            g = first[0].get(k)
            if g is None:
                raise SystemExit(f"{k} has not been graded yet")
            out[k] = {"grade": g, "by": "first"} if g != "unsure" else {}
            continue
        a, b = first[0].get(k), first[1].get(k)
        if a is None or b is None:
            raise SystemExit(f"{k} has not been graded by both yet")
        if a == b != "unsure":
            out[k] = {"grade": a, "by": "first"}
            continue
        a2, b2 = second[0].get(k), second[1].get(k)
        out[k] = {"grade": a2, "by": "again"} if a2 is not None and a2 == b2 != "unsure" else {}
    return out


def checks(key: Mapping[str, Mapping[str, Any]]) -> set[str]:
    """30 random sentences per generator for your check on the graders (METHOD §7)."""
    rng = random.Random(CHECK_SEED)
    out: set[str] = set()
    for g in GENERATORS:
        pool = sorted(k for k, v in key.items() if g in v["answers"] and k not in out)
        out |= set(rng.sample(pool, min(CHECK_PER_GENERATOR, len(pool))))
    return out


def leftover() -> None:
    settled = consensus()
    key = json.loads((OUT / "key.json").read_text(encoding="utf-8"))
    left = {k for k, v in settled.items() if not v}
    sample = checks(key)
    lines = [
        "# Sentences left for you",
        "",
        RULE,
        "",
        "Grade each listed sentence in leftover.csv.",
        "",
    ]
    for block in blocks().values():
        mine = [k for k in _ITEM.findall(block) if k in left | sample]
        if not mine:
            continue
        kept = [
            line
            for line in block.splitlines()
            if not line.startswith("- **Q") or any(f"**{k}.**" in line for k in mine)
        ]
        lines += [f"## {kept[0]}", *kept[1:], ""]
    (OUT / "leftover.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (OUT / "leftover.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item", "why", "correct", "note"])
        writer.writeheader()
        for k in sorted(left | sample):
            why = (
                "grader unsure"
                if k in left and len(GRADERS) == 1
                else "graders differ"
                if k in left
                else "random check"
            )
            writer.writerow({"item": k, "why": why, "correct": "", "note": ""})
    by = {"first": 0, "again": 0}
    for v in settled.values():
        if v:
            by[v["by"]] += 1
    print(json.dumps({"items": len(settled), **by, "left": len(left), "random check": len(sample)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sheet", "grade", "leftover"])
    parser.add_argument("grader", nargs="?", choices=GRADERS)
    parser.add_argument("--again", action="store_true")
    args = parser.parse_args()
    if args.command == "sheet":
        print(json.dumps(build()))
    elif args.command == "grade":
        if args.grader is None:
            parser.error("grade needs a grader: opus55 or flash")
        grade(args.grader, again=args.again)
    else:
        leftover()


if __name__ == "__main__":
    main()
