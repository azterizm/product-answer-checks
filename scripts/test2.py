"""Test 2's training data (docs/METHOD2.md §1–§5): made-up products, their questions, two models'
answers, Flash's grades, and the judge prompts and targets they give.

  uv run python scripts/test2.py build      # → test2/products.jsonl, questions.jsonl, build.json
  uv run python scripts/test2.py answer     # Flash through the router, resumable
  uv run --with modal modal run scripts/test2_modal.py::answers       # Qwen 2.5 7B, Modal L4
  uv run python scripts/test2.py sheet      # the blind sheet of both answers
  uv run python scripts/test2.py grade      # Flash grades it, one call per question, resumable
  uv run python scripts/test2.py record     # the judge prompts, verifier 0.4.0 (isolated)
  uv run python scripts/test2.py dataset    # → results/raw/test2/, test2/dataset.json
  uv run python scripts/test2.py validation # V1, after the judges' labels (judge1b.py::labels)

The listings are ours (no Icecat text), but answers, sheet and prompts stay in ``results/raw/``
(git-ignored) as in test 1; counts are committed under ``test2/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import answer as an  # type: ignore[import-not-found]  # noqa: E402
import check as ck  # type: ignore[import-not-found]  # noqa: E402
import grade as gr  # type: ignore[import-not-found]  # noqa: E402
import judge as jd  # type: ignore[import-not-found]  # noqa: E402
import router  # type: ignore[import-not-found]  # noqa: E402

from product_answer_checks import battery as bt  # noqa: E402
from product_answer_checks import sheets as sh  # noqa: E402
from product_answer_checks import synthetic as sy  # noqa: E402

COMMITTED = ROOT / "test2"
RAW = ROOT / "results/raw/test2"
GRADING = RAW / "grading"
SET = "test2"  # answers in results/raw/answers/test2.<generator>.jsonl
GENERATORS = ("flash", "qwen7b")
SHEET_SEED = 20261104
WORKERS = 4
# Stop 64's training data, in the verifier's results (local, git-ignored there).
LEGAL = ROOT.parent / "legal-rag-verifier/results/distill/students/train.jsonl"
# Stop 64's rate: two epochs of its 18,152 clauses (76,478,800 characters of prompt) in 1,873 s
# on one H100, start-up included (the verifier's GRID, stop 64; its Modal ledger).
CHARS_PER_SECOND = 2 * 76_478_800 / 1873
H100_PER_SECOND = 3.95 * 1.15 / 3600
LIMIT_USD = 3.50  # METHOD2 §5
STOP_USD = 3.75
_lock = threading.Lock()


# ---------------------------------------------------------------------- products and questions


def products() -> list[dict[str, Any]]:
    where = sy.split()
    out = []
    for pid in sy.IDS:
        en, fr = sy.sheet(pid, "en"), sy.sheet(pid, "fr")
        out.append(
            {
                "id": pid,
                "split": where[pid],
                "category": sh.category(en),
                "brand": sh.brand(en),
                "title": sh.title(en),
                "energy": sh.energy_features(en) is not None,
                "listing_sha256": {
                    "en": bt.sha256(sh.listing(en)),
                    "fr": bt.sha256(sh.listing(fr)),
                },
            }
        )
    return out


def records(questions: Sequence[bt.Question]) -> list[dict[str, Any]]:
    where = sy.split()
    return [{**q.record(), "split": where[q.product]} for q in questions]


def build() -> dict[str, Any]:
    questions = sy.questions()
    COMMITTED.mkdir(exist_ok=True)
    for name, rows in (("products", products()), ("questions", records(questions))):
        text = "".join(bt.dumps(r) + "\n" for r in rows)
        (COMMITTED / f"{name}.jsonl").write_text(text, encoding="utf-8")
    where = sy.split()
    out = {
        "products": Counter(where.values()),
        "questions": len(questions),
        "types": Counter(q.type for q in questions),
        "questions_by_split": Counter(where[q.product] for q in questions),
        "triples": len(sy.triples()),
        "prompt_sha256": bt.sha256(bt.PROMPT),
    }
    (COMMITTED / "build.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


def questions() -> list[bt.Question]:
    """The questions, rebuilt, checked against the committed rows (as ``answer.sealed``)."""
    rebuilt = sy.questions()
    stored = (COMMITTED / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    if [json.loads(x) for x in stored] != [json.loads(bt.dumps(r)) for r in records(rebuilt)]:
        raise SystemExit("test2/questions.jsonl does not match a fresh build; run build first")
    return rebuilt


# ---------------------------------------------------------------------- answers (Flash)


def answer(limit: int = 0) -> None:
    """Flash's answer to each question; an error is retried once, then left missing (as test 1)."""
    an.env()
    have = an.done(SET, "flash")
    todo = [q for q in questions() if q.qid not in have][: limit or None]
    print(f"flash: {len(todo)} to answer", flush=True)
    failed: list[str] = []

    def one(q: bt.Question) -> None:
        error = ""
        for _ in range(2):
            t0 = time.perf_counter()
            try:
                out = router.chat(router.GEMINI, an.messages(q))
            except router.RouterError as exc:
                error = repr(exc)[:300]
                continue
            text = (out.get("content") or "").strip()
            record = {"qid": q.qid, "text": text, "model": router.GEMINI}
            record["seconds"] = round(time.perf_counter() - t0, 3)
            an.append(SET, "flash", [record])
            return
        failed.append(f"{q.qid}: {error}")

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(one, todo))
    for line in failed:
        print(f"MISSING {line}", flush=True)
    print(f"flash: {len(an.done(SET, 'flash'))} answers", flush=True)


# ---------------------------------------------------------------------- grades (Flash)


def sheet() -> dict[str, int]:
    """The blind sheet, as test 1's: per question both answers lettered at random, each distinct
    sentence numbered once."""
    qs = {q.qid: q for q in questions()}
    answers = {g: an.done(SET, g) for g in GENERATORS}
    rng = random.Random(SHEET_SEED)
    order = sorted(qs)
    rng.shuffle(order)
    lines = ["# Test 2: sentences to grade", "", gr.RULE, ""]
    key: dict[str, Any] = {}
    counts = {"questions": len(order), "items": 0, "repeats": 0}
    for n, qid in enumerate(order, 1):
        q = qs[qid]
        texts: dict[str, list[str]] = {}
        for g in GENERATORS:
            if qid in answers[g]:
                texts.setdefault(answers[g][qid]["text"].strip(), []).append(g)
        group = sorted(texts)
        rng.shuffle(group)
        block = [f"## Q{n:04d}. {q.question}", "", "Source:", "", "```", q.premise, "```", ""]
        first: dict[str, str] = {}
        for letter, text in zip("ABCD", group, strict=False):
            block += [f"**Answer {letter}**", ""]
            for k, s in enumerate(gr.sentences(text), 1):
                code = f"Q{n:04d}{letter}{k}"
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
    GRADING.mkdir(parents=True, exist_ok=True)
    (GRADING / "sheet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (GRADING / "key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False) + "\n", "utf-8")
    return counts


def blocks() -> dict[str, str]:
    text = (GRADING / "sheet.md").read_text(encoding="utf-8")
    parts = re.split(r"(?m)^## (Q\d+)\. ", text)[1:]
    return {q: f"{q}. {body.strip()}" for q, body in zip(parts[::2], parts[1::2], strict=True)}


def grades() -> dict[str, str]:
    path = GRADING / "grader-flash.jsonl"
    out: dict[str, str] = {}
    if path.exists():
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
            out |= row["grades"]
    return out


def grade() -> None:
    """Flash, test 1's rule and prompt, one call per question, appended as each returns."""
    an.env()
    have = grades()
    jobs = []
    for q, block in blocks().items():
        items = gr._ITEM.findall(block)
        if items and not set(items) <= set(have):
            user = gr.USER.format(rule=gr.RULE, block=block)
            messages = [{"role": "system", "content": gr.SYSTEM}, {"role": "user", "content": user}]
            jobs.append((q, items, messages))
    print(f"flash grades: {len(jobs)} questions", flush=True)
    stopped: list[str] = []

    def one(job: tuple[str, list[str], list[dict[str, str]]]) -> None:
        q, items, messages = job
        for _ in range(3):
            if stopped:
                return
            try:
                text = router.chat(router.GEMINI, messages)["content"]
            except router.RouterError as exc:
                stopped.append(repr(exc)[:300])
                return
            try:
                parsed = gr.parse(jd.extract_json(text), items, again=False)
            except (ValueError, json.JSONDecodeError):
                continue
            row = {"q": q, "grades": parsed, "model": router.GEMINI}
            with _lock, (GRADING / "grader-flash.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            return
        print(f"{q}: no valid reply, asked again on the next run", flush=True)

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(one, jobs))
    if stopped:
        print(f"STOPPED: {stopped[0]}", flush=True)
    print(json.dumps({"graded": len(grades()), "items": len(json.loads(_key_text()))}))


def _key_text() -> str:
    return (GRADING / "key.json").read_text(encoding="utf-8")


def sentence_grades() -> dict[tuple[str, str], str]:
    """(question id, sentence) → Flash's grade."""
    have = grades()
    return {(v["qid"], v["text"]): have[k] for k, v in json.loads(_key_text()).items() if k in have}


# ---------------------------------------------------------------------- judge prompts


def record() -> None:
    """``record2.py`` on 0.4.0, isolated, over every answer of both generators."""
    qs = {q.qid: q for q in questions()}
    graded = sentence_grades()
    jobs = []
    for g in GENERATORS:
        for qid, a in sorted(an.done(SET, g).items()):
            said = ck.sentences(a["text"])
            jobs.append(
                {"qid": qid, "generator": g, "premise": qs[qid].premise,
                 "query": qs[qid].question, "sentences": said,
                 "grades": [graded.get((qid, s.strip())) for s in said]}
            )  # fmt: skip
    stdin = "".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs)
    out = subprocess.run(  # noqa: S603
        [*ck.isolated(ck.TOTALS_RELEASE), str(ROOT / "scripts/record2.py")],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    rows = [json.loads(x) for x in out.stdout.splitlines()]
    if any(r["config"].get("totals") != "line" for r in rows):
        raise SystemExit("the recorder's verifier is not running totals")
    by_key = {(j["qid"], j["generator"]): j for j in jobs}
    RAW.mkdir(parents=True, exist_ok=True)
    with (RAW / "prompts.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            j = by_key[r["qid"], r["generator"]]
            f.write(json.dumps({**r, "sentences": j["sentences"], "grades": j["grades"]}) + "\n")
    print(f"{len(rows)} answers recorded → {RAW / 'prompts.jsonl'}")


# ---------------------------------------------------------------------- the training set


def reply(label: str) -> str:
    """The target, as the verifier's ``students.reply`` writes it."""
    return json.dumps({"label": label, "passage": 0})


def examples(
    rows: Sequence[Mapping[str, Any]], where: Mapping[int, str], product: Mapping[str, int]
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    """Prompts and targets by split (METHOD2 §4), each prompt once, and what was left out."""
    out: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    seen: set[str] = set()
    stats: Counter[str] = Counter()
    for r in rows:
        split = where[product[r["qid"]]]
        for sentence_prompts, grade, first in zip(
            r["prompts"], r["grades"], r["claims"], strict=True
        ):
            stats[f"sentences {r['generator']}"] += 1
            if first != "EMIT":
                stats["sentences the claim check rolls back"] += 1
                continue
            if grade == "yes":
                label = "supported"
            elif grade == "no" and len(sentence_prompts) == 1:
                label = "contradicted"
            else:
                why = "wrong, several clauses" if grade == "no" else f"grade {grade or 'missing'}"
                stats[f"left out: {why}"] += 1
                continue
            for user in sentence_prompts:
                sha = jd.prompt_key(user)
                if sha in seen:
                    stats["repeated prompt"] += 1
                    continue
                seen.add(sha)
                out[split].append(
                    {
                        "sha": sha,
                        "source": f"{r['qid']}/{r['generator']}",
                        "messages": [
                            {"role": "system", "content": jd.SYSTEM},
                            {"role": "user", "content": user},
                            {"role": "assistant", "content": reply(label)},
                        ],
                    }
                )
                stats[f"{split} {label}"] += 1
                stats[f"{split} {label} {r['generator']}"] += 1
    return out, stats


def plan(legal_chars: int, product_chars: int) -> dict[str, Any]:
    """METHOD2 §5's cost rule: two epochs with the product clauses twice, else once, else one
    epoch; the estimate from stop 64's rate."""
    for epochs, copies in ((2, 2), (2, 1), (1, 1)):
        seconds = epochs * (legal_chars + copies * product_chars) / CHARS_PER_SECOND
        usd = seconds * H100_PER_SECOND
        if usd <= LIMIT_USD or (epochs, copies) == (1, 1):
            break
    return {
        "epochs": epochs,
        "product_copies": copies,
        "estimate_seconds": round(seconds),
        "estimate_usd": round(usd, 2),
        "stop_seconds": math.floor(STOP_USD / H100_PER_SECOND),
    }


def dataset() -> dict[str, Any]:
    rows = [json.loads(x) for x in (RAW / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    product = {q.qid: q.product for q in questions()}
    out, stats = examples(rows, sy.split(), product)
    legal = LEGAL.read_bytes()
    legal_rows = legal.decode("utf-8").splitlines()
    legal_chars = sum(len(json.loads(x)["messages"][1]["content"]) for x in legal_rows)
    product_chars = sum(len(e["messages"][1]["content"]) for e in out["train"])
    chosen = plan(legal_chars, product_chars)
    folder = RAW / "run3"
    folder.mkdir(parents=True, exist_ok=True)
    product_lines = [json.dumps(e, ensure_ascii=False) + "\n" for e in out["train"]]
    with (folder / "train.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(x + "\n" for x in legal_rows)
        for _ in range(chosen["product_copies"]):
            f.writelines(product_lines)
    validation = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in out["validation"])
    (folder / "validation.jsonl").write_text(validation, encoding="utf-8")
    summary = {
        "legal": {"file": "legal-rag-verifier results/distill/students/train.jsonl",
                  "sha256": hashlib.sha256(legal).hexdigest(), "clauses": len(legal_rows),
                  "prompt_characters": legal_chars},
        "product": {"train": len(out["train"]), "validation": len(out["validation"]),
                    "prompt_characters": product_chars},
        "plan": chosen,
        "train_file_sha256": hashlib.sha256((folder / "train.jsonl").read_bytes()).hexdigest(),
        **dict(sorted(stats.items())),
    }  # fmt: skip
    (COMMITTED / "dataset.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


# ---------------------------------------------------------------------- V1


def labels(run: str) -> dict[str, dict[str, Any]]:
    path = RAW / f"labels.{run}.jsonl"
    if not path.exists():
        return {}
    return {r["sha"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def validation(runs: Sequence[str] = ("run2", "run3")) -> dict[str, Any]:
    """V1: on the validation clauses, how often each judge gives the teacher's label."""
    rows = [json.loads(x) for x in (RAW / "run3/validation.jsonl").read_text("utf-8").splitlines()]
    teacher = {r["sha"]: json.loads(r["messages"][2]["content"])["label"] for r in rows}
    out: dict[str, Any] = {"validation_clauses": len(teacher), "teacher": Counter(teacher.values())}
    for run in runs:
        mine = labels(run)
        both = [k for k in teacher if k in mine]
        out[run] = {
            "labelled": len(both),
            "same_label": sum(mine[k]["label"] == teacher[k] for k in both),
            "labels": Counter(mine[k]["label"] for k in both),
            "differ": Counter(
                f"teacher {teacher[k]} / judge {mine[k]['label']}"
                for k in both
                if mine[k]["label"] != teacher[k]
            ),
        }
    (COMMITTED / "validation.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = ["build", "answer", "sheet", "grade", "record", "dataset", "validation"]
    parser.add_argument("command", choices=commands)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.command == "answer":
        answer(args.limit)
    elif args.command == "grade":
        grade()
    elif args.command == "record":
        record()
    else:
        out = {"build": build, "sheet": sheet, "dataset": dataset, "validation": validation}
        print(json.dumps(out[args.command](), indent=1))


if __name__ == "__main__":
    main()
