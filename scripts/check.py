"""Test 1's arms 1, 2 and 2b on the answers (docs/METHOD.md §6). A sentence that is not emitted is
dropped, as in the verifier's stop 71.

  uv run python scripts/check.py claims battery flash        # arm 2: the pinned build, CPU
  uv run python scripts/check.py claims-020 battery flash    # arm 2b: 0.2.0 from PyPI, French half
  uv run python scripts/check.py claims76 dev flash          # arm 2t: totals, verifier stop 76
  uv run python scripts/check.py judge battery flash         # arm 1: Opus 4.6 judges Flash

Checked answers stay in ``results/raw/checked/`` (git-ignored), with each sentence's verdict.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import answer as an  # type: ignore[import-not-found]  # noqa: E402
import judge as jd  # type: ignore[import-not-found]  # noqa: E402

from legal_rag_verifier.premise import Premise  # noqa: E402
from legal_rag_verifier.segmenter import split_sentences  # noqa: E402
from legal_rag_verifier.verifier import Verifier  # noqa: E402

OUT = ROOT / "results/raw/checked"
OTHER = {"flash": "opus46", "opus46": "flash"}
OLD = "legal-rag-verifier==0.2.0"
PINNED = "66450a6"  # the build the runs used, released as 0.3.0; 0.2.0 reports its version wrongly
# Arms 2t and 3t (METHOD, Changes 2026-10-23): the verifier's stop 76 build, the one the runs used,
# released unchanged as 0.4.0.
TOTALS = "9ab6603"
TOTALS_RELEASE = "legal-rag-verifier==0.4.0"
_lock = threading.Lock()


def path(set_: str, arm: str, name: str) -> Path:
    return OUT / f"{set_}.{arm}.{name}.jsonl"


def sentences(text: str) -> list[str]:
    return [s.text for s in split_sentences(text)]


def write(set_: str, arm: str, name: str, rows: list[dict[str, Any]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with path(set_, arm, name).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def checked(
    answer: Mapping[str, Any], said: Sequence[str], verdicts: Sequence[str]
) -> dict[str, Any]:
    kept = [s for s, v in zip(said, verdicts, strict=True) if v == "EMIT"]
    return {"qid": answer["qid"], "sentences": said, "verdicts": verdicts, "text": " ".join(kept)}


def isolated(requirement: str) -> list[str]:
    return ["uv", "run", "--isolated", "--no-project", "--with", requirement, "python"]


def claim_jobs(
    set_: str, name: str, *, french_only: bool = False
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    questions = {q.qid: q for q in an.sealed(set_)}
    answers = an.done(set_, name)
    jobs = [
        {"qid": qid, "premise": questions[qid].premise, "query": questions[qid].question,
         "sentences": sentences(a["text"])}
        for qid, a in sorted(answers.items())
        if not french_only or questions[qid].lang == "fr"
    ]  # fmt: skip
    return jobs, answers


def run_claimcheck(
    command: Sequence[str], jobs: Sequence[Mapping[str, Any]], *flags: str
) -> dict[str, dict[str, Any]]:
    """``scripts/claimcheck.py`` under ``command``, in a subprocess: question id → its output."""
    stdin = "".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs)
    result = subprocess.run(  # noqa: S603
        [*command, str(ROOT / "scripts/claimcheck.py"), *flags],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    return {out["qid"]: out for out in map(json.loads, result.stdout.splitlines())}


def claim_rows(
    jobs: Sequence[Mapping[str, Any]],
    answers: Mapping[str, Mapping[str, Any]],
    outs: Mapping[str, Mapping[str, Any]],
    build: str,
) -> list[dict[str, Any]]:
    return [
        {**checked(answers[j["qid"]], j["sentences"], outs[j["qid"]]["verdicts"]),
         "reasons": outs[j["qid"]]["reasons"], "failed": outs[j["qid"]].get("failed"),
         "check_seconds": outs[j["qid"]]["seconds"], "verifier": build}
        for j in jobs
    ]  # fmt: skip


def claims(set_: str, name: str, *, old: bool) -> None:
    """Arm 2 (pinned build) or 2b (0.2.0, French half only), in a subprocess so 2b is isolated."""
    jobs, answers = claim_jobs(set_, name, french_only=old)
    command = isolated(OLD) if old else [sys.executable]
    build = "0.2.0" if old else PINNED
    rows = claim_rows(jobs, answers, run_claimcheck(command, jobs), build)
    arm = "claims-020" if old else "claims"
    write(set_, arm, name, rows)
    print(f"{set_} {arm} {name}: {len(rows)} answers, verifier {build}")


def offer_totals(offers: Sequence[Mapping[str, Any]]) -> set[int]:
    """Each offer's price plus its shipping, in cents, where shipping is not free."""
    return {o["price"] + o["shipping"] for o in offers if o["shipping"]}


def only_totals_failed(failed: Sequence[str], totals: set[int]) -> bool:
    """Every failed claim of a sentence is a euro figure equal to an offer's price plus shipping."""
    return bool(failed) and all(
        v.startswith("EUR:") and Decimal(v[4:]) * 100 in totals for v in failed
    )


def changes(
    reference: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    offers: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[tuple[str, int]]]:
    """Sentences whose verdict changed, and those expected to change: rolled back by arm 2 with
    only offer totals failed (METHOD, Changes 2026-10-23)."""
    changed, expected = [], []
    for qid, ref in sorted(reference.items()):
        totals = offer_totals(offers[qid])
        verdicts = zip(ref["verdicts"], after[qid]["verdicts"], strict=True)
        for i, (before, now) in enumerate(verdicts):
            if before != now:
                changed.append((qid, i))
            if before == "ROLLBACK" and only_totals_failed(ref["failed"][i], totals):
                expected.append((qid, i))
    return {"changed": changed, "expected": expected}


def totals(set_: str, name: str) -> None:
    """Arm 2t: the stop 76 build with ``totals="line"``. First its control: the same build with
    totals off must give arm 2's verdicts (a fresh run of 0.3.0, and on the battery the stored arm
    2 file too); any difference stops the run. On dev, only the expected sentences may change."""
    jobs, answers = claim_jobs(set_, name)
    offers = {q.qid: q.offers for q in an.sealed(set_)}
    build = isolated(TOTALS_RELEASE)
    reference = run_claimcheck([sys.executable], jobs)
    control = run_claimcheck(build, jobs)
    stored = path(set_, "claims", name)
    if set_ == "battery" and stored.exists():
        old = {r["qid"]: r["verdicts"] for r in map(json.loads, stored.read_text().splitlines())}
        if old != {q: r["verdicts"] for q, r in reference.items()}:
            raise SystemExit("0.3.0 no longer gives the stored arm 2 verdicts")
    moved = [q for q in reference if control[q]["verdicts"] != reference[q]["verdicts"]]
    if moved or set(control) != set(reference):
        raise SystemExit(f"control: totals off differs from arm 2 on {moved or 'the answer set'}")
    after = run_claimcheck(build, jobs, "--totals")
    write(set_, "claims76", name, claim_rows(jobs, answers, after, f"{TOTALS} totals=line"))
    diff = changes(reference, after, offers)
    print(
        f"{set_} claims76 {name}: {len(after)} answers; control same as arm 2; "
        f"{len(diff['changed'])} verdicts changed, {len(diff['expected'])} expected"
    )
    if set_ == "dev":
        unkept = [k for k in diff["expected"] if after[k[0]]["verdicts"][k[1]] != "EMIT"]
        if sorted(diff["changed"]) != sorted(diff["expected"]) or unkept:
            raise SystemExit(f"dev check failed: {diff}, not kept {unkept}")
        print("dev check passed")


def judge(set_: str, name: str) -> None:
    """Arm 1: the other model judges each sentence through the verifier, as stop 71's check."""
    an.env()
    judge_name = OTHER[name]
    questions = {q.qid: q for q in an.sealed(set_)}
    answers = an.done(set_, name)
    labels = OUT / f"labels.{judge_name}.jsonl"

    def one(qid: str) -> dict[str, Any]:
        q, a = questions[qid], answers[qid]
        scorer = jd.RouterJudge(an.MODELS[judge_name], labels)
        verifier = Verifier(scorer, jd.STRICT_FIGURES)
        premise = Premise.from_text(q.premise)
        kept: list[str] = []
        said = sentences(a["text"])
        verdicts = []
        t0 = time.perf_counter()
        for s in said:
            scorer.prefix = " ".join(kept)
            v = verifier.check_sentence(premise, s, query=q.question).verdict.value
            verdicts.append(v)
            if v == "EMIT":
                kept.append(s)
        return {**checked(a, said, verdicts), "check_seconds": round(time.perf_counter() - t0, 3),
                "judge": an.MODELS[judge_name]}  # fmt: skip

    with ThreadPoolExecutor(an.WORKERS) as pool:
        rows = list(pool.map(one, sorted(answers)))
    write(set_, "judge", name, rows)
    print(f"{set_} judge {name} by {judge_name}: {len(rows)} answers")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=["claims", "claims-020", "claims76", "judge"])
    parser.add_argument("set", choices=["dev", "battery"])
    parser.add_argument("model", choices=sorted(an.MODELS))
    args = parser.parse_args()
    if args.arm == "judge":
        judge(args.set, args.model)
    elif args.arm == "claims76":
        totals(args.set, args.model)
    else:
        claims(args.set, args.model, old=args.arm == "claims-020")


if __name__ == "__main__":
    main()
