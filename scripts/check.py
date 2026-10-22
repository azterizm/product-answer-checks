"""Test 1's arms 1, 2 and 2b on the answers (docs/METHOD.md §6). A sentence that is not emitted is
dropped, as in the verifier's stop 71.

  uv run python scripts/check.py claims battery flash        # arm 2: the pinned build, CPU
  uv run python scripts/check.py claims-020 battery flash    # arm 2b: 0.2.0 from PyPI, French half
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


def claims(set_: str, name: str, *, old: bool) -> None:
    """Arm 2 (pinned build) or 2b (0.2.0, French half only), in a subprocess so 2b is isolated."""
    questions = {q.qid: q for q in an.sealed(set_)}
    answers = an.done(set_, name)
    jobs = [
        {"qid": qid, "premise": questions[qid].premise, "query": questions[qid].question,
         "sentences": sentences(a["text"])}
        for qid, a in sorted(answers.items())
        if not old or questions[qid].lang == "fr"
    ]  # fmt: skip
    if old:
        command = ["uv", "run", "--isolated", "--no-project", "--with", OLD, "python"]
    else:
        command = [sys.executable]
    stdin = "".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs)
    result = subprocess.run(  # noqa: S603
        [*command, str(ROOT / "scripts/claimcheck.py")],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    rows = []
    by_qid = {j["qid"]: j for j in jobs}
    for line in result.stdout.splitlines():
        out = json.loads(line)
        job = by_qid[out["qid"]]
        rows.append({**checked(answers[out["qid"]], job["sentences"], out["verdicts"]),
                     "reasons": out["reasons"], "check_seconds": out["seconds"],
                     "verifier": "0.2.0" if old else PINNED})  # fmt: skip
    arm = "claims-020" if old else "claims"
    write(set_, arm, name, rows)
    print(f"{set_} {arm} {name}: {len(rows)} answers, verifier {'0.2.0' if old else PINNED}")


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
    parser.add_argument("arm", choices=["claims", "claims-020", "judge"])
    parser.add_argument("set", choices=["dev", "battery"])
    parser.add_argument("model", choices=sorted(an.MODELS))
    args = parser.parse_args()
    if args.arm == "judge":
        judge(args.set, args.model)
    else:
        claims(args.set, args.model, old=args.arm == "claims-020")


if __name__ == "__main__":
    main()
