"""Test 2's legal check (docs/METHOD2.md §6): run3 on the verifier's two fresh tests with your
grades, scored as stop 64's Z2 (the judge alone, ``STRICT_FIGURES``), beside run2.

Runs in the verifier's environment and imports its scripts, so the prompts, the replay and the
score are the verifier's own (``fresh_arms.small``):

  cd ../legal-rag-verifier
  FRESH_SET=fresh uv run python ../product-answer-checks/scripts/legal3.py prompts
  FRESH_SET=fresh2 uv run python ../product-answer-checks/scripts/legal3.py prompts
  (product repo) uv run --with modal modal run scripts/judge1b.py::labels --tasks fresh,fresh2
  FRESH_SET=fresh uv run python ../product-answer-checks/scripts/legal3.py score
  FRESH_SET=fresh2 uv run python ../product-answer-checks/scripts/legal3.py score

run2 is the verifier's ``minicpm5-1b-student2`` (the published adapter), from its stored labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
VERIFIER = ROOT.parent / "legal-rag-verifier"
sys.path[:0] = [str(VERIFIER / "src"), str(VERIFIER / "scripts")]

import auditor_judge as aj  # type: ignore[import-not-found]  # noqa: E402
import fresh_arms as fa  # type: ignore[import-not-found]  # noqa: E402
import fresh_eval as fe  # type: ignore[import-not-found]  # noqa: E402
import fresh_report as fr  # type: ignore[import-not-found]  # noqa: E402

OUT = ROOT / "results/raw/test2/legal"
SET = os.environ.get("FRESH_SET", "fresh")
RUN2 = "minicpm5-1b-student2"  # stop 64, published as run2
PUBLISHED = {"fresh": 0.890, "fresh2": 0.821}  # stop 64's Z2 for run2


def prompts() -> dict[str, int]:
    asked = fa.requests("short")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{SET}.prompts.json").write_text(json.dumps(asked) + "\n", encoding="utf-8")
    return {"set": SET, "prompts": len(asked)}  # type: ignore[dict-item]


def labels(judge: str) -> dict[str, dict[str, Any]]:
    if judge == RUN2:
        return dict(fa.read(RUN2, "short"))
    path = OUT / f"{SET}.labels.{judge}.jsonl"
    rows = map(json.loads, path.read_text(encoding="utf-8").splitlines())
    return {r["sha"]: r for r in rows if r["label"] in aj.LABELS}


def score() -> dict[str, Any]:
    data = fa.recorded()
    us, built = data["units"], data["built"]
    graded = fe.merged_grades(us)
    units = [
        {**u, "right": graded[u["key"]] == "yes", "wrong": graded[u["key"]] == "no"} for u in us
    ]
    units = [u for u in units if u["right"] or u["wrong"]]
    asked = fa.requests("short", data)
    out: dict[str, Any] = {"set": SET, "sentences": len(units), "prompts": len(asked)}
    correct = {}
    for title, name in (("run2", RUN2), ("run3", "run3")):
        table = labels(name)
        scorer = fa.Scorer(name, "short", table, {})
        verdicts = fa.replay(us, built, scorer, set(), aj.STRICT_FIGURES)
        total = fr.score(units, verdicts)["total"]
        out[title] = {**total, "labelled": sum(k in table for k in asked)}
        correct[title] = {u["key"]: fr._rolled(verdicts[u["key"]]) == u["wrong"] for u in units}
    if out["run2"]["accuracy"] != PUBLISHED[SET]:
        raise SystemExit(f"run2 scores {out['run2']['accuracy']}, not stop 64's {PUBLISHED[SET]}")
    out["run3_against_run2"] = fa.paired(correct["run3"], correct["run2"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{SET}.score.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prompts", "score"])
    args = parser.parse_args()
    print(json.dumps(prompts() if args.command == "prompts" else score(), indent=1))


if __name__ == "__main__":
    main()
