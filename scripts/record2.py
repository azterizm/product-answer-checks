"""Test 2's judge prompts (docs/METHOD2.md §4): each answer through the verifier as arm 3t runs it,
with a scorer that records every prompt the 1B judge would see and answers as the grade says.

Reads JSON lines on stdin, ``{"qid", "generator", "premise", "query", "sentences", "grades"}`` (a
grade per sentence: "yes", "no", "unsure" or null), and writes per answer the claim check's verdicts
and, for each sentence it passes, the prompts of its clauses. Like ``claimcheck.py`` it imports only
``legal_rag_verifier`` and ``judge.py``, so it runs on 0.4.0 in an isolated environment.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import judge as jd  # type: ignore[import-not-found]

from legal_rag_verifier.premise import Premise
from legal_rag_verifier.verifier import NLIProbs, Verifier, VerifierConfig

LABEL = {"yes": "supported", "no": "contradicted"}  # anything else: not_stated


class Recorder:
    """The judge's place in the verifier: keeps each short prompt, answers as the grade says."""

    model_id = "grades (test 2 teacher)"

    def __init__(self) -> None:
        self.prefix = ""
        self.label = "not_stated"
        self.prompts: list[str] = []

    def score(self, premises: Sequence[str], hypothesis: str) -> list[NLIProbs]:
        self.prompts.append(jd.short_prompt(hypothesis, premises, self.prefix))
        return list(jd.one_hot(self.label, 0, len(premises)))


def record(
    row: dict[str, Any], claims: Verifier, judged: Verifier, scorer: Recorder
) -> dict[str, Any]:
    premise = Premise.from_text(row["premise"])
    first = [
        claims.check_sentence(premise, s, query=row["query"]).verdict.value
        for s in row["sentences"]
    ]
    kept: list[str] = []
    verdicts: list[str] = []
    prompts: list[list[str]] = []
    for sentence, grade, verdict in zip(row["sentences"], row["grades"], first, strict=True):
        if verdict != "EMIT":
            verdicts.append("ROLLBACK")
            prompts.append([])
            continue
        scorer.prefix = " ".join(kept)
        scorer.label = LABEL.get(grade or "", "not_stated")
        scorer.prompts = []
        v = judged.check_sentence(premise, sentence, query=row["query"]).verdict.value
        verdicts.append(v)
        prompts.append(scorer.prompts)
        if v == "EMIT":
            kept.append(sentence)
    return {
        "qid": row["qid"],
        "generator": row["generator"],
        "claims": first,
        "verdicts": verdicts,
        "prompts": prompts,
        "config": judged.config.to_dict(),
    }


def main() -> None:
    # Only 0.4.0 has ``totals``; this file runs there, while the project holds 0.3.0.
    claims = Verifier(config=VerifierConfig(totals="line"))  # type: ignore[call-arg, unused-ignore]
    scorer = Recorder()
    strict = dataclasses.replace(jd.STRICT_FIGURES, totals="line")  # type: ignore[call-arg, unused-ignore]
    judged = Verifier(scorer, strict)  # arm 3t's judge stage; the line above is arm 2t
    for line in sys.stdin:
        print(json.dumps(record(json.loads(line), claims, judged, scorer)), flush=True)


if __name__ == "__main__":
    main()
