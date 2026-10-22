"""The claim check on answers, for arms 2 and 2b (docs/METHOD.md §6). Reads JSON lines on
stdin, ``{"qid", "premise", "query", "sentences"}``, and writes each sentence's verdict.

It imports only ``legal_rag_verifier``, so the same file runs on the pinned build (arm 2) and on
0.2.0 from PyPI in an isolated environment (arm 2b).
"""

from __future__ import annotations

import json
import sys
import time

from legal_rag_verifier.premise import Premise
from legal_rag_verifier.verifier import Verifier


def main() -> None:
    import legal_rag_verifier  # noqa: PLC0415

    verifier = Verifier()
    for line in sys.stdin:
        row = json.loads(line)
        premise = Premise.from_text(row["premise"])
        t0 = time.perf_counter()
        verdicts = [
            verifier.check_sentence(premise, s, query=row["query"]) for s in row["sentences"]
        ]
        seconds = time.perf_counter() - t0
        out = {
            "qid": row["qid"],
            "verdicts": [v.verdict.value for v in verdicts],
            "reasons": [sorted(r.value for r in v.reasons) for v in verdicts],
            "seconds": round(seconds, 6),
            "version": getattr(legal_rag_verifier, "__version__", "?"),
        }
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
