"""Test 1's answers (docs/METHOD.md §5): each question to Opus 4.6 and Flash through the router.

  uv run python scripts/answer.py dev flash      # the dev set, to check the prompt and the scripts
  uv run python scripts/answer.py battery opus46 # the sealed battery, resumable

The premises are rebuilt from ``data/`` and checked against ``battery/seal.json`` first. Answers can
quote the sheets, so they stay in ``results/raw/answers/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import json
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

import build_battery as bb  # type: ignore[import-not-found]  # noqa: E402
import fetch_icecat as fi  # type: ignore[import-not-found]  # noqa: E402
import router  # type: ignore[import-not-found]  # noqa: E402

from product_answer_checks import battery as bt  # noqa: E402

OUT = ROOT / "results/raw/answers"
MODELS = {"flash": router.GEMINI, "opus46": router.OPUS46}
WORKERS = 4
_lock = threading.Lock()


def env() -> None:
    """Read ``.env`` into the environment for the router's key file."""
    import os  # noqa: PLC0415

    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name and value:
                os.environ.setdefault(name.strip(), value.strip())


def sealed(set_: str) -> list[bt.Question]:
    """The set's questions, rebuilt, each premise checked against the seal."""
    seal = json.loads(fi.SEAL.read_text(encoding="utf-8"))
    sel, _ = bb.selection()
    battery, dev = bt.questions(sel, fi.load)
    questions = battery if set_ == "battery" else dev
    moved = [q.qid for q in questions if seal["premises"].get(q.qid) != bt.sha256(q.premise)]
    if moved:
        raise SystemExit(f"{len(moved)} premises differ from the seal, first {moved[0]}")
    return questions


def messages(q: bt.Question) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": bt.PROMPT.format(premise=q.premise)},
        {"role": "user", "content": q.question},
    ]


def path(set_: str, name: str) -> Path:
    return OUT / f"{set_}.{name}.jsonl"


def done(set_: str, name: str) -> dict[str, dict[str, Any]]:
    p = path(set_, name)
    if not p.exists():
        return {}
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
    return {r["qid"]: r for r in rows if r.get("text") is not None}


def append(set_: str, name: str, records: Sequence[Mapping[str, Any]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with _lock, path(set_, name).open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def answer(set_: str, name: str) -> None:
    """One answer per question; an error is retried once, then left missing (METHOD §5)."""
    env()
    have = done(set_, name)
    todo = [q for q in sealed(set_) if q.qid not in have]
    print(f"{set_} {name}: {len(todo)} to answer", flush=True)
    failed: list[str] = []

    def one(q: bt.Question) -> None:
        for _ in range(2):
            t0 = time.perf_counter()
            try:
                out = router.chat(MODELS[name], messages(q))
            except router.RouterError as exc:
                error = repr(exc)[:300]
                continue
            text = (out.get("content") or "").strip()
            record = {"qid": q.qid, "text": text, "model": MODELS[name]}
            record["seconds"] = round(time.perf_counter() - t0, 3)
            append(set_, name, [record])
            return
        failed.append(f"{q.qid}: {error}")

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(one, todo))
    for line in failed:
        print(f"MISSING {line}", flush=True)
    print(f"{set_} {name}: {len(done(set_, name))} answers", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("set", choices=["dev", "battery"])
    parser.add_argument("model", choices=sorted(MODELS))
    args = parser.parse_args()
    answer(args.set, args.model)


if __name__ == "__main__":
    main()
