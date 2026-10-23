"""Modal spend, as the verifier's ``modal_ledger.py``: GPU time at list prices plus 15 %, recorded
when each run returns, with a hard cap: test 1's (docs/METHOD.md, Changes) and test 2's, whose runs
are named ``test2 …`` (docs/METHOD2.md §7)."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results/raw/modal_ledger.json"
PER_HOUR = {"L4": 0.80, "L40S": 1.95, "H100": 3.95, "CPU": 0.10}
OVERHEAD = 1.15
CAP = 3.0  # test 1
STAGES = {"test2": 5.0}  # your go, 2026-10-23


def usd(gpu: str, seconds: float) -> float:
    return round(PER_HOUR[gpu] * seconds / 3600 * OVERHEAD, 4)


def entries() -> list[dict[str, Any]]:
    if not LEDGER.exists():
        return []
    data: list[dict[str, Any]] = json.loads(LEDGER.read_text(encoding="utf-8"))
    return data


def _stage(run: str) -> str:
    return next((s for s in STAGES if run.startswith(f"{s} ")), "")


def spent(stage: str = "") -> float:
    """Spend of test 1's runs, or of one stage's."""
    return round(sum(float(e["usd"]) for e in entries() if _stage(e["run"]) == stage), 4)


def check(gpu: str, estimate_seconds: float, stage: str = "") -> None:
    cap = STAGES[stage] if stage else CAP
    if spent(stage) + usd(gpu, estimate_seconds) > cap:
        raise SystemExit(f"Modal {stage or 'test 1'}: ${spent(stage):.2f} spent, cap ${cap:.2f}")


def record(run: str, gpu: str, seconds: float, note: str = "") -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = entries()
        row = {"run": run, "gpu": gpu, "seconds": round(seconds, 1), "usd": usd(gpu, seconds)}
        data.append({**row, "note": note})
        tmp = LEDGER.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        tmp.replace(LEDGER)
