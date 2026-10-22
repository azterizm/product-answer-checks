"""Modal spend for test 1, as the verifier's ``modal_ledger.py``: GPU time at list prices plus 15 %,
recorded when each run returns, with a hard cap (docs/METHOD.md, Changes)."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results/raw/modal_ledger.json"
PER_HOUR = {"L40S": 1.95, "CPU": 0.10}
OVERHEAD = 1.15
CAP = 3.0


def usd(gpu: str, seconds: float) -> float:
    return round(PER_HOUR[gpu] * seconds / 3600 * OVERHEAD, 4)


def entries() -> list[dict[str, Any]]:
    if not LEDGER.exists():
        return []
    data: list[dict[str, Any]] = json.loads(LEDGER.read_text(encoding="utf-8"))
    return data


def spent() -> float:
    return round(sum(float(e["usd"]) for e in entries()), 4)


def check(gpu: str, estimate_seconds: float) -> None:
    if spent() + usd(gpu, estimate_seconds) > CAP:
        raise SystemExit(f"Modal: ${spent():.2f} spent, cap ${CAP:.2f}")


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
