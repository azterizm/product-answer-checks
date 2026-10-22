"""The local LLM router: one endpoint, three models, retries with backoff, a spend ledger.

Copied from ``legal-rag-verifier`` ``scripts/router.py`` at da40159; only the paths, the models and
the cap differ. The key is read from ``LLM_ROUTER_KEY`` or the file named by ``ROUTER_KEY_FILE``.

Only ``/v1/chat/completions`` on ``localhost:8317`` is called, and only with the models named in
``MODELS`` (the model value picks the route). Rate limits, server errors and timeouts back off and
retry; a reply's token usage goes to a ledger before the next call starts. ``claude-opus-5-5(high)``
has a hard cap: a call that could take the ledger past it is not made (``BudgetExceededError``).
"""

from __future__ import annotations

import fcntl
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
URL = "http://localhost:8317/v1/chat/completions"
KEY_FILE = ROOT / ".router_key"  # git-ignored; ROUTER_KEY_FILE names another
GEMINI = "gemini-3.8-flash-high"
OPUS46 = "claude-opus-4-6-thinking"
OPUS55 = "claude-opus-5-5(high)"
MODELS = frozenset({GEMINI, OPUS46, OPUS55})
# USD per million input and output tokens, for capped models only.
PRICES = {OPUS55: (4.0, 20.0)}
CAPS = {OPUS55: 10.0}  # your go, 2026-10-22 (METHOD, Changes)
LEDGER = ROOT / "results/raw/router_ledger.json"
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
MAX_OUTPUT_GUESS = 8000  # tokens reserved per call when checking the cap
_lock = threading.Lock()


class BudgetExceededError(RuntimeError):
    """The next call could take a capped model past its cap."""


class RouterError(RuntimeError):
    """A call still failed after every retry."""


def _key() -> str:
    key_file = Path(os.environ.get("ROUTER_KEY_FILE") or KEY_FILE)
    return os.environ.get("LLM_ROUTER_KEY") or key_file.read_text(encoding="utf-8").strip()


def ledger() -> dict[str, dict[str, float]]:
    if not LEDGER.exists():
        return {}
    data: dict[str, dict[str, float]] = json.loads(LEDGER.read_text(encoding="utf-8"))
    return data


def _write_ledger(data: dict[str, dict[str, float]]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(LEDGER)


def cost(model: str, input_tokens: float, output_tokens: float) -> float:
    if model not in PRICES:
        return 0.0
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1e6


def _chars(messages: Sequence[dict[str, str]]) -> int:
    return sum(len(m["content"]) for m in messages)


def _reserve(model: str, messages: Sequence[dict[str, str]]) -> None:
    """Refuse a call whose worst case could take the model past its cap."""
    if model not in CAPS:
        return
    spent = ledger().get(model, {}).get("usd", 0.0)
    worst = cost(model, 1.5 * _chars(messages) / 4, MAX_OUTPUT_GUESS)
    if spent + worst > CAPS[model]:
        raise BudgetExceededError(f"{model}: ${spent:.2f} spent, cap ${CAPS[model]:.2f}")


def _record(
    model: str, messages: Sequence[dict[str, str]], usage: dict[str, Any], text: str
) -> None:
    """Add one call to the ledger. Without usage, tokens are estimated from characters, x1.5."""
    estimated = not usage.get("prompt_tokens")
    tokens_in = float(usage.get("prompt_tokens") or 1.5 * _chars(messages) / 4)
    tokens_out = float(usage.get("completion_tokens") or 1.5 * len(text) / 4)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with _lock, LEDGER.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # several processes share the ledger
        data = ledger()
        entry = data.setdefault(
            model,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0, "estimated": 0},
        )
        entry["calls"] += 1
        entry["input_tokens"] += tokens_in
        entry["output_tokens"] += tokens_out
        entry["usd"] = round(entry["usd"] + cost(model, tokens_in, tokens_out), 6)
        entry["estimated"] += int(estimated)
        _write_ledger(data)


def _backoff(attempt: int, retry_after: str | None) -> float:
    if retry_after and retry_after.isdigit():
        return min(float(retry_after), 300.0)
    return float(min(300.0, 5 * 2**attempt + random.uniform(0, 3)))


def chat(
    model: str,
    messages: Sequence[dict[str, str]],
    *,
    temperature: float = 0.0,
    attempts: int = 6,
    timeout: float = 600,
) -> dict[str, Any]:
    """One reply: ``{content, usage, ms}``. Retries rate limits, server errors and timeouts."""
    if model not in MODELS:
        raise ValueError(f"{model!r} is not a router model this repo may call")
    _reserve(model, messages)
    body = json.dumps({"model": model, "messages": list(messages), "temperature": temperature})
    failure = ""
    for attempt in range(attempts):
        request = urllib.request.Request(
            URL,
            data=body.encode(),
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {_key()}"},
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                out = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_STATUS:
                raise RouterError(f"{model}: HTTP {exc.code}") from exc
            failure = f"HTTP {exc.code}"
            wait = _backoff(attempt, exc.headers.get("Retry-After"))
            after = exc.headers.get("Retry-After")
            note = f"{failure} (Retry-After {after})"
            print(f"router {model}: {note}, retry in {wait:.0f} s", file=sys.stderr, flush=True)
            time.sleep(wait)
            continue
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            failure = repr(exc)
            wait = _backoff(attempt, None)
            print(f"router {model}: {failure}, retry in {wait:.0f} s", file=sys.stderr, flush=True)
            time.sleep(wait)
            continue
        ms = (time.perf_counter() - t0) * 1000
        content = out["choices"][0]["message"].get("content") or ""
        usage = out.get("usage") or {}
        _record(model, messages, usage, content)
        return {"content": content, "usage": usage, "ms": ms}
    raise RouterError(f"{model}: still failing after {attempts} attempts ({failure})")
