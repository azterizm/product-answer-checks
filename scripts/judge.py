"""The other model as the judge of each sentence, for arm 1 (docs/METHOD.md §6).

Copied from ``legal-rag-verifier`` at da40159: the prompt, the reply parsing and ``one_hot`` from
``scripts/auditor_judge.py``, and ``RouterJudge`` from ``scripts/sota71.py`` (stop 71's check, the
short reply). Only the label file's place differs.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from legal_rag_verifier.verifier import NLIProbs, VerifierConfig

LABELS = ("supported", "contradicted", "not_stated")
STRICT_FIGURES = VerifierConfig(
    neutral_policy="strict", neutral_with_grounded_claim="rollback", ungrounded_figure="judge"
)
_lock = threading.Lock()


SYSTEM = (
    "You check whether a sentence is supported by source passages. Use only the passages, not "
    "your own knowledge. Reply with a single JSON object."
)
USER = """Passages:
{passages}

Answer so far. Use it only to work out what words such as "it" or "this" refer to. It is not \
evidence.
<<<
{prefix}
>>>

Sentence to check:
<<<
{sentence}
>>>

Labels:
- "supported": a passage states the sentence, or the sentence follows from the passages by \
reasoning, such as applying a default, substituting a definition, following a reference to another \
passage, converting units or doing arithmetic.
- "contradicted": the passages state or imply something incompatible with the sentence, such as a \
different figure, condition, scope or outcome.
- "not_stated": neither. Use this when the sentence may be true but the passages do not establish \
it.

Return JSON with keys "label" (one of the three), "passage" (the number of the passage that \
decides it, or 0) and "reason" (one sentence)."""


def user_prompt(sentence: str, candidates: Sequence[str], prefix: str) -> str:
    passages = "\n\n".join(f"[{n}] {text}" for n, text in enumerate(candidates, 1))
    return USER.format(passages=passages, prefix=prefix or "(none)", sentence=sentence)


SHORT_REPLY = (
    'Return JSON with keys "label" (one of the three) and "passage" (the number of the passage '
    "that decides it, or 0). Give no reason."
)


def short_prompt(sentence: str, candidates: Sequence[str], prefix: str) -> str:
    head, _ = user_prompt(sentence, candidates, prefix).rsplit("Return JSON", 1)
    return head + SHORT_REPLY


def prompt_key(user: str) -> str:
    return hashlib.sha256(f"{SYSTEM}\n{user}".encode()).hexdigest()


def extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in output")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("not a JSON object")  # noqa: TRY004
    return parsed


def parse_reply(text: str) -> dict[str, Any]:
    """A judge reply as ``{label, passage, reason}``; ValueError when it is not one."""
    reply = extract_json(text)
    label = str(reply.get("label", "")).strip().lower().replace(" ", "_")
    if label not in LABELS:
        raise ValueError(f"label {label!r}")
    try:
        passage = int(reply.get("passage") or 0)
    except (TypeError, ValueError):
        passage = 0
    return {"label": label, "passage": passage, "reason": str(reply.get("reason", ""))}


RETRY = "Reply with ONLY the JSON object, with keys label, passage and reason."


def ask(chat: Callable[[list[dict[str, str]]], dict[str, Any]], user: str) -> dict[str, Any]:
    """At most 3 asks for a valid reply. ``chat`` returns ``{content, ms, tokens}``."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    ms = tokens = 0.0
    content = ""
    for attempt in range(1, 4):
        reply = chat(messages)
        ms += reply["ms"]
        tokens += reply.get("tokens") or 0
        content = reply["content"] or ""
        try:
            return {**parse_reply(content), "attempts": attempt, "ms": ms, "tokens": tokens}
        except (ValueError, json.JSONDecodeError):
            messages = [*messages[:2], {"role": "assistant", "content": content}]
            messages.append({"role": "user", "content": RETRY})
    return {
        "label": "error",
        "passage": 0,
        "reason": content[-300:],
        "attempts": 3,
        "ms": ms,
        "tokens": tokens,
    }


def one_hot(label: str, passage: int, n: int) -> list[NLIProbs]:
    """A judge label as entailment-model output over ``n`` candidates. Supported: entailment 1
    on its passage; contradicted: contradiction 1 there; otherwise neutral 1 everywhere. A
    passage number out of range applies the label to every candidate."""
    neutral = NLIProbs(0.0, 1.0, 0.0)
    if label == "supported":
        hit = NLIProbs(1.0, 0.0, 0.0)
    elif label == "contradicted":
        hit = NLIProbs(0.0, 0.0, 1.0)
    else:
        return [neutral] * n
    if not 1 <= passage <= n:
        return [hit] * n
    return [hit if i == passage - 1 else neutral for i in range(n)]


def router_chat(model: str) -> Callable[[list[dict[str, str]]], dict[str, Any]]:
    import router  # type: ignore[import-not-found]  # noqa: PLC0415 - not needed in arm 3's container

    def chat(messages: list[dict[str, str]]) -> dict[str, Any]:
        out = router.chat(model, messages)
        return {**out, "tokens": out["usage"].get("completion_tokens")}

    return chat


class RouterJudge:
    """A frontier model as the verifier's judge, the short reply, labels kept by prompt."""

    def __init__(self, model_id: str, file: Path) -> None:
        self.model_id = model_id
        self.chat = router_chat(model_id)
        self.file = file
        self.labels = {}
        if file.exists():
            for line in file.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.labels[row["sha"]] = row
        self.prefix = ""
        self.ms = 0.0

    def score(self, premises: Sequence[str], hypothesis: str) -> list[NLIProbs]:
        user = short_prompt(hypothesis, premises, self.prefix)
        sha = prompt_key(user)
        label = self.labels.get(sha)
        if label is None or label["label"] not in LABELS:
            label = ask(self.chat, user)
            self.ms += float(label.get("ms") or 0)
            with _lock:
                self.labels[sha] = label
                self.file.parent.mkdir(parents=True, exist_ok=True)
                with self.file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"sha": sha, **label}, ensure_ascii=False) + "\n")
        name = label["label"] if label["label"] in LABELS else "not_stated"
        return list(one_hot(name, int(label.get("passage") or 0), len(premises)))
