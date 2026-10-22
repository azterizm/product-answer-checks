"""Qwen 27B on one Modal L40S (docs/METHOD.md, Changes): the second grader of the 550 sentences, and
arm 1's judge on Flash's answers. Paid: Modal spend is capped in ``scripts/modal_ledger.py``.

  uv run --with modal modal run scripts/qwen27.py::run

One run: Qwen grades every question; Flash and Qwen grade again what they differ on; Qwen checks
Flash's answers sentence by sentence (arm 1), one answer at a time. Resumable: a rerun skips what
is done. Sends the listings, offers, questions and sentences; returns grades and labels.
"""

# No `from __future__ import annotations`: Modal reads function annotations at runtime.
import json
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import modal

ROOT = Path(__file__).resolve().parent.parent
SGLANG_IMAGE = "lmsysorg/sglang:v0.5.21"
# The verifier's 27B judge, "qwen27-ar4" (its stops 61 to 73): 4-bit AutoRound, thinking off.
MODEL = "Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ"
REVISION = "1b9d56b428e9f841d7ad566f87660b5f47553c97"
NAME = "qwen27-ar4"
ARGS = ["--reasoning-parser", "qwen3", "--context-length", "65536"]
SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "max_tokens": 1024,
    "seed": 0,
}  # the card's non-thinking sampling, as the verifier
TEMPLATE = {"enable_thinking": False}
PORT = 30000
GPU = "L40S"
GRADE_WORKERS = 16
CHUNK = 40

image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .env({"HF_HOME": "/hf", "PYTHONPATH": "/app"})
    .add_local_python_source("legal_rag_verifier")
    .add_local_file(ROOT / "scripts/judge.py", "/app/judge.py")
)
weights = modal.Volume.from_name("product-answer-checks-hf", create_if_missing=True)
app = modal.App("product-answer-checks-qwen27")


def _wait(server: "subprocess.Popen[bytes]", url: str, limit: float = 2400) -> None:
    deadline = time.time() + limit
    while True:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5) as r:  # noqa: S310
                if r.status == 200:  # noqa: PLR2004
                    return
        except OSError:
            pass
        if server.poll() is not None or time.time() > deadline:
            raise RuntimeError("SGLang server did not come up")
        time.sleep(5)


@app.cls(image=image, gpu=GPU, volumes={"/hf": weights}, timeout=3600, max_containers=1,
         scaledown_window=300)  # fmt: skip
class Qwen27:
    @modal.enter()
    def start(self) -> None:
        self.started = time.time()
        self.server = subprocess.Popen(  # noqa: S603
            [
                sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL,
                "--revision", REVISION, "--port", str(PORT), "--mem-fraction-static", "0.85", *ARGS,
            ]
        )  # fmt: skip
        self.url = f"http://127.0.0.1:{PORT}"
        _wait(self.server, self.url)
        weights.commit()
        self.startup = time.time() - self.started
        self.reported = False

    @modal.exit()
    def stop(self) -> None:
        self.server.terminate()

    def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body = {"model": MODEL, "messages": messages, **SAMPLING, "chat_template_kwargs": TEMPLATE}
        request = urllib.request.Request(  # noqa: S310 - the container's own server
            f"{self.url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310
            out = json.loads(response.read())
        return {
            "content": out["choices"][0]["message"].get("content") or "",
            "ms": (time.perf_counter() - t0) * 1000,
            "tokens": out.get("usage", {}).get("completion_tokens"),
        }

    def _startup(self) -> float:
        """The start-up time, reported once for the ledger."""
        if self.reported:
            return 0.0
        self.reported = True
        return self.startup

    @modal.method()
    def complete(self, batch: list[list[dict[str, str]]]) -> dict[str, Any]:
        """One reply per conversation, many at a time (grading)."""
        t0 = time.time()
        with ThreadPoolExecutor(GRADE_WORKERS) as pool:
            replies = list(pool.map(self.chat, batch))
        return {"replies": [r["content"] for r in replies], "seconds": time.time() - t0,
                "startup": self._startup()}  # fmt: skip

    @modal.method()
    def check(self, jobs: list[dict[str, Any]]) -> dict[str, Any]:
        """Arm 1: each answer's sentences through the verifier with Qwen as its judge, the short
        reply, the answer so far as the prefix (stop 71's check); one answer at a time."""
        import judge as jd  # type: ignore[import-not-found]  # noqa: PLC0415

        from legal_rag_verifier.premise import Premise  # noqa: PLC0415
        from legal_rag_verifier.verifier import Verifier  # noqa: PLC0415

        server = self

        class Scorer:
            model_id = f"{MODEL}@{REVISION}"

            def __init__(self) -> None:
                self.prefix = ""
                self.labels: list[dict[str, Any]] = []

            def score(self, premises: Any, hypothesis: str) -> list[Any]:
                label = jd.ask(server.chat, jd.short_prompt(hypothesis, premises, self.prefix))
                self.labels.append({k: label.get(k) for k in ("label", "passage", "ms")})
                name = label["label"] if label["label"] in jd.LABELS else "not_stated"
                return list(jd.one_hot(name, int(label.get("passage") or 0), len(premises)))

        t_all = time.time()
        results = []
        for job in jobs:
            scorer = Scorer()
            verifier = Verifier(scorer, jd.STRICT_FIGURES)
            premise = Premise.from_text(job["premise"])
            kept: list[str] = []
            verdicts = []
            t0 = time.perf_counter()
            for sentence in job["sentences"]:
                scorer.prefix = " ".join(kept)
                v = verifier.check_sentence(premise, sentence, query=job["query"]).verdict.value
                verdicts.append(v)
                if v == "EMIT":
                    kept.append(sentence)
            seconds = round(time.perf_counter() - t0, 4)
            results.append(
                {
                    "qid": job["qid"],
                    "verdicts": verdicts,
                    "seconds": seconds,
                    "labels": scorer.labels,
                }
            )
        return {"results": results, "seconds": time.time() - t_all, "startup": self._startup()}


# ---------------------------------------------------------------------- local side
def _local() -> tuple[Any, Any, Any, Any]:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import answer as an  # type: ignore[import-not-found]  # noqa: PLC0415
    import check as ck  # type: ignore[import-not-found]  # noqa: PLC0415
    import grade as gr  # type: ignore[import-not-found]  # noqa: PLC0415
    import modal_ledger as ml  # type: ignore[import-not-found]  # noqa: PLC0415

    return an, ck, gr, ml


def _grade(qwen: Any, gr: Any, ml: Any, *, again: bool) -> None:
    """Qwen's share of a grading pass: what is left, in chunks, three tries per question."""
    for _ in range(3):
        jobs = gr.todo("qwen27", again=again)
        if not jobs:
            return
        print(f"qwen27{' again' if again else ''}: {len(jobs)} questions", flush=True)
        for start in range(0, len(jobs), CHUNK):
            chunk = jobs[start : start + CHUNK]
            ml.check(GPU, 900)
            reply = qwen.complete.remote([messages for _, _, messages in chunk])
            ml.record(
                "qwen27 grading", GPU, reply["seconds"] + reply["startup"], f"{len(chunk)} calls"
            )
            for (q, items, _), text in zip(chunk, reply["replies"], strict=True):
                gr.save("qwen27", q, items, text, again=again)


@app.local_entrypoint()
def run() -> None:
    an, ck, gr, ml = _local()
    qwen = Qwen27()
    _grade(qwen, gr, ml, again=False)
    flash_again = threading.Thread(target=gr.grade, args=("flash",), kwargs={"again": True})
    flash_again.start()
    _grade(qwen, gr, ml, again=True)
    flash_again.join()

    out = ck.path("battery", "judge", "flash")
    have = (
        {json.loads(r)["qid"] for r in out.read_text(encoding="utf-8").splitlines()}
        if out.exists()
        else set()
    )
    questions = {q.qid: q for q in an.sealed("battery")}
    answers = an.done("battery", "flash")
    jobs = [
        {"qid": qid, "premise": questions[qid].premise, "query": questions[qid].question,
         "sentences": ck.sentences(a["text"])}
        for qid, a in sorted(answers.items()) if qid not in have
    ]  # fmt: skip
    print(f"arm 1 on Flash's answers: {len(jobs)} to check", flush=True)
    for start in range(0, len(jobs), CHUNK):
        ml.check(GPU, 1200)
        reply = qwen.check.remote(jobs[start : start + CHUNK])
        ml.record(
            "qwen27 arm 1",
            GPU,
            reply["seconds"] + reply["startup"],
            f"{len(reply['results'])} answers",
        )
        with out.open("a", encoding="utf-8") as f:
            for r in reply["results"]:
                said = next(j["sentences"] for j in jobs if j["qid"] == r["qid"])
                row = ck.checked({"qid": r["qid"]}, said, r["verdicts"])
                row |= {
                    "check_seconds": r["seconds"],
                    "judge": NAME,
                    "gpu": GPU,
                    "labels": r["labels"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"{len(reply['results'])} answers, {reply['seconds']:.0f} s; Modal ${ml.spent():.2f}",
            flush=True,
        )
