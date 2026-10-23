"""Arm 3 (docs/METHOD.md §6, Changes): the claim check, then the stop 64 1B judge on the sentences
it passes, the judge on one Modal L40S. Paid: run only on a go; Modal spend is capped in its ledger.

  uv run --with modal modal run scripts/judge1b.py::merge                 # once: adapter into base
  uv run --with modal modal run scripts/judge1b.py::run --split battery --model flash

Arm 3t (METHOD, Changes 2026-10-23) runs on arm 2t's verdicts with ``totals="line"``, from an
environment that holds only the verifier's stop 76 build, so that build is what the container gets:

  uv run --isolated --no-project --with modal \
    --with "legal-rag-verifier @ git+file://<the verifier repo>@<check.TOTALS>" \
    modal run scripts/judge1b.py::run --split battery --model flash --totals

Sends the listings, offers, questions and the sentences the claim check passed; returns labels.
"""

# No `from __future__ import annotations`: Modal reads function annotations at runtime.
import dataclasses
import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import modal

ROOT = Path(__file__).resolve().parent.parent
SGLANG_IMAGE = "lmsysorg/sglang:v0.5.21"
BASE = "openbmb/MiniCPM5-1B"
BASE_REVISION = "87179e5c1f455ef22e6223592d2d61351b525bfc"
ADAPTER = "azterizm/minicpm5-1b-clause-judge"
ADAPTER_REVISION = "7cd8ee5f54e53b0268bdab49b0e5a48fe99ef0d0"
RUN = "run2"  # stop 64: trained on both pools
MERGED = "/runs/minicpm5-1b-run2-merged"
SAMPLING = {"temperature": 0, "max_tokens": 1024, "seed": 0}
TEMPLATE = {"enable_thinking": False}
PORT = 30000
GPU = "L40S"
WORKERS = 1  # one answer at a time, so each answer's time is not shared (METHOD, Changes)
CHUNK = 60
merge_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.14.0", "transformers==5.17.0", "peft==0.21.0", "accelerate==1.15.0")
    .env({"HF_HOME": "/hf"})
)
serve_image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .env({"PYTHONPATH": "/app"})
    .add_local_python_source("legal_rag_verifier")
    .add_local_file(ROOT / "scripts/judge.py", "/app/judge.py")
)
runs = modal.Volume.from_name("product-answer-checks-runs", create_if_missing=True)
app = modal.App("product-answer-checks-arm3")


# ---------------------------------------------------------------------- Modal side
@app.function(image=merge_image, cpu=4, memory=16384, timeout=3600, volumes={"/runs": runs})
def merge_adapter() -> str:
    """The published adapter merged into the base in bf16, with the base's code and tokenizer."""
    import shutil  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    base = AutoModelForCausalLM.from_pretrained(
        BASE, revision=BASE_REVISION, dtype=torch.bfloat16, trust_remote_code=True
    )
    adapter = snapshot_download(ADAPTER, revision=ADAPTER_REVISION, allow_patterns=[f"{RUN}/*"])
    merged = PeftModel.from_pretrained(base, str(Path(adapter) / RUN)).merge_and_unload()
    target = Path(MERGED)
    merged.save_pretrained(str(target), safe_serialization=True)
    AutoTokenizer.from_pretrained(
        BASE, revision=BASE_REVISION, trust_remote_code=True
    ).save_pretrained(str(target))
    snapshot = Path(snapshot_download(BASE, revision=BASE_REVISION))
    for f in [*snapshot.glob("*.py"), *snapshot.glob("*.jinja"), snapshot / "config.json"]:
        if f.exists():
            shutil.copy(f, target / f.name)  # the base's config keeps auto_map and archs
    (target / "SOURCE.json").write_text(
        json.dumps({"base": [BASE, BASE_REVISION], "adapter": [ADAPTER, ADAPTER_REVISION, RUN]})
    )
    runs.commit()
    return str(target)


def _wait(server: "subprocess.Popen[bytes]", url: str, limit: float = 1500) -> None:
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


@app.cls(image=serve_image, gpu=GPU, volumes={"/runs": runs}, timeout=3600, max_containers=1,
         scaledown_window=60)  # fmt: skip
class Judge1B:
    @modal.enter()
    def start(self) -> None:
        self.started = time.time()
        self.server = subprocess.Popen(  # noqa: S603
            [
                sys.executable, "-m", "sglang.launch_server", "--model-path", MERGED,
                "--port", str(PORT), "--mem-fraction-static", "0.85", "--context-length", "65536",
            ]
        )  # fmt: skip
        self.url = f"http://127.0.0.1:{PORT}"
        _wait(self.server, self.url)
        self.startup = time.time() - self.started

    @modal.exit()
    def stop(self) -> None:
        self.server.terminate()

    def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body = {"model": MERGED, "messages": messages, **SAMPLING, "chat_template_kwargs": TEMPLATE}
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

    @modal.method()
    def check(self, jobs: list[dict[str, Any]]) -> dict[str, Any]:
        """Each answer: sentences the claim check dropped stay dropped; the rest go through the
        verifier with the 1B judge as its entailment head (stop 71's arrangement)."""
        import judge as jd  # type: ignore[import-not-found]  # noqa: PLC0415

        from legal_rag_verifier.premise import Premise  # noqa: PLC0415
        from legal_rag_verifier.verifier import Verifier  # noqa: PLC0415

        server = self

        class OneB:
            model_id = f"{ADAPTER}@{ADAPTER_REVISION}/{RUN}"

            def __init__(self) -> None:
                self.prefix = ""
                self.labels: list[dict[str, Any]] = []

            def score(self, premises: Any, hypothesis: str) -> list[Any]:
                label = jd.ask(server.chat, jd.short_prompt(hypothesis, premises, self.prefix))
                self.labels.append(
                    {k: label.get(k) for k in ("label", "passage", "ms", "attempts")}
                )
                name = label["label"] if label["label"] in jd.LABELS else "not_stated"
                return list(jd.one_hot(name, int(label.get("passage") or 0), len(premises)))

        def config(job: dict[str, Any]) -> Any:
            if job.get("totals"):  # raises on a build without ``totals``
                return dataclasses.replace(jd.STRICT_FIGURES, totals="line")
            return jd.STRICT_FIGURES

        def one(job: dict[str, Any]) -> dict[str, Any]:
            scorer = OneB()
            verifier = Verifier(scorer, config(job))
            premise = Premise.from_text(job["premise"])
            kept: list[str] = []
            verdicts = []
            t0 = time.perf_counter()
            for sentence, first in zip(job["sentences"], job["claims"], strict=True):
                if first != "EMIT":
                    verdicts.append("ROLLBACK")
                    continue
                scorer.prefix = " ".join(kept)
                v = verifier.check_sentence(premise, sentence, query=job["query"]).verdict.value
                verdicts.append(v)
                if v == "EMIT":
                    kept.append(sentence)
            seconds = round(time.perf_counter() - t0, 4)
            return {"qid": job["qid"], "verdicts": verdicts, "judge_seconds": seconds,
                    "labels": scorer.labels, "config": verifier.config.to_dict()}  # fmt: skip

        t0 = time.time()
        with ThreadPoolExecutor(WORKERS) as pool:
            results = list(pool.map(one, jobs))
        return {"results": results, "seconds": time.time() - t0, "startup": self.startup}


# ---------------------------------------------------------------------- local side
def _ledger() -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    import modal_ledger  # type: ignore[import-not-found]  # noqa: PLC0415

    return modal_ledger


@app.local_entrypoint()
def merge() -> None:
    ledger = _ledger()
    ledger.check("CPU", 1800)
    t0 = time.time()
    print(merge_adapter.remote())
    ledger.record("merge adapter", "CPU", time.time() - t0, f"{ADAPTER}@{ADAPTER_REVISION}/{RUN}")
    print(f"Modal spent ${ledger.spent():.2f} of ${ledger.CAP:.2f}")


@app.local_entrypoint()
def run(split: str = "battery", model: str = "flash", totals: bool = False) -> None:  # noqa: FBT001, FBT002
    ledger = _ledger()
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import answer as an  # type: ignore[import-not-found]  # noqa: PLC0415
    import check as ck  # type: ignore[import-not-found]  # noqa: PLC0415

    if totals:  # the build this environment holds is the one the container gets
        from legal_rag_verifier.verifier import VerifierConfig  # noqa: PLC0415

        VerifierConfig(totals="line")  # type: ignore[call-arg, unused-ignore]
    questions = {q.qid: q for q in an.sealed(split)}
    rows = ck.path(split, "claims76" if totals else "claims", model).read_text().splitlines()
    first = {r["qid"]: r for r in map(json.loads, rows)}
    out = ck.path(split, "judge1b76" if totals else "judge1b", model)
    arm = "arm 3t" if totals else "arm 3"
    done = out.read_text(encoding="utf-8").splitlines() if out.exists() else []
    have = {r["qid"] for r in map(json.loads, done)}
    jobs = [
        {"qid": qid, "premise": questions[qid].premise, "query": questions[qid].question,
         "sentences": r["sentences"], "claims": r["verdicts"], "totals": totals}
        for qid, r in sorted(first.items()) if qid not in have
    ]  # fmt: skip
    print(f"{split} {model}: {len(jobs)} answers to check", flush=True)
    judge = Judge1B()
    startup_recorded = False
    for start in range(0, len(jobs), CHUNK):
        ledger.check(GPU, 900)
        t0 = time.time()
        reply = judge.check.remote(jobs[start : start + CHUNK])
        if totals and any(r["config"].get("totals") != "line" for r in reply["results"]):
            raise SystemExit("the container's verifier is not running totals")
        seconds = reply["seconds"] + (0 if startup_recorded else reply["startup"])
        startup_recorded = True
        ledger.record(
            f"{arm} {split} {model}",
            GPU,
            max(seconds, time.time() - t0),
            f"{len(reply['results'])} answers",
        )
        with out.open("a", encoding="utf-8") as f:
            for r in reply["results"]:
                sentences = first[r["qid"]]["sentences"]
                row = ck.checked({"qid": r["qid"]}, sentences, r["verdicts"])
                row |= {
                    "claims": first[r["qid"]]["verdicts"],
                    "claim_seconds": first[r["qid"]]["check_seconds"],
                    "judge_seconds": r["judge_seconds"],
                    "labels": r["labels"],
                    "gpu": GPU,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"{len(reply['results'])} answers, {reply['seconds']:.0f} s; "
            f"Modal ${ledger.spent():.2f}",
            flush=True,
        )
