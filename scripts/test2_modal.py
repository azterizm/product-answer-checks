"""Test 2 on Modal (docs/METHOD2.md §3 and §5). Paid: run only on a go; spend is capped at $5 in
this repo's Modal ledger (stage ``test2``), checked before each call.

  uv run --with modal modal run scripts/test2_modal.py::answers   # Qwen 2.5 7B answers, one L4
  uv run --with modal modal run scripts/test2_modal.py::train     # run3 on one H100, then merged

The generator, ``train_small`` and ``merge_small`` are copied from ``legal-rag-verifier``
``scripts/modal_distill.py`` at ce08220 (stops 43, 63 and 64); only the volumes, the paths and the
ledger differ. Sends the made-up listings, questions and the training file; nothing from Icecat.
"""

# No `from __future__ import annotations`: Modal reads function annotations at runtime.
import hashlib
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
GENERATOR = "Qwen/Qwen2.5-7B-Instruct"
GENERATOR_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"  # as the verifier's stops 27 to 64
STUDENT = "openbmb/MiniCPM5-1B"
STUDENT_REVISION = "87179e5c1f455ef22e6223592d2d61351b525bfc"
STAGE = "test2"
PORT = 30000
CHUNK = 40
WORKERS = 16
TRAIN_SEED = 20261009
BATCH_TOKENS = 12_000  # stop 64
TRAIN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

sglang_image = modal.Image.from_registry(SGLANG_IMAGE).env({"HF_HOME": "/hf"})
small_train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.14.1", "transformers==5.18.0", "peft==0.21.0", "accelerate==1.15.0")
    .env({"HF_HOME": "/hf"})
)
weights = modal.Volume.from_name("product-answer-checks-hf", create_if_missing=True)
runs = modal.Volume.from_name("product-answer-checks-runs", create_if_missing=True)
app = modal.App("product-answer-checks-test2")


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


# ---------------------------------------------------------------------- answers (stop 43's)
@app.cls(image=sglang_image, gpu="L4", volumes={"/hf": weights}, timeout=3600, max_containers=1,
         scaledown_window=60)  # fmt: skip
class Generator:
    """Qwen 2.5 7B Instruct behind SGLang, started once per container."""

    @modal.enter()
    def start(self) -> None:
        self.started = time.time()
        self.server = subprocess.Popen(  # noqa: S603
            [
                sys.executable, "-m", "sglang.launch_server",
                "--model-path", GENERATOR, "--revision", GENERATOR_REVISION,
                "--port", str(PORT), "--mem-fraction-static", "0.85",
            ]
        )  # fmt: skip
        self.url = f"http://127.0.0.1:{PORT}"
        _wait(self.server, self.url)
        weights.commit()
        self.startup = time.time() - self.started

    @modal.exit()
    def stop(self) -> None:
        self.server.terminate()

    def _one(self, request: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": GENERATOR,
            "messages": request["messages"],
            "temperature": 0,
            "max_tokens": 512,
        }
        error = ""
        for attempt in range(3):
            try:
                req = urllib.request.Request(  # noqa: S310 - the container's own server
                    f"{self.url}/v1/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=600) as r:  # noqa: S310
                    out = json.loads(r.read())
                choice = out["choices"][0]
                return {
                    "key": request["key"],
                    "text": choice["message"]["content"],
                    "finish": choice.get("finish_reason"),
                    "tokens": out.get("usage", {}).get("completion_tokens"),
                    "model": f"{GENERATOR}@{GENERATOR_REVISION}",
                }
            except OSError as exc:
                error = repr(exc)
                time.sleep(5 * (attempt + 1))
        return {"key": request["key"], "text": None, "error": error}

    @modal.method()
    def answer(self, chunk: list[dict[str, Any]]) -> dict[str, Any]:
        t0 = time.time()
        with ThreadPoolExecutor(WORKERS) as pool:
            results = list(pool.map(self._one, chunk))
        return {"results": results, "seconds": time.time() - t0, "startup": self.startup}


# ---------------------------------------------------------------------- training (stops 63, 64)
@app.function(
    image=small_train_image, gpu="H100", timeout=6 * 3600, volumes={"/hf": weights, "/runs": runs}
)
def train_small(
    epochs: int = 2,
    max_tokens: int = 8192,
    *,
    run: str = "run3",
    batch_tokens: int = BATCH_TOKENS,
    budget_seconds: float = 0.0,
) -> dict:
    """LoRA on the teacher's short replies. Each example is the prompt as the model sees it with
    thinking off, then the reply and the end-of-turn text; the loss is on the reply only.
    Checkpoints are committed to the runs volume, so a lost container resumes."""
    import torch  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    out = Path("/runs/minicpm5-1b") / run
    tokenizer = AutoTokenizer.from_pretrained(STUDENT, revision=STUDENT_REVISION)
    marker = "\x00REPLY\x00"
    probe = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": marker},
    ]
    full = tokenizer.apply_chat_template(probe, tokenize=False, enable_thinking=False)
    end = full[full.index(marker) + len(marker) :].strip() or tokenizer.eos_token

    def encode(messages: list[dict[str, str]]) -> dict[str, list[int]]:
        system, user, reply = messages
        prompt = tokenizer.apply_chat_template(
            [system, user], add_generation_prompt=True, return_dict=True, enable_thinking=False
        )["input_ids"]
        target = tokenizer(reply["content"] + end, add_special_tokens=False)["input_ids"]
        ids = [*prompt, *target]
        return {"input_ids": ids, "labels": [-100] * len(prompt) + list(target)}

    lines = Path("/runs/test2/data/train.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [encode(json.loads(x)["messages"]) for x in lines]
    kept = [r for r in rows if len(r["input_ids"]) <= max_tokens]

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        width = max(len(b["input_ids"]) for b in batch)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        ids = [b["input_ids"] + [pad] * (width - len(b["input_ids"])) for b in batch]
        labels = [b["labels"] + [-100] * (width - len(b["labels"])) for b in batch]
        mask = [[1] * len(b["input_ids"]) + [0] * (width - len(b["input_ids"])) for b in batch]
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }

    model = AutoModelForCausalLM.from_pretrained(
        STUDENT, revision=STUDENT_REVISION, dtype=torch.bfloat16, device_map={"": 0}
    )
    targets = [
        name
        for name, module in model.named_modules()
        if type(module) is torch.nn.Linear and name.rsplit(".", 1)[-1] in TRAIN_TARGETS
    ]
    config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    started = time.time()
    stopped_early: list[int] = []

    class Commit(TrainerCallback):  # type: ignore[misc]
        def on_save(self, *args: Any, **kwargs: Any) -> None:
            runs.commit()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if budget_seconds and time.time() - started > budget_seconds:
                stopped_early.append(state.global_step)
                control.should_training_stop = True

    # Stop 64: batches built by length under a token budget (examples x longest example), at
    # most 16 examples, four batches per update.
    order = sorted(range(len(kept)), key=lambda i: len(kept[i]["input_ids"]))
    batches: list[list[int]] = [[]]
    for i in order:  # ascending, so the current example is the longest of its batch
        last = batches[-1]
        full_batch = len(last) >= 16 or (len(last) + 1) * len(kept[i]["input_ids"]) > batch_tokens  # noqa: PLR2004
        if last and full_batch:
            batches.append([i])
        else:
            last.append(i)
    data = [collate([kept[i] for i in batch]) for batch in batches if batch]
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # a float below 1 is a ratio of total steps
        bf16=True,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        report_to=[],
        seed=TRAIN_SEED,
        remove_unused_columns=False,
    )

    class ReplyOnly(Trainer):  # type: ignore[misc]
        """Cross-entropy on the reply tokens, the output layer applied only at their positions."""

        def compute_loss(
            self,
            model: Any,
            inputs: Any,
            return_outputs: bool = False,  # noqa: FBT001, FBT002 - the library's signature
            **kwargs: Any,
        ) -> Any:
            core = model.get_base_model()
            hidden = core.model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ).last_hidden_state
            targets = inputs["labels"][:, 1:]  # position t predicts token t + 1
            keep = targets != -100  # noqa: PLR2004
            scores = core.lm_head(hidden[:, :-1][keep])
            loss = torch.nn.functional.cross_entropy(scores.float(), targets[keep])
            return (loss, None) if return_outputs else loss

    trainer = ReplyOnly(
        model=model,
        args=args,
        train_dataset=data,
        data_collator=lambda items: items[0],
        callbacks=[Commit()],
    )
    t0 = time.time()
    resume = any((out / "checkpoints").glob("checkpoint-*"))
    result = trainer.train(resume_from_checkpoint=True if resume else None)
    model.save_pretrained(str(out / "adapter"))
    runs.commit()
    return {
        "examples": len(kept),
        "dropped_over_max_tokens": len(rows) - len(kept),
        "longest_tokens": max(len(r["input_ids"]) for r in rows),
        "trainable_parameters": trainable,
        "lora_layers": len(targets),
        "batches": len(data),
        "stopped_at_step": stopped_early[0] if stopped_early else None,
        "steps": trainer.state.global_step,
        "end_of_turn": end,
        "seconds": time.time() - t0,
        "loss": result.training_loss,
        "log": trainer.state.log_history[-12:],
        "resumed": resume,
    }


@app.function(
    image=small_train_image, gpu="H100", timeout=3600, volumes={"/hf": weights, "/runs": runs}
)
def merge_small(run: str = "run3") -> str:
    """The adapter merged into the base in bf16, with every other file of the base beside it
    (tokenizer, chat template, processor settings), so the server loads it like the base."""
    import shutil  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    folder = Path("/runs/minicpm5-1b") / run
    base = AutoModelForCausalLM.from_pretrained(
        STUDENT, revision=STUDENT_REVISION, dtype=torch.bfloat16
    )
    merged = PeftModel.from_pretrained(base, str(folder / "adapter")).merge_and_unload()
    target = folder / "merged"
    if target.exists():
        shutil.rmtree(target)
    merged.save_pretrained(str(target), safe_serialization=True)
    snapshot = Path(snapshot_download(STUDENT, revision=STUDENT_REVISION))
    for f in snapshot.iterdir():
        weights_file = f.suffix == ".safetensors" or f.name == "model.safetensors.index.json"
        if f.is_file() and not weights_file and not (target / f.name).exists():
            shutil.copy(f, target / f.name)
    runs.commit()
    return str(target)


# ---------------------------------------------------------------------- local side
def _local() -> tuple[Any, Any, Any]:
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    import answer as an  # type: ignore[import-not-found]  # noqa: PLC0415
    import modal_ledger  # type: ignore[import-not-found]  # noqa: PLC0415
    import test2  # type: ignore[import-not-found]  # noqa: PLC0415

    return modal_ledger, an, test2


@app.local_entrypoint()
def answers(limit: int = 0) -> None:
    """Every question to the 7B; chunks appended as they return, a rerun asks only what is left."""
    ledger, an, t2 = _local()
    have = an.done(t2.SET, "qwen7b")
    todo = [
        {"key": q.qid, "messages": an.messages(q)} for q in t2.questions() if q.qid not in have
    ][: limit or None]
    print(f"qwen7b: {len(have)} done, {len(todo)} to answer", flush=True)
    if not todo:
        return
    ledger.check("L4", 1800, STAGE)
    chunks = [todo[i : i + CHUNK] for i in range(0, len(todo), CHUNK)]
    startup_recorded = False
    for out in Generator().answer.map(chunks, return_exceptions=True):
        if isinstance(out, Exception):
            print(f"chunk failed, asked again on the next run: {out!r}"[:300], flush=True)
            continue
        good = [r for r in out["results"] if r.get("text") is not None]
        an.append(
            t2.SET,
            "qwen7b",
            [{"qid": r["key"], "text": r["text"].strip(), "model": r["model"],
              "finish": r["finish"], "tokens": r["tokens"]} for r in good],
        )  # fmt: skip
        seconds = out["seconds"] + (0 if startup_recorded else out["startup"])
        startup_recorded = True
        ledger.record(f"{STAGE} answers qwen7b", "L4", seconds, f"{len(good)} answers")
    print(f"qwen7b: {len(an.done(t2.SET, 'qwen7b'))} answers; Modal {STAGE} "
          f"${ledger.spent(STAGE):.2f} of ${ledger.STAGES[STAGE]:.2f}")  # fmt: skip


@app.local_entrypoint()
def train(run: str = "run3") -> None:
    """Upload the training file ``dataset`` wrote, train as its plan says, merge. Ledgered."""
    ledger, _, t2 = _local()
    summary = json.loads((t2.COMMITTED / "dataset.json").read_text(encoding="utf-8"))
    plan = summary["plan"]
    folder = t2.RAW / "run3"
    if (
        hashlib.sha256((folder / "train.jsonl").read_bytes()).hexdigest()
        != summary["train_file_sha256"]
    ):
        raise SystemExit("the training file is not the one test2/dataset.json describes")
    ledger.check("H100", plan["estimate_seconds"] + 600, STAGE)
    with runs.batch_upload(force=True) as batch:
        for split in ("train", "validation"):
            batch.put_file(str(folder / f"{split}.jsonl"), f"/test2/data/{split}.jsonl")
    t0 = time.time()
    try:
        out = train_small.remote(plan["epochs"], run=run, budget_seconds=plan["stop_seconds"])
    finally:
        ledger.record(f"{STAGE} train {run}", "H100", time.time() - t0)
    (folder / "train_result.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "log"}, indent=1))
    ledger.check("H100", 600, STAGE)
    t0 = time.time()
    try:
        print(merge_small.remote(run))
    finally:
        ledger.record(f"{STAGE} merge {run}", "H100", time.time() - t0)
    print(f"Modal {STAGE}: ${ledger.spent(STAGE):.2f} of ${ledger.STAGES[STAGE]:.2f}")
