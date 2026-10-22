# product-answer-checks

What it costs to check an AI shopping answer against its product listing before it is shown, and what each check
catches. Two hosted models answered 120 product questions from Open Icecat listings, half in French and half in
English, with constructed offers. Each answer was then checked four ways. Method, fixed before the runs and with
every later change dated: [`docs/METHOD.md`](docs/METHOD.md). Counts: [`results/report.json`](results/report.json).

## Result

Claude Opus 4.6, 120 answers, 288 sentences, 32 of them wrong (35 at most, see grading):

| Check | Wrong sentences kept | Wrong prices kept | Right sentences removed | Time added per answer |
|---|---|---|---|---|
| None | 32–35 | 13–16 | 0 | 0 |
| Second model (Gemini 3.8 Flash) checks each sentence | 0 | 0 | 91–94 | 12.6 s (hosted, through a router) |
| Claim check (prices, dates, units) | 21–23 | 3–5 | 105–106 | 6 ms (CPU) |
| Claim check, then a 1B judge | 5–7 | 2–4 | 147–148 | 0.13 s (CPU, then one L40S) |

Gemini 3.8 Flash, 120 answers, 262 sentences, none wrong (one unsettled):

| Check | Wrong sentences kept | Right sentences removed | Time added per answer |
|---|---|---|---|
| None | 0–1 | 0 | 0 |
| Second model (Qwen 27B) checks each sentence | 0 | 55–56 | 1.7 s (one L40S) |
| Claim check | 0 | 74–75 | 6 ms (CPU) |
| Claim check, then a 1B judge | 0 | 107–108 | 0.14 s (CPU, then one L40S) |

- A second model catches every wrong sentence and removes about a third of the right ones; it adds seconds.
- The claim check costs milliseconds but keeps most wrong sentences: Opus 4.6's errors were mostly reasoning
  (two products said to fit a budget that shipping breaks), not misquoted figures. It removes right sentences
  that state a total the model added up ("€908.98 (€903.99 + €4.99)"), which it cannot ground.
- The 1B judge after it catches most of the rest, at 0.13 s, but removes the most right sentences. By the
  reading rule fixed beforehand (at most 2 more wrong sentences kept than the second model, and fewer right
  sentences removed), it does not cost less for the same safety, on either generator.
- French listings: the claim check (verifier `66450a6`, with the French formats) removes about as many right
  sentences on the French half as on the English (36 and 38 on Flash's answers, 51 and 54 on Opus 4.6's). The
  release before it (0.2.0) removed 74 and 88 on the French half.

## Limits

- Constructed offers (Icecat has no prices); Open Icecat categories only (electronics, computing, appliances).
- English questions and answers only; one run per generator.
- Not Joko's system, catalogue or prompts.
- Grading by two models, not by hand: Gemini 3.8 Flash and Qwen 27B (Claude Opus 5.5, planned as a grader, was
  unavailable). They agreed on 506 of 550 sentences at once and 40 more after giving reasons; 4 stay unsettled,
  hence the ranges. Flash grades its own answers too. 60 random sentences are set aside for a hand check.
- The second-model check used a different judge per generator (Flash for Opus 4.6's answers, through a router;
  Qwen 27B for Flash's, on a GPU), so their times do not compare with each other.
- The 1B judge (`azterizm/minicpm5-1b-clause-judge`, run2) was trained on UK legislation and has never seen a
  product sentence.
- Small counts: 32–35 wrong sentences, all from one generator.

## Reproduce

Python 3.11+, [uv](https://docs.astral.sh/uv/), an Open Icecat username (free) in `ICECAT_USERNAME`.

```sh
uv run python scripts/fetch_icecat.py rebuild      # the sealed sheets, checked against battery/seal.json
uv run python scripts/answer.py battery flash      # answers through an OpenAI-compatible router
uv run python scripts/check.py claims battery flash
uv run --with modal modal run scripts/judge1b.py::run --split battery --model flash
uv run python scripts/grade.py sheet               # then the graders; scripts/qwen27.py on Modal
uv run python scripts/report.py
```

The battery (`battery/`) holds the Icecat IDs, the constructed offers, the questions, the prompt and SHA-256
hashes of every sheet and premise; sheet text is not redistributed. Spend for this run: Modal $0.84; the router
calls were within existing plans.

Licence: AGPL-3.0.
