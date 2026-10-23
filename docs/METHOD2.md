# Test 2: the 1B judge trained on product sentences (method fixed 2026-10-23, before any code)

Your decisions of 2026-10-23: every proposal of the draft, then, after the licence finding (§1), own listings for
training. A change after this commit is listed under "Changes" with its date.

## Question

Test 1's cheapest strong arm was the claim check, then the published 1B judge: it kept 5 of Opus 4.6's 32 wrong
sentences in 0.13 s per answer, but removed 148 right ones (114 with totals, arm 3t) and emptied 33 answers. The
judge was trained on UK legislation and software documentation and had never seen a product sentence. **Does
training it on product sentences make it remove fewer right sentences while keeping as few wrong ones, at the same
speed?** This answers DataSnipper's own line, fine-tuning models to reduce latency and cut costs.

## 1. Products: own listings, no Icecat text in training

- **Why.** The Open Icecat Open Content License (version 1.4, 11 February 2026), §10, makes the permission to copy,
  redistribute or modify Open Icecat content "strictly conditional upon the data NOT being utilized for machine
  learning training, algorithmic model generation, or automated synthetic content creation". Every training
  example holds the listing the judge reads, so no Icecat sheet is used to build training data. Test 1's battery is
  used only at test time (§6), as in test 1.
- **The generator** (`src/product_answer_checks/synthetic.py`, seed 20261104): made-up products in 12 categories
  taken from METHOD §1's in-scope words: laptops, desktop PCs, monitors, TVs, smartphones, tablets, headphones,
  washing machines, fridge-freezers, dishwashers, vacuum cleaners, coffee machines. Per category, a table of 15 to
  20 features (group, name and unit in English and French, a list or range of values), written by hand from
  general product knowledge, not read off any Icecat sheet. Invented brands and model names. The monitors, TVs and
  the three large appliances carry an energy class and an energy consumption.
- Each product is a sheet in the shape `sheets.py` reads (title, short summary, feature groups), so test 1's code
  writes its listing (`Group — Name: Value`) and its questions unchanged. French values use the decimal comma,
  French units and "Oui"/"Non".
- **10 products per category, 120 in all.** Split by product with `random.Random(20261104)`: 108 train, 12
  validation.

## 2. Questions and offers

- Test 1's code (`battery.py`, `offers.py`) with one change: a seed argument, whose default keeps test 1's battery
  as sealed (checked against `battery/seal.json`). Each question's offers come from `random.Random(20261104 + 1000
  × product + type)`, so a product's questions do not share one draw. The same offer date, traps, templates and
  prompt (`battery/prompt.txt`), both listing languages.
- Per product: one question each of types A, C and D, and E for the 50 products with energy features. Type B:
  triples of one category and one split sharing a numeric feature (test 1's rule), each product in at most one
  triple, in product order.
- About 890 questions. Committed before any answer: the products (IDs, categories, brands, titles, split, listing
  hashes) and the questions (no listing text, as test 1). Training data, not a battery: not sealed.

## 3. Answers

- **Qwen 2.5 7B Instruct**, stop 64's generator and sampling (`legal-rag-verifier` `scripts/modal_distill.py`:
  SGLang 0.5.21 on one Modal L4, greedy, at most 512 tokens), the test 1 prompt. It is there to err on the traps,
  so the training set holds wrong sentences of the kind the judge must catch.
- **Gemini 3.8 Flash** through the router, router defaults, as test 1: right sentences in a frontier model's
  phrasing (totals, comparisons, recommendations), which the published judge removes.
- Opus 4.6 is left out: its router budget is spent.
- Sentences by the verifier's `split_sentences`.

## 4. Labels (the teacher)

- **Test 1's grader:** Gemini 3.8 Flash in the one-grader mode of `scripts/grade.py`, with its rule and prompt:
  each sentence right, wrong or unsure against the question, listings and offers, one call per question, both
  answers lettered at random. Reason: the judge must learn which sentences are right, and test 1 shows an
  entailment judge removes many right ones (Qwen 27B as arm 1 removed 56 of Flash's 262 right sentences). A
  student copies its teacher.
- **From sentence grade to training example.** Each answer runs through the verifier as arm 3t does (0.4.0,
  `totals="line"`; the claim check first, then `STRICT_FIGURES` with totals on the sentences it passes) with a
  scorer that records each judge prompt exactly as the 1B would see it (candidate passages, the answer so far,
  the clause) and answers as the grade says, so the answer so far holds the sentences a right judge keeps. The
  target is the judge's short reply:
  - a right sentence: each of its clauses `{"label": "supported", "passage": 0}`;
  - a wrong sentence of one clause: `{"label": "contradicted", "passage": 0}`;
  - a wrong sentence of several clauses, and an unsure sentence: left out (counted), since the grade does not
    say which clause is wrong.
  - Passage 0 applies the label to every candidate (`judge.one_hot`), so it decides the verdict alone.
- Sentences the claim check rolls back never reach the judge and give no example.

## 5. Training

- MiniCPM5-1B at `87179e5c`, stop 64's recipe and code (`train_small`, copied): LoRA rank 16, alpha 32, dropout
  0.05, on the attention and MLP projections, learning rate 1e-4, cosine, 3 % warm-up, seed 20261009, bf16, loss
  on the reply only, batches by length under 12,000 tokens, four batches per update; one H100; checkpoints on this
  repo's runs volume.
- **Data:** from the base, stop 64's 18,152 legal and software clauses (the verifier's
  `results/distill/students/train.jsonl`, local, sha256 recorded) plus the product training clauses, each product
  clause twice per epoch, two epochs. One adapter, `run3`.
- **Cost rule, fixed now:** the training time is estimated from stop 64's rate (characters of prompt per second).
  If two epochs with the product clauses doubled come to more than $3.50, the product clauses go once per epoch;
  if still more, one epoch. The training call stops itself at the time $3.75 buys and keeps what it has; that
  would be reported.

## 6. Scoring

- **V1, validation:** on the validation clauses, the share where the judge gives the teacher's label, for `run3`
  and, as the baseline, the published `run2` (stop 64: 86.7 % on its own validation set).
- **Arms on test 1's battery,** the same answers, sentence split and grades (nothing graded again), the new
  judge served exactly as arm 3 (merged, SGLang 0.5.21, one L40S, greedy, one answer at a time, `STRICT_FIGURES`):
  - **4t:** claim check 0.4.0 with totals, then `run3`. The main arm.
  - **4:** claim check 0.3.0 (arm 2), then `run3`, beside arm 3.
- **R1, fixed now:** test 1's reading rule for 4t, per generator: at most 2 more wrong sentences kept than arm 1,
  and fewer right sentences removed than arm 1. On Opus 4.6 that is at most 2 wrong kept and fewer than 94 right
  removed; on Flash, at most 2 and fewer than 56.
- **R2, fixed now:** 4t against 3t (`run2` after totals): right sentences that only one of the two removes, exact
  two-sided sign test; wrong sentences kept as counts (3t: 9 on Opus 4.6).
- Also reported, per generator and half: empty answers, wrong prices kept, seconds per answer at the median.
- **Noise:** the published judge's verdict changed on 4 of 306 identical inputs between two runs (stop 76). Each
  arm runs once, so differences of one or two sentences are within that.
- **Legal check:** `run3` on the verifier's two fresh tests with your grades (stop 64's Z2: the judge alone,
  `STRICT_FIGURES`), beside `run2` (89.0 % and 82.1 %), to show what product training does to the legal judge.

## 7. Spend

- **Modal: a cap of $5 for test 2** in this repo's ledger, checked before each call. The ledgers leave about $6.66
  of the $7.88 before test 1. Estimate: 7B answers $0.25, training $3.00, merge $0.10, V1 $0.10, the two arms
  $0.45, legal check $0.20: about $4.10, the weights downloaded again.
- **Router:** Flash answers and grades, no charge. No Opus call.
- **Publishing:** `run3` to Hugging Face (`azterizm/minicpm5-1b-clause-judge`) only after the results, on your go.

## 8. Limits (stated in the README)

- The training listings are made up, simpler and cleaner than real sheets; the questions, offers and traps come
  from the battery's own templates and generator. Test 2 measures the gain where the task matches but the listings
  do not.
- One grader, one sample per sentence; the student copies its grader's errors. Flash grades its own answers.
- Labels are per sentence, while the judge sees clauses.
- One run per arm; the judge itself is not exactly repeatable.

## Changes

None yet.
