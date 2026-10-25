# product-answer-checks

[![CI](https://github.com/azterizm/product-answer-checks/actions/workflows/ci.yml/badge.svg)](https://github.com/azterizm/product-answer-checks/actions/workflows/ci.yml)

Measures what it costs to check AI shopping answers against product listings, and what each check catches.

## Results

Two models answered 120 product questions. Each question came with a product listing and shop offers. Half of
the listings were in French, half in English. Each answer was checked in four ways.

Claude Opus 4.6: 288 sentences, 32 wrong.

| Check | Wrong sentences kept | Wrong prices kept | Correct sentences removed | Time added per answer |
|---|---|---|---|---|
| No check | 32 | 13 | 0 | 0 |
| Second model checks each sentence, Gemini 3.8 Flash | 0 | 0 | 94 | 12.6 s |
| Claim check of prices, dates and units | 21 | 3 | 106 | 6 ms |
| Claim check with totals (added 23 Oct) | 30 | 12 | 25 | 6.5 ms |
| Claim check, then a 1B model | 5 | 2 | 148 | 0.13 s |
| Claim check with totals, then the 1B model (23 Oct) | 9 | 5 | 114 | 0.19 s |
| Claim check, then the 1B model trained on product answers (test 2, 25 Oct) | 14 | 1 | 130 | 0.12 s |
| Claim check with totals, then the product-trained 1B model (test 2) | 20 | 7 | 68 | 0.19 s |

Gemini 3.8 Flash: 262 sentences, none wrong.

| Check | Wrong sentences kept | Correct sentences removed | Time added per answer |
|---|---|---|---|
| No check | 0 | 0 | 0 |
| Second model checks each sentence, Qwen 27B | 0 | 56 | 1.7 s |
| Claim check of prices, dates and units | 0 | 75 | 6 ms |
| Claim check with totals (added 23 Oct) | 0 | 11 | 6.6 ms |
| Claim check, then a 1B model | 0 | 108 | 0.14 s |
| Claim check with totals, then the 1B model (23 Oct) | 0 | 87 | 0.14 s |
| Claim check, then the 1B model trained on product answers (test 2, 25 Oct) | 0 | 93 | 0.13 s |
| Claim check with totals, then the product-trained 1B model (test 2) | 0 | 52 | 0.14 s |

- A second model removed every wrong sentence. It also removed about one in three correct sentences and added
  seconds to each answer.
- The claim check added milliseconds but kept most wrong sentences. Most errors were reasoning errors, such as a
  product said to be under budget when shipping puts it over. The check also removed correct sentences that
  state a total the model calculated.
- The claim check with totals also accepts a figure equal to one offer's price plus its shipping. On Opus 4.6
  it removed 25 correct sentences instead of 106: 81 of the 94 that state such a total were kept. Empty answers
  fell from 9 to 2. It kept 9 more wrong sentences. Each states a correct total inside a wrong claim, such as
  the wrong cheapest shop or a budget it does not meet. A wrong sum, "€311.00 + €4.99 = €316.99", is still
  removed. It needs the claim check's 0.4.0, released 23 Oct, with `totals="line"`.
- With the 1B model after it, the claim check with totals removed 114 correct sentences instead of 148 and
  kept 9 wrong ones instead of 5. The 1B model removed 6 of the 9 wrong sentences that totals let through. Run
  twice on the same input, the 1B model gave a different verdict on 4 of 306 sentences.
- The claim check followed by the 1B model removed most wrong sentences in 0.13 s. It removed the most correct
  sentences. It did not meet the pass rule set before the test.
- Test 2 trained the 1B model on product answers as well as its legal data: two models' answers to 886
  questions about 120 made-up products, graded by Gemini 3.8 Flash. After the claim check with totals it removed
  68 correct Opus 4.6 sentences instead of 114, and 52 of Flash's instead of 87, at the same speed. On Opus 4.6
  it kept 20 wrong sentences instead of 9. Most of the 11 more open a budget answer, "Two products meet both
  criteria:", when only one does once shipping is added. It met the pass rule on Flash's answers, not on Opus
  4.6's.
- On the two legal tests the 1B model was built for, the product-trained model scored 87.3 % and 83.4 %, against
  89.0 % and 82.1 % before. Neither difference is clear.
- French listings now lose about as many correct sentences as English listings: 36 and 39 for Flash, 51 and 55
  for Opus 4.6. The previous release of the claim check removed 74 and 88 on the French half.

Times were measured on different hardware. The second model ran through a hosted API for the Opus 4.6 answers
and on one GPU for the Flash answers. The claim check ran on a laptop CPU. The 1B model ran on one GPU.

## Method

- Products and listings come from Open Icecat, in French and English for the same product.
- Offers are constructed. Each set includes common traps: an old price shown next to the current one, a lower
  price with higher shipping, and the cheapest offer out of stock.
- Five question types: best price, fit under a budget, specifications, delivery and returns, energy use.
- A wrong price is a wrong sentence that states a euro amount.
- The method was fixed before the test. Every later change is dated in [docs/METHOD.md](docs/METHOD.md).
- Test 2's method, changes and results: [docs/METHOD2.md](docs/METHOD2.md). Its training listings are made up
  in code (`src/product_answer_checks/synthetic.py`), with test 1's questions, offers and traps.
- All counts are in [results/report.json](results/report.json).

## Limits

- Offers are constructed. Icecat has no prices.
- Products are limited to electronics, computing and appliances.
- Questions and answers are in English. Each model answered once.
- Two models graded the sentences, Gemini 3.8 Flash and Qwen 27B. They agreed on 546 of 550. A person graded the
  other 4 and checked 52 random sentences, agreeing with the models on 50. Flash also graded its own answers.
- The second model check used a different model for each answer set. Its times do not compare between the two
  tables.
- The first 1B model was trained on UK legislation and software documentation, not on product text. Test 2's
  was also trained on made-up listings, simpler than real ones, with the same question templates and offers as
  the test, and on one model's grades.
- There are few wrong sentences, 32, all from one model.
- This is not a test of any company's production system.

## Reproduce

Requirements:

- Python 3.11 or later and [uv](https://docs.astral.sh/uv/).
- An Open Icecat account. Set `ICECAT_USERNAME`.
- An OpenAI compatible endpoint at `localhost:8317` that serves `claude-opus-4-6-thinking` and
  `gemini-3.8-flash-high`. Set `LLM_ROUTER_KEY`.
- A [Modal](https://modal.com) account for the 1B model and Qwen 27B. Test 1 cost $0.84, the totals rows $0.38,
  test 2 $4.55 (training $3.49).

Settings can be kept in a `.env` file in the repository root. The tests need none of them: `uv run pytest`.

```sh
uv run python scripts/fetch_icecat.py rebuild
uv run python scripts/answer.py battery flash
uv run python scripts/check.py claims battery flash
uv run python scripts/check.py claims76 battery flash   # totals: 0.4.0 from PyPI
uv run --with modal modal run scripts/judge1b.py::run --split battery --model flash
uv run python scripts/grade.py sheet
uv run --with modal modal run scripts/qwen27.py::run
uv run python scripts/report.py
```

Test 2 (training data, training, arms 4 and 4t), in order:

```sh
uv run python scripts/test2.py build
uv run python scripts/test2.py answer
uv run --with modal modal run scripts/test2_modal.py::answers
uv run python scripts/test2.py sheet
uv run python scripts/test2.py grade
uv run python scripts/test2.py record
uv run python scripts/test2.py dataset   # needs the verifier's stop 64 training file
uv run --with modal modal run --detach scripts/test2_modal.py::train
uv run --with modal modal run scripts/judge1b.py::labels --tasks validation,fresh,fresh2
uv run --isolated --no-project --with modal --with legal-rag-verifier==0.4.0 \
  modal run scripts/judge1b.py::run --split battery --model flash --totals --judge run3
```

The `battery` folder holds the product IDs, offers, questions, prompt and a hash of every listing.
`fetch_icecat.py rebuild` reports any listing that changed since the test.

## Data and licence

- Product data: [Open Icecat](https://icecat.biz), Open Content Licence, provided as is. Listing text is not
  included. Each user downloads it with their own account.
- Test 2 trains on made-up listings only. The Open Content Licence (version 1.4, §10) does not permit machine
  learning training on Open Icecat content.
- The product-trained 1B model is published as `run3` at
  [huggingface.co/azterizm/minicpm5-1b-clause-judge](https://huggingface.co/azterizm/minicpm5-1b-clause-judge),
  Apache-2.0.
- Code, offers and questions: [AGPL-3.0](LICENSE).
- Claim check: [legal-rag-verifier](https://github.com/azterizm/legal-rag-verifier) 0.3.0; the totals rows
  0.4.0.
