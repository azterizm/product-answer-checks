# product-answer-checks

[![CI](https://github.com/azterizm/product-answer-checks/actions/workflows/ci.yml/badge.svg)](https://github.com/azterizm/product-answer-checks/actions/workflows/ci.yml)

Measures what it costs to check AI shopping answers against product listings, and what each check catches.

## Results

Two models answered 120 product questions. Each question came with a product listing and shop offers. Half of
the listings were in French, half in English. Each answer was checked in four ways.

Claude Opus 4.6: 288 sentences, 32 to 35 wrong.

| Check | Wrong sentences kept | Wrong prices kept | Correct sentences removed | Time added per answer |
|---|---|---|---|---|
| No check | 32 to 35 | 13 to 16 | 0 | 0 |
| Second model checks each sentence, Gemini 3.8 Flash | 0 | 0 | 91 to 94 | 12.6 s |
| Claim check of prices, dates and units | 21 to 23 | 3 to 5 | 105 to 106 | 6 ms |
| Claim check, then a 1B model | 5 to 7 | 2 to 4 | 147 to 148 | 0.13 s |

Gemini 3.8 Flash: 262 sentences, 0 to 1 wrong.

| Check | Wrong sentences kept | Correct sentences removed | Time added per answer |
|---|---|---|---|
| No check | 0 to 1 | 0 | 0 |
| Second model checks each sentence, Qwen 27B | 0 | 55 to 56 | 1.7 s |
| Claim check of prices, dates and units | 0 | 74 to 75 | 6 ms |
| Claim check, then a 1B model | 0 | 107 to 108 | 0.14 s |

- A second model removed every wrong sentence. It also removed about one in three correct sentences and added
  seconds to each answer.
- The claim check added milliseconds but kept most wrong sentences. Most errors were reasoning errors, such as a
  product said to be under budget when shipping puts it over. The check also removed correct sentences that
  state a total the model calculated.
- The claim check followed by the 1B model removed most wrong sentences in 0.13 s. It removed the most correct
  sentences. It did not meet the pass rule set before the test.
- French listings now lose about as many correct sentences as English listings: 36 and 38 for Flash, 51 and 54
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
- All counts are in [results/report.json](results/report.json).

## Limits

- Offers are constructed. Icecat has no prices.
- Products are limited to electronics, computing and appliances.
- Questions and answers are in English. Each model answered once.
- Two models graded the sentences, Gemini 3.8 Flash and Qwen 27B. People did not. The models agreed on 546 of 550
  sentences. The 4 open sentences are shown as ranges. Flash also graded its own answers.
- The second model check used a different model for each answer set. Its times do not compare between the two
  tables.
- The 1B model was trained on UK legislation, not on product text.
- There are few wrong sentences, 32 to 35, all from one model.
- This is not a test of any company's production system.

## Reproduce

Requirements:

- Python 3.11 or later and [uv](https://docs.astral.sh/uv/).
- An Open Icecat account. Set `ICECAT_USERNAME`.
- An OpenAI compatible endpoint at `localhost:8317` that serves `claude-opus-4-6-thinking` and
  `gemini-3.8-flash-high`. Set `LLM_ROUTER_KEY`.
- A [Modal](https://modal.com) account for the 1B model and Qwen 27B. This run cost $0.84.

Settings can be kept in a `.env` file in the repository root. The tests need none of them: `uv run pytest`.

```sh
uv run python scripts/fetch_icecat.py rebuild
uv run python scripts/answer.py battery flash
uv run python scripts/check.py claims battery flash
uv run --with modal modal run scripts/judge1b.py::run --split battery --model flash
uv run python scripts/grade.py sheet
uv run --with modal modal run scripts/qwen27.py::run
uv run python scripts/report.py
```

The `battery` folder holds the product IDs, offers, questions, prompt and a hash of every listing.
`fetch_icecat.py rebuild` reports any listing that changed since the test.

## Data and licence

- Product data: [Open Icecat](https://icecat.biz), Open Content Licence, provided as is. Listing text is not
  included. Each user downloads it with their own account.
- Code, offers and questions: [AGPL-3.0](LICENSE).
- Claim check: [legal-rag-verifier](https://github.com/azterizm/legal-rag-verifier) 0.3.0.
