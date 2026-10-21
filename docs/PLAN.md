# Test 1: four ways to check a product answer

Status: **draft, not sealed.** Written 2026-10-21 from the Joko outreach note §4–§6. The detailed build plan
(§2) is to be confirmed before the battery is built; every point marked *to confirm* waits on it.

## Question

When a hosted model answers a shopping question from the product listing and an offer, how many wrong
sentences and wrong prices does each check catch, how many correct sentences does it remove, and how much time
does it add per answer?

## Battery

- 60 products from Open Icecat, each with its French and its English sheet: 120 questions, one per product and
  language. *To confirm:* categories, the question types (A, B, C; C is spec-sheet questions), how products are
  drawn (seeded), and the Open Content Licence's terms for what may be committed.
- Each product gets a constructed offer (price, delivery time, warranty) beside its sheet. French offers are
  written as French merchants show them ("279,99 €", "3 à 5 jours ouvrés", "Garantie 2 ans"), not "EUR 279,99"
  (verifier API.md §6).
- Before the battery is sealed: count the figures in the French sheets that the claim check does not read (the
  API.md §6 limits: "5 000 mAh" with a plain space, French number words). Any found are fixed in the verifier
  under its own gate first, then 0.3.0 is tagged.
- The battery is sealed (sha256 recorded) before any answer is generated.

## Answers

Claude Opus 4.6 and Gemini 3.8 Flash answer every question with the sheet and the offer in the prompt (240
answers). *To confirm:* the prompt, temperature, answer language.

## Arms

| Arm | Check |
|---|---|
| 0 | No check |
| 1 | The other model checks each sentence against the listing (as stop 71 in the verifier) |
| 2 | The claim check (`legal-rag-verifier` 0.3.0): figures, dates and units |
| 3 | The claim check, then the 1B judge (`azterizm/minicpm5-1b-clause-judge`) on the sentences it passes |

*To confirm:* arm 2b (the claim check of 0.2.0, without the French formats) as a baseline.

## Grading

One grading pass by Claude Opus 5.5 over every sentence: right or wrong against the sheet and offer, and
whether it states a price. Grades the grader is unsure of go to you. *To confirm:* the rubric and the share
you grade by hand.

## Measures

Per arm and model: wrong sentences left, wrong prices left, correct sentences removed, time added per answer
(the claim check timed on the new code; the 1B judge on a GPU, said so). Differences reported with paired tests
as in the verifier's stop 71.

## Spend

Paid calls: Opus 4.6 and Flash answers (240), the cross-model check (arm 1), the Opus 5.5 grading pass, and a
GPU for the 1B judge (arm 3). None before your spend caps and go.
