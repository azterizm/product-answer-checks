# Test 1: what each check costs on product answers (method fixed 2026-10-22, before any download)

The build plan (Joko, §2) made exact. Everything below is fixed before the draw, the battery and any model call;
a change after this commit is listed under "Changes" with its date and reason.

## Question

On product questions answered from listings, what does each check cost in wrong sentences kept, right sentences
removed and seconds added per answer?

## 1. Products

**Source.** Open Icecat through its JSON API (`live.icecat.biz/api`, username only; the Open Content Licence
accepted by the account holder). Each product is fetched in English and in French with the same Icecat ID.

**Draw.** Icecat IDs drawn uniformly from [90,000,000, 135,000,000) with `random.Random(20261022)`, in draw order,
until the battery is full or 40,000 IDs are drawn. The range holds the recent catalogue (about 4 % of IDs are Open
Icecat products in a probe of 300). Requests: at most 32 at a time, about 40 per second, under the fair-use 50 (see Changes).

**Eligible** when all hold:

1. Both sheets return `OK`, and the French sheet's local title is in French.
2. A word of the English category name, lower-cased, starts with one of: laptop, notebook, tablet, pc, workstation, monitor,
   smartphone, mobile phone, smartwatch, headphone, headset, earphone, speaker, soundbar, tv, television, projector,
   camera, e-book, power bank, console, printer, keyboard, mouse, mice, router, hard drive, solid state drive,
   flash drive, vacuum, coffee, kettle, toaster, blender, food processor, microwave, hair, shaver, toothbrush, iron,
   fryer, fan, air purifier, heater, dishwasher, washing machine, refrigerator, fridge, freezer, oven, dryer, hob,
   cooker; and none of: spare part, accessor, case, cover, cable, mount, bag, part, screen protector, warranty,
   replacement, filter, stand, dock, adapter, ink, toner, licence, license (so "hair" does not match "chairs").
   Fashion and grocery are not covered.
3. Both sheets have at least 15 features with a value.
4. At most 2 products per (brand, category) are taken; later ones are skipped.

**Sets, in draw order of eligible products.**

- **Dev:** the first 10. Used only to check the prompt and the scripts; never in the battery.
- **Type E (energy):** the next 12 whose sheets have an energy efficiency class (feature 2705) and a feature whose
  English name starts with "Energy consumption" or "Power consumption".
- **Type B (fit under budget):** the next 12 whose category has two more eligible, unused products such that the
  three share a numeric feature (below) with values v1 ≥ v2 > v3; the companions are the first two such products in
  draw order, used in no other question.
- **Type C (spec facts):** the next 12 with a weight (feature 94) and one more numeric feature.
- **Types A and D:** the rest in draw order, alternately A then D, 12 each.

A **numeric feature** has an English value that is a number with at most one unit ("80 Wh", "2.4 kg", "43 dB",
"16"), and is not weight, not in the packaging group, and not a yes/no. Weight is used only in type C.
- If the draw ends short: what passes is used if at least 40 products (plan, risks), and the count is stated.

## 2. Listings and offers

**Listing** (one per product and language): the sheet's title, its short summary, then every feature as
`Group — Name: Value`, from that language's sheet. The marketing description and images are left out.

**Offers are constructed, not real** (Icecat has no prices), by `scripts/build_battery.py` with
`random.Random(20261022 + Icecat ID)`. Offers are as of **Monday 2 November 2026**, stated above them. Each product
gets 3 or 4 offers from fictional merchants (Boutique A–D in French, Shop A–D in English): price, an optional old
price, shipping, an estimated delivery date, a return window and stock. The French and English offers hold the same
values, written as merchants in each language write them (French with no-break spaces before "€" and between digit
groups):

- French: `Boutique A — Prix : 279,99 € (au lieu de 329,99 €) — Livraison : 4,99 € — Livraison estimée le jeudi 5
  novembre 2026 — Retours sous 30 jours — En stock`
- English: `Shop A — Price: €279.99 (was €329.99) — Shipping: €4.99 — Estimated delivery Thursday 5 November 2026 —
  Returns within 30 days — In stock`

Base price: log-uniform in a band by category (small appliances €19–199; electronics and computing €49–1,999;
large appliances €249–1,299), ending in .99 or .00. Offers vary around it by up to ±15 %. Shipping is free or
€2.99–9.99; delivery 1 to 8 working days after the offers' date; returns 14, 30 or 60 days.

**Traps** (each offer set holds at least one): a struck-through old price; a lower price whose shipping makes the
total higher; the cheapest offer out of stock. Type A always has 4 offers and all three. Types C, D and E have the
struck-through price and one of the other two (3 offers), plus one plain offer when the seed gives 4. Type B has
one offer per product: the shipping trap below, and a struck-through old price on one of the three.

## 3. Questions (in English for both halves, so only the listing language differs)

| Type | Template (phrasing picked by the seed where there are several) |
|---|---|
| A. Best price | "Where is this cheapest, delivered, from a shop that has it in stock?" / "Which shop gives me the lowest total price including delivery?" / "Where should I buy this to pay the least, shipping included?" |
| B. Fit under budget | "Which of these three has a {feature} of at least {value} and costs under €{budget} delivered?" Three listings, one offer each. The feature is the numeric one the three share with the most distinct values (ties by the seed); the threshold is the second-highest value, so two qualify on the spec; of those, one is under the budget delivered and the other only before shipping; the third is the cheapest. |
| C. Spec facts | "How much does it weigh, and what is its {feature}?" The feature is drawn by the seed from the numeric features other than weight and packaging. |
| D. Delivery and returns | "Which shop delivers by {weekday day month} and takes returns for at least 30 days?" Offers built so that exactly one in-stock offer meets both: one other delivers in time with a shorter return window, one other has the returns but arrives later; an out-of-stock offer meets neither. |
| E. Energy | "What is its energy efficiency class and its {consumption feature}?" |

Offers, traps and questions are written without looking at the claim check's patterns, and the checker is not run
on any battery row before the seal.

**Before the seal**, a census of the French listings for the forms the claim check is known not to read
(`legal-rag-verifier` API.md §6): digit groups split by a plain space and not followed by a decimal comma or a
currency ("5 000 mAh"), and a comma before exactly three digits. It uses its own patterns, not the checker. Counts
are reported; if any appear, you decide whether the verifier is fixed first, under its own gate.

## 4. Seal

`battery/seal.json`, written before any generator call: the SHA-256 of each raw sheet (FR and EN), each rendered
listing, the offers, the questions and the prompt, the Icecat fetch date, and the draw's counts. **Committed:**
the Icecat IDs, categories, brands, types, companions, hashes, offers, questions, the prompt and
`scripts/fetch_icecat.py`, which rebuilds the sheets with the reader's own Open Icecat username. **Not committed:**
sheet text and raw sheets (`data/`, git-ignored), as the plan's default.

## 5. Generation

- **Claude Opus 4.6** (`claude-opus-4-6-thinking`) and **Gemini 3.8 Flash** (`gemini-3.8-flash-high`) through the
  router, as in the verifier's stop 71: one run each, router defaults, 4 at a time. 240 answers.
- **Prompt** (exact text sealed in `battery/prompt.txt`): a system message "You are a shopping assistant. Answer
  the customer's question using only the product listings and offers below. Answer in English, in at most 4
  sentences." followed by the listings and offers; the question as the user message.
- Answers in English for both halves (prose checks are English only; French answers are a stated limit).
- Time per answer is recorded. Errors are retried once; an answer still missing is reported as missing.

## 6. Arms

Post-hoc checks on the same 240 answers, split into sentences by the verifier's `split_sentences`. A rejected
sentence is dropped, as in stop 71. The premise for every check is the question's listings and offers.

| Arm | Check | Runs on |
|---|---|---|
| 0 | No check | — |
| 1 | The other model checks each sentence (stop 71's `check`, copied with its source commit) | Router |
| 2 | Claim check, `legal-rag-verifier` 0.3.0 build | CPU |
| 2b | Claim check, 0.2.0, French half only | CPU |
| 2t | Claim check with totals (`totals="line"`), verifier build `9ab6603` (stop 76, unreleased); added 2026-10-23, see Changes | CPU |
| 3 | Claim check, then the stop 64 1B judge (`azterizm/minicpm5-1b-clause-judge`) on the sentences it passes | CPU, then one Modal L40S (SGLang) |
| 3t | Claim check with totals (arm 2t), then the same 1B judge with `totals="line"`; added 2026-10-23, see Changes | CPU, then one Modal L40S (SGLang) |

The 1B judge is used as published; it has never seen a product sentence, and is reported so.

## 7. Grading

- Gemini 3.8 Flash grades every sentence of arm 0 against the question, the listings and the offers: **right**,
  **wrong** or **unsure** (the verifier's `live68_grade.py` flow, copied; Opus 5.5 was planned as a second grader,
  see Changes). A price sentence is one with "€", "EUR" or "euro".
- **Wrong:** it states something the listings and offers do not support, or contradicts them. A sentence that
  gives a struck-through price as the current price is wrong. A total that leaves out shipping when the question
  asks for the delivered price is wrong.
- You grade every unsure sentence by hand, and a random 30 sentences per generator (seed 20261030) as a check on
  the grader (stop 71 saw Opus 5.5 soft on Opus 4.6).

## 8. Report

Per generator, and for the FR and EN halves: wrong sentences kept, right sentences removed, empty answers, seconds
added per answer at the median with the hardware named per arm (router calls, CPU, L40S); a separate row for price
sentences; arm 2 against 2b on the French half (false rollbacks before and after stop 75).

**Reading rule.** It decides only the one plain sentence in the email; every number is reported either way. Arm 3
"costs less for the same safety" if, per generator, it keeps at most 2 more wrong sentences than arm 1 and removes
fewer right sentences. Otherwise the email says what happened.

## 9. Limits (stated in the README)

Constructed offers. Open Icecat categories only. English questions and answers only. One run per generator. Small
counts of wrong sentences (frontier models erred on 3 to 10 of about 500 in stop 71). The 1B judge untrained on
products. Not Joko's system, catalogue or prompts.

## 10. Spend

Router answers and checks: free. Opus 5.5 grading, about 1,000 sentences: about $2–3. Modal L40S for the 1B judge:
about $1. Neither runs before your spend go.

## Changes

- 2026-10-22, before any download: category words match at the start of a word; the order in which types B, C,
  A and D are filled, and what a numeric feature is, made exact; type B's traps and D's out-of-stock offer stated;
  French offers use no-break spaces.
- 2026-10-22, during the draw (7,718 IDs drawn, none used): requests at up to 32 at a time, about 40 a second,
  still under the fair-use 50, by your choice. The battery is read from the draw cut at the first ID that
  completes it, so the faster, batched fetch gives the same products as a one-at-a-time draw.
- 2026-10-22, before the seal: the census (`battery/census.json`) found 7 plain-space groups, all in model names
  ("Galaxy Book3 360"), read right; and 38 commas before three digits: lists written the same in both languages
  ("10,100,1000 Mbit/s") and 18 French decimals with three places ("0,965 kWh"), read as thousands. By your choice
  the verifier was fixed first, under its own gate (stop 75, `66450a6`: a lone "0," and three digits is a
  decimal); the pin moved to that commit. 5 of the 18 ("1,031", "3,584") stay thousands: a stated limit.
- 2026-10-22, after the answers began and before any grading or arm 3 call (your go on Modal and Opus 5.5, and
  your offer of Gemini 3.8 Flash):
  - **Grading by two graders**, the verifier's `live68_grade.py` flow as it stands (stop 68): Opus 5.5 and Flash
    each grade every sentence once ("yes" right, "no" wrong, "unsure"); the sentences they differ on, or either
    marks unsure, are asked again of both with a reason first; what still differs goes to you, with the 30
    random sentences per generator. Flash grades its own answers too, so agreement is reported per generator.
  - **Price sentence:** a sentence that contains "€", "EUR" or "euro" (any case). Fixed here, not by the checker.
  - **Arm 3's judge:** `azterizm/minicpm5-1b-clause-judge` adapter `run2` (stop 64) at revision `7cd8ee5`, merged
    into `openbmb/MiniCPM5-1B` at `87179e5` when the container starts, served by SGLang `v0.5.21` on one L40S,
    greedy, at most 1,024 tokens, thinking off, the short-reply prompt it was trained on (`scripts/judge.py`).
    Each answer's loop runs in the container, so its time is the L40S's; the claim check's time is added from
    arm 2 (CPU).
  - **Spend caps:** Opus 5.5 at $10 in this repo's router ledger; Modal at $3 in its own ledger. A call that could
    pass either cap is not made.
- 2026-10-22, after arm 3's dev smoke run, before the battery: arm 3 checks one answer at a time in the container,
  so an answer's time is not shared with others' calls (the dev run at 8 at a time gave 360 ms per call).
- 2026-10-22, during grading: Opus 5.5 returned HTTP 503 at the router from 15:03 and graded nothing. By your
  choice Flash is the only grader, in the flow's one-grader mode (the verifier's stop 71): its grade stands; what it
  marks unsure goes to you, with the 30 random sentences per generator, which are now the only check on the
  grader. Flash grades its own answers as well as Opus 4.6's; the report gives your agreement with it per
  generator. No Opus 5.5 call was made or paid for.
- 2026-10-22, to finish today (your call: no waiting; Opus 5.5 still HTTP 503, Opus 4.6 rate-limited for 3.7 hours):
  - **Arm 1 on Flash's answers:** Qwen 27B checks them in place of Opus 4.6: the verifier's 27B judge
    (`qwen27-ar4`, `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ` at `1b9d56b`, thinking off, the card's
    non-thinking sampling, the short reply prompt), on one L40S, one answer at a time. The 83 Opus 4.6 labels
    already taken are not used. Arm 1's judge and hardware therefore differ by generator (Flash on the router for
    Opus 4.6's answers; Qwen 27B on an L40S for Flash's), and the report says so.
  - **A second grader:** Qwen 27B grades all 550 sentences on the same server, in Opus 5.5's place in the
    two-grader flow (first pass; both again, reason first, where they differ). What still differs is reported as
    a range (counted right, then wrong), not waited on; your 60 random checks become optional.
- 2026-10-22, after the runs: the verifier released as 0.3.0 (its code is `66450a6`, the build every arm ran on);
  `pyproject.toml` now asks for `legal-rag-verifier>=0.3.0` from PyPI instead of the local commit.
- 2026-10-22, your hand grades: 52 of the 60 random sentences (26 per generator) and 2 of the 4 the graders left
  open. You agree with the graders on 50 of 52 (26 of 26 on Flash's answers, 24 of 26 on Opus 4.6's; the two
  differences go opposite ways). As fixed in §7, the random sentences check the graders and do not replace their
  grades; your grades decide the 2 open sentences. 2 stay open, both from Opus 4.6, reported as ranges.
- 2026-10-22, the last 2 open sentences: Q083A1 and Q093B1, Opus 4.6 on A10 in English and in French, the same
  claim ("The cheapest option including shipping is Shop C ... for a total of €133.99"; the cheaper Shop D is out of
  stock, said in the answer's next sentence). You graded both right, after the report with them as ranges was
  published. Every count is now one number; the reading rule is unchanged (false on both generators).
- 2026-10-23, before arm 2t ran on dev or the battery (your go, stop 76 in the verifier): **arm 2t, the claim
  check with totals.** 94 of arm 2's 106 right sentences removed on Opus 4.6 state an offer's price plus shipping,
  which the listing never prints. The verifier's build `9ab6603` adds one opt-in rule, `totals="line"`: a money
  figure equal to two money figures of one currency on one line of the source, added, neither inside brackets (so
  an offer's price plus its shipping, not its old price plus shipping), is grounded. Built on the 20 dev questions;
  the battery's sentences were not read for it.
  - Runs: dev, then the battery **once**, both generators, on CPU, through `scripts/claimcheck.py --totals` in an
    environment with that build only. The same build with totals off must give arm 2's verdicts exactly, on dev
    and on the battery (a control; any difference stops the run).
  - Dev, before the battery: every sentence arm 2 rolled back whose only failed figures are one offer's price plus
    non-free shipping is kept; "€316.99 (€311.00 + €4.99)" (dev09-en, a wrong total) still rolls back; no other
    verdict changes.
  - **The sums gap:** right sentences arm 2 removed that state a euro figure equal to one offer's price plus its
    shipping, shipping not free, read from the committed offers (`battery/questions.jsonl`), not from the checker.
    Counted per generator; on Opus 4.6 it is reported whether or not it is 94. **Recovered:** those arm 2t keeps.
    The ones not recovered are listed with the reason arm 2t gives.
  - Arm 2t gets the full row of §8 (wrong kept, wrong prices kept, right removed, empty, seconds), and the wrong
    sentences it keeps that arm 2 removed are listed by question. The sentences and grades are test 1's (same
    answers, same split); nothing is graded again.
  - No reading rule: the counts are reported as they come. Arm 3 with totals (the 1B judge after arm 2t) needs
    Modal (about $0.25) and is not run without your go.
- 2026-10-23, arm 2t ran: dev passed as fixed (the control equal to 0.3.0; 16 and 12 verdicts changed, exactly the
  expected ones; "€316.99" still rolled back). The battery once, control equal to arm 2's stored verdicts. Opus
  4.6: wrong kept 30 (12 prices), right removed 25, empty 2, against arm 2's 21 (3), 106, 9. The sums gap is 94,
  81 recovered; the other 13 are 12 type B value-swap rollbacks and a bare number. 9 wrong sentences newly kept,
  each a right total inside a wrong claim. Flash: right removed 11 (75), empty 1 (8), none wrong.
- 2026-10-23, before any call (your go on Modal): **arm 3t, arm 3 on arm 2t.** The sentences arm 2t passes go to
  the 1B judge exactly as in arm 3 (same merged weights, server, prompt, `STRICT_FIGURES`, one answer at a time),
  with `totals="line"` added to the judge's config, so the verifier inside the container is the build `9ab6603`
  (shipped from an environment holding only that build; the run stops if the container's config lacks
  `totals="line"`). Dev first on Opus 4.6's 20 answers as the smoke run, then the battery once, both generators.
  Time is arm 2t's claim check plus the container's judge time. Reported: the §8 row, the wrong sentences arm 3t
  keeps that arm 3 removed, and the reading rule of §8 computed for 3t against arm 1 beside arm 3's, as
  information: that rule was fixed for arm 3, not for 3t. Spend: about $0.25, under the ledger's $3 cap.

