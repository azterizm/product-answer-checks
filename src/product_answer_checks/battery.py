"""Questions, premises and the seal of test 1's battery (METHOD.md §2–§4)."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar

from product_answer_checks import offers as of
from product_answer_checks import sheets as sh
from product_answer_checks.selection import SEED, Selection

Sheet = sh.Sheet
Sheets = Callable[[int, str], Sheet]
LANGS = ("fr", "en")
PROMPT = (
    "You are a shopping assistant. Answer the customer's question using only the product listings "
    "and offers below. Answer in English, in at most 4 sentences.\n\n{premise}"
)
A_PHRASES = (
    "Where is this cheapest, delivered, from a shop that has it in stock?",
    "Which shop gives me the lowest total price including delivery?",
    "Where should I buy this to pay the least, shipping included?",
)
DEV_TYPES = ("A", "C", "D")
T = TypeVar("T")


@dataclass
class Question:
    """One question in one listing language; ``premise`` (not committed) is listings and offers."""

    qid: str
    product: int
    type: str
    lang: str
    question: str
    products: list[int]
    offers: list[dict[str, Any]]
    offer_lines: list[str]
    intended: dict[str, Any]
    premise: str = field(repr=False)

    def record(self) -> dict[str, Any]:
        """The committed row: everything but the premise, which quotes the sheets."""
        out = asdict(self)
        out["premise_sha256"] = sha256(out.pop("premise"))
        return out


def need(value: T | None, what: str) -> T:
    """``value``, or a ValueError naming what the sheet lacks (selection keeps such out)."""
    if value is None:
        raise ValueError(f"the sheet has no {what}")
    return value


def sha256(text: str | bytes) -> str:
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def _name(feature: sh.Feature) -> str:
    """A feature name inside a sentence: lower-case its first letter unless it starts an acronym."""
    name = feature.name
    return name[0].lower() + name[1:] if len(name) > 1 and name[1].islower() else name


def _offer_dict(offer: of.Offer) -> dict[str, Any]:
    out = asdict(offer)
    out["delivery"] = offer.delivery.isoformat()
    return out


def _single_premise(listing: str, offers: list[of.Offer], lang: str) -> tuple[str, list[str]]:
    lines = of.lines(offers, lang)
    return "\n".join([listing, "", of.header(lang), *lines]), lines


def _letter(offers: list[of.Offer], role: str) -> str:
    return "ABCD"[next(i for i, o in enumerate(offers) if o.role.startswith(role))]


def single_questions(
    qid: str, type_: str, pid: int, sheets: Sheets, *, seed: int | None = None
) -> list[Question]:
    """Types A, C, D and E: one product, its offers, one question; same values in both languages.
    ``seed`` defaults to test 1's (``SEED + pid``)."""
    en = sheets(pid, "en")
    rng = random.Random(SEED + pid if seed is None else seed)
    offers = of.single(type_, sh.price_band(sh.category(en)), rng)
    intended: dict[str, Any]
    if type_ == "A":
        question = rng.choice(A_PHRASES)
        intended = {"shop": _letter(offers, "best")}
    elif type_ == "C":
        weight = need(sh.feature(en, sh.WEIGHT), "weight")
        fid = rng.choice(sorted(sh.numeric_features(en)))
        other = need(sh.feature(en, fid), f"feature {fid}")
        question = f"How much does it weigh, and what is its {_name(other)}?"
        intended = {"weight": weight.value, "feature": fid, "value": other.value}
    elif type_ == "D":
        due = of.deadline(offers)
        question = (
            f"Which shop delivers by {of.day_month_en(due)} and takes returns for at least "
            f"{of.RETURNS_WANTED} days?"
        )
        intended = {"shop": _letter(offers, "d-meets"), "by": due.isoformat()}
    elif type_ == "E":
        klass, use = need(sh.energy_features(en), "energy class and consumption")
        question = f"What is its energy efficiency class and its {_name(use)}?"
        intended = {"class": klass.value, "feature": use.id, "consumption": use.value}
    else:
        raise ValueError(f"not a single-product type: {type_}")
    out = []
    for lang in LANGS:
        premise, lines = _single_premise(sh.listing(sheets(pid, lang)), offers, lang)
        out.append(
            Question(
                qid=f"{qid}-{lang}",
                product=pid,
                type=type_,
                lang=lang,
                question=question,
                products=[pid],
                offers=[_offer_dict(o) for o in offers],
                offer_lines=lines,
                intended=intended,
                premise=premise,
            )
        )
    return out


def budget_questions(
    qid: str,
    pid: int,
    companions: Sequence[int],
    fid: str,
    sheets: Sheets,
    *,
    seed: int | None = None,
) -> list[Question]:
    """Type B: three products sharing a numeric feature, one offer each (METHOD §3)."""
    rng = random.Random(SEED + pid if seed is None else seed)
    order = [pid, *companions]
    rng.shuffle(order)
    values = {p: sh.numeric_features(sheets(p, "en"))[fid].value for p in order}
    ranked = sorted(order, key=lambda p: values[p], reverse=True)
    qualifying, cheap = ranked[:2], ranked[2]
    under = rng.choice(qualifying)
    role = {under: "under", next(p for p in qualifying if p != under): "over", cheap: "cheap"}
    budget, offers = of.budget_three(
        sh.price_band(sh.category(sheets(pid, "en"))),
        (role[order[0]], role[order[1]], role[order[2]]),
        rng,
    )
    threshold = need(sh.feature(sheets(ranked[1], "en"), fid), f"feature {fid}")
    question = (
        f"Which of these three has a {_name(threshold)} of at least {threshold.value} and costs "
        f"under €{budget // 100:,} delivered?"
    )
    intended = {
        "product": order.index(under) + 1,
        "feature": fid,
        "threshold": threshold.value,
        "budget": budget,
    }
    out = []
    for lang in LANGS:
        word = "Produit" if lang == "fr" else "Product"
        blocks, lines = [], []
        for i, (p, offer) in enumerate(zip(order, offers, strict=True)):
            line = (of.line_fr if lang == "fr" else of.line_en)("ABC"[i], offer)
            lines.append(line)
            blocks.append(
                "\n".join(
                    [f"{word} {i + 1}", sh.listing(sheets(p, lang)), "", of.header(lang), line]
                )
            )
        out.append(
            Question(
                qid=f"{qid}-{lang}",
                product=pid,
                type="B",
                lang=lang,
                question=question,
                products=order,
                offers=[_offer_dict(o) for o in offers],
                offer_lines=lines,
                intended=intended,
                premise="\n\n".join(blocks),
            )
        )
    return out


def questions(selection: Selection, sheets: Sheets) -> tuple[list[Question], list[Question]]:
    """The battery's questions and the dev set's, both languages."""
    battery: list[Question] = []
    for type_ in ("A", "B", "C", "D", "E"):
        for n, pid in enumerate(selection.types[type_], 1):
            qid = f"{type_}{n:02d}"
            if type_ == "B":
                battery += budget_questions(
                    qid, pid, selection.companions[pid], selection.b_feature[pid], sheets
                )
            else:
                battery += single_questions(qid, type_, pid, sheets)
    dev: list[Question] = []
    for n, pid in enumerate(selection.dev, 1):
        en = sheets(pid, "en")
        type_ = DEV_TYPES[(n - 1) % len(DEV_TYPES)]
        if type_ == "C" and not (sh.feature(en, sh.WEIGHT) and sh.numeric_features(en)):
            type_ = "A"
        dev += single_questions(f"dev{n:02d}", type_, pid, sheets)
    return battery, dev


# ---------------------------------------------------------------- the census (METHOD §3)
CENSUS = {
    "plain-space groups": re.compile(r"(?<![\d,.])\d{1,3}(?: \d{3})+(?![\d,]| ?(?:€|EUR))"),
    "comma before three digits": re.compile(r"(?<![\d,])\d{1,3},\d{3}(?!\d)"),
}


def census(texts: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Counts of the forms the claim check does not read, with up to 10 examples each."""
    out: dict[str, dict[str, Any]] = {}
    for name, pattern in CENSUS.items():
        hits = [(key, m.group(0)) for key, text in texts.items() for m in pattern.finditer(text)]
        out[name] = {
            "count": len(hits),
            "listings": len({k for k, _ in hits}),
            "examples": [f"{k}: {v}" for k, v in hits[:10]],
        }
    return out


def dumps(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True)
