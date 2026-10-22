"""Constructed offers, with their traps, written in French and in English (METHOD.md §2 and §3)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from datetime import date, timedelta

AS_OF = date(2026, 11, 2)  # Monday
SHIPPING = (299, 399, 499, 599, 699, 799, 899, 999)  # cents
RETURNS = (14, 30, 60)
RETURNS_WANTED = 30
HALF = 0.5
WEEKEND = 5  # Saturday, by date.weekday()
NBSP = " "
_DAYS_EN = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_DAYS_FR = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_MONTHS_EN = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_MONTHS_FR = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)


@dataclass(frozen=True)
class Offer:
    """One merchant's offer; money in cents."""

    price: int
    shipping: int
    delivery: date
    returns: int
    in_stock: bool = True
    old_price: int | None = None
    role: str = "plain"  # best, oos, shipping, struck, plain, or a type D / B role

    @property
    def total(self) -> int:
        return self.price + self.shipping


# ---------------------------------------------------------------- money and dates
def price_at(euros: float, rng: random.Random) -> int:
    """A shop price near ``euros``, ending in .99 or .00 (cents)."""
    whole = max(1, round(euros))
    return whole * 100 - 1 if rng.random() < HALF else whole * 100


def base_price(band: tuple[int, int], rng: random.Random) -> float:
    lo, hi = band
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def working_days_after(start: date, n: int) -> date:
    day = start
    while n:
        day += timedelta(days=1)
        if day.weekday() < WEEKEND:
            n -= 1
    return day


def money_en(cents: int) -> str:
    return f"€{cents / 100:,.2f}"


def money_fr(cents: int) -> str:
    euros, rest = divmod(cents, 100)
    return f"{euros:,}".replace(",", NBSP) + f",{rest:02d}{NBSP}€"


def date_en(day: date) -> str:
    return f"{_DAYS_EN[day.weekday()]} {day.day} {_MONTHS_EN[day.month - 1]} {day.year}"


def date_fr(day: date) -> str:
    number = "1er" if day.day == 1 else str(day.day)
    return f"{_DAYS_FR[day.weekday()]} {number} {_MONTHS_FR[day.month - 1]} {day.year}"


def day_month_en(day: date) -> str:
    return f"{_DAYS_EN[day.weekday()]} {day.day} {_MONTHS_EN[day.month - 1]}"


# ---------------------------------------------------------------- writing
def line_en(letter: str, offer: Offer) -> str:
    price = money_en(offer.price)
    if offer.old_price is not None:
        price += f" (was {money_en(offer.old_price)})"
    shipping = money_en(offer.shipping) if offer.shipping else "free"
    stock = "In stock" if offer.in_stock else "Out of stock"
    return (
        f"Shop {letter} — Price: {price} — Shipping: {shipping} — Estimated delivery "
        f"{date_en(offer.delivery)} — Returns within {offer.returns} days — {stock}"
    )


def line_fr(letter: str, offer: Offer) -> str:
    price = money_fr(offer.price)
    if offer.old_price is not None:
        price += f" (au lieu de {money_fr(offer.old_price)})"
    shipping = money_fr(offer.shipping) if offer.shipping else "gratuite"
    stock = "En stock" if offer.in_stock else "Rupture de stock"
    return (
        f"Boutique {letter} — Prix{NBSP}: {price} — Livraison{NBSP}: {shipping} — "
        f"Livraison estimée le {date_fr(offer.delivery)} — Retours sous {offer.returns} jours — "
        f"{stock}"
    )


def header(lang: str) -> str:
    if lang == "fr":
        return f"Offres relevées le {date_fr(AS_OF)}{NBSP}:"
    return f"Offers as of {date_en(AS_OF)}:"


def lines(offers: list[Offer], lang: str) -> list[str]:
    write = line_fr if lang == "fr" else line_en
    return [write("ABCD"[i], o) for i, o in enumerate(offers)]


# ---------------------------------------------------------------- construction
def _plain(rng: random.Random) -> tuple[date, int]:
    return working_days_after(AS_OF, rng.randint(1, 8)), rng.choice(RETURNS)


def _four(base: float, rng: random.Random) -> dict[str, Offer]:
    """The best delivered offer and its three traps, built around ``base`` euros."""
    p_best = price_at(base, rng)
    s_best = rng.choice((0, 299))
    p_ship = p_best - 100 * rng.randint(1, 4)
    s_ship = rng.choice((799, 899, 999))  # always more than the price gap plus the best's shipping
    p_oos = p_ship - 100 * max(1, round(base * rng.uniform(0.05, 0.15)))
    p_struck = p_best + 100 * max(1, round(base * rng.uniform(0.03, 0.10)))
    old = price_at(p_struck / 100 * rng.uniform(1.1, 1.3), rng)
    old = max(old, p_struck + 100)
    offers = {
        "best": Offer(p_best, s_best, *_plain(rng), role="best"),
        "shipping": Offer(p_ship, s_ship, *_plain(rng), role="shipping"),
        "oos": Offer(p_oos, rng.choice((0, *SHIPPING)), *_plain(rng), in_stock=False, role="oos"),
        "struck": Offer(
            p_struck, s_best + rng.choice((0, 200, 400)), *_plain(rng), old_price=old, role="struck"
        ),
    }
    p_plain = p_best + 100 * max(1, round(base * rng.uniform(0.03, 0.12)))
    offers["plain"] = Offer(p_plain, s_best + rng.choice((0, 200, 400)), *_plain(rng))
    return offers


def single(type_: str, band: tuple[int, int], rng: random.Random) -> list[Offer]:
    """Offers for one product (types A, C, D, E), in shop order."""
    four = _four(base_price(band, rng), rng)
    if type_ == "A":
        chosen = [four[r] for r in ("best", "shipping", "oos", "struck")]
    else:
        trap = rng.choice(("shipping", "oos"))
        chosen = [four["best"], four[trap], four["struck"]]
        if rng.random() < HALF or (type_ == "D" and trap == "oos"):
            chosen.append(four["plain"])
    if type_ == "D":
        chosen = _dated(chosen, rng)
    rng.shuffle(chosen)
    return chosen


def deadline(offers: list[Offer]) -> date:
    """The date a type D question asks for: the qualifying offer's delivery date."""
    return next(o.delivery for o in offers if o.role.startswith("d-meets"))


def _dated(offers: list[Offer], rng: random.Random) -> list[Offer]:
    """Type D: exactly one in-stock offer delivers by the deadline and takes 30-day returns."""
    days = rng.randint(3, 5)
    due = working_days_after(AS_OF, days)
    in_stock = [o for o in offers if o.in_stock]
    rng.shuffle(in_stock)
    meets, early, late, *others = in_stock
    out = {
        id(meets): replace(
            meets, delivery=due, returns=rng.choice((30, 60)), role=f"d-meets/{meets.role}"
        ),
        id(early): replace(
            early,
            delivery=working_days_after(AS_OF, rng.randint(1, days)),
            returns=14,
            role=f"d-early/{early.role}",
        ),
        id(late): replace(
            late,
            delivery=working_days_after(due, rng.randint(1, 3)),
            returns=rng.choice((30, 60)),
            role=f"d-late/{late.role}",
        ),
    }
    for o in [*others, *(o for o in offers if not o.in_stock)]:
        out[id(o)] = replace(o, delivery=working_days_after(due, rng.randint(1, 3)), returns=14)
    return [out[id(o)] for o in offers]


def budget_three(
    band: tuple[int, int], roles: tuple[str, str, str], rng: random.Random
) -> tuple[int, list[Offer]]:
    """Type B: a budget (cents) and one offer per product, for roles ``under``, ``over``, ``cheap``.

    ``under`` meets the spec and the budget delivered; ``over`` meets the spec and the budget only
    before shipping; ``cheap`` fails the spec and is the cheapest. One shows a struck-through price.
    """
    base = base_price(band, rng)
    budget = math.ceil(base * 1.1 / 10) * 10 * 100
    p_under = min(price_at(budget / 100 * (1 - rng.uniform(0.05, 0.15)), rng), budget - 100)
    s_under = rng.choice((0, 299, 499))
    if p_under + s_under >= budget:
        s_under = 0
    p_over = price_at(budget / 100 - rng.randint(1, 3), rng)
    s_over = rng.choice((499, 599, 699, 799, 899, 999))
    p_cheap = price_at(min(p_under, p_over) / 100 * (1 - rng.uniform(0.05, 0.2)), rng)
    p_cheap = min(p_cheap, min(p_under, p_over) - 100)
    by_role = {
        "under": Offer(p_under, s_under, *_plain(rng), role="under"),
        "over": Offer(p_over, s_over, *_plain(rng), role="over"),
        "cheap": Offer(p_cheap, rng.choice((0, *SHIPPING)), *_plain(rng), role="cheap"),
    }
    struck = rng.choice(list(by_role))
    o = by_role[struck]
    old = max(price_at(o.price / 100 * rng.uniform(1.1, 1.3), rng), o.price + 100)
    by_role[struck] = replace(o, old_price=old)
    return budget, [by_role[r] for r in roles]
