"""Which drawn products go to the dev set and to each question type (METHOD.md §1, "Sets")."""

from __future__ import annotations

import itertools
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from product_answer_checks.sheets import Quantity

SEED = 20261022
DEV = 10
PER_TYPE = 12
TYPES = ("A", "B", "C", "D", "E")
PER_BRAND_CATEGORY = 2
MIN_BATTERY = 40


@dataclass(frozen=True)
class Candidate:
    """An eligible product, as selection needs it."""

    id: int
    category: str
    brand: str
    energy: bool
    weight: bool
    numeric: Mapping[str, Quantity] = field(default_factory=dict)


@dataclass
class Selection:
    dev: list[int] = field(default_factory=list)
    types: dict[str, list[int]] = field(default_factory=lambda: {t: [] for t in TYPES})
    companions: dict[int, list[int]] = field(default_factory=dict)
    b_feature: dict[int, str] = field(default_factory=dict)

    @property
    def battery(self) -> int:
        return sum(len(ids) for ids in self.types.values())

    @property
    def complete(self) -> bool:
        return len(self.dev) == DEV and all(len(ids) == PER_TYPE for ids in self.types.values())


def b_feature(three: Sequence[Candidate], seed: int) -> str | None:
    """The numeric feature the three share, in one unit, with v1 ≥ v2 > v3, most distinct values."""
    shared = set(three[0].numeric).intersection(*(c.numeric for c in three[1:]))
    fits: list[tuple[int, str]] = []
    for fid in sorted(shared):
        quantities = [c.numeric[fid] for c in three]
        if len({q.unit for q in quantities}) != 1:
            continue
        values = sorted((q.value for q in quantities), reverse=True)
        if values[1] > values[2]:
            fits.append((len(set(values)), fid))
    if not fits:
        return None
    best = max(n for n, _ in fits)
    return random.Random(seed).choice([fid for n, fid in fits if n == best])


def select(candidates: Sequence[Candidate]) -> Selection:
    """Fill the dev set, then E, B, C, then A and D alternately, all in draw order."""
    seen: Counter[tuple[str, str]] = Counter()
    pool = []
    for c in candidates:
        seen[c.brand, c.category] += 1
        if seen[c.brand, c.category] <= PER_BRAND_CATEGORY:
            pool.append(c)
    out = Selection(dev=[c.id for c in pool[:DEV]])
    rest = pool[DEV:]
    used: set[int] = set()

    def take(type_: str, c: Candidate) -> None:
        out.types[type_].append(c.id)
        used.add(c.id)

    for c in rest:
        if len(out.types["E"]) < PER_TYPE and c.energy:
            take("E", c)
    for c in rest:
        if len(out.types["B"]) == PER_TYPE:
            break
        if c.id in used:
            continue
        others = [o for o in rest if o.category == c.category and o.id != c.id and o.id not in used]
        for pair in itertools.combinations(others, 2):
            fid = b_feature([c, *pair], SEED + c.id)
            if fid is not None:
                take("B", c)
                used.update(o.id for o in pair)
                out.companions[c.id] = [o.id for o in pair]
                out.b_feature[c.id] = fid
                break
    for c in rest:
        if len(out.types["C"]) < PER_TYPE and c.id not in used and c.weight and c.numeric:
            take("C", c)
    turn = itertools.cycle("AD")
    for c in rest:
        if c.id in used:
            continue
        open_types = [t for t in "AD" if len(out.types[t]) < PER_TYPE]
        if not open_types:
            break
        type_ = next(turn)
        take(type_ if type_ in open_types else open_types[0], c)
    return out
