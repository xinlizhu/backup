from __future__ import annotations

import random
import re
from collections import defaultdict

from cir_qdiag.schemas import SampleRecord


KEYWORDS: dict[str, tuple[str, ...]] = {
    "attribute_quantity": (
        "one",
        "two",
        "three",
        "several",
        "many",
        "fewer",
        "more",
        "all",
        "only",
    ),
    "attribute_property": (
        "red",
        "blue",
        "green",
        "wooden",
        "metal",
        "large",
        "small",
        "long",
        "short",
    ),
    "predicate_relation": (
        "left",
        "right",
        "behind",
        "in front",
        "next to",
        "above",
        "below",
        "between",
    ),
    "predicate_action": (
        "sitting",
        "standing",
        "holding",
        "walking",
        "running",
        "looking",
        "facing",
    ),
    "explicit_edit_language": ("add", "remove", "without", "replace", "include"),
}


def sample_records(
    records: list[SampleRecord],
    limit: int,
    seed: int,
    strategy: str = "random",
    random_count: int | None = None,
    targeted_count: int | None = None,
) -> list[SampleRecord]:
    if limit > len(records):
        raise ValueError(f"requested {limit} samples from only {len(records)} records")
    rng = random.Random(seed)
    if strategy == "random":
        selected = rng.sample(records, limit)
    elif strategy == "hybrid":
        n_random = random_count if random_count is not None else round(limit * 0.6)
        n_targeted = targeted_count if targeted_count is not None else limit - n_random
        if n_random + n_targeted != limit:
            raise ValueError("random_count + targeted_count must equal limit")
        random_part = rng.sample(records, n_random)
        selected_ids = {x.sample_id for x in random_part}
        pool = [x for x in records if x.sample_id not in selected_ids]
        targeted_part = _balanced_targeted_sample(pool, n_targeted, rng)
        selected = random_part + targeted_part
        rng.shuffle(selected)
    else:
        raise ValueError(f"unknown sampling strategy: {strategy}")
    for record in selected:
        record.metadata.random_seed = seed
    return selected


def _balanced_targeted_sample(
    records: list[SampleRecord], count: int, rng: random.Random
) -> list[SampleRecord]:
    buckets: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        text = record.positive_text.lower()
        for category, words in KEYWORDS.items():
            if any(re.search(rf"\b{re.escape(word)}\b", text) for word in words):
                buckets[category].append(record)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    chosen: list[SampleRecord] = []
    chosen_ids: set[str] = set()
    categories = list(KEYWORDS)
    cursor = 0
    while len(chosen) < count and categories:
        category = categories[cursor % len(categories)]
        while buckets[category] and buckets[category][-1].sample_id in chosen_ids:
            buckets[category].pop()
        if buckets[category]:
            item = buckets[category].pop()
            chosen.append(item)
            chosen_ids.add(item.sample_id)
            cursor += 1
        else:
            categories.remove(category)
            cursor = 0
    if len(chosen) < count:
        remainder = [x for x in records if x.sample_id not in chosen_ids]
        chosen.extend(rng.sample(remainder, count - len(chosen)))
    return chosen
