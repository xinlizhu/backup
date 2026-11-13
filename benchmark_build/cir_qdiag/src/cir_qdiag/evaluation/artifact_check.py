from __future__ import annotations

from collections import Counter
from typing import Any

from cir_qdiag.schemas import SampleRecord


def run_text_only_baseline(records: list[SampleRecord], seed: int = 42) -> dict[str, Any]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:
        raise RuntimeError("install the evaluation extra: pip install -e '.[evaluation]'") from exc
    texts: list[str] = []
    labels: list[int] = []
    groups: list[int] = []
    for record in records:
        if not record.negatives:
            continue
        texts.append(record.positive_text)
        labels.append(1)
        groups.append(record.pair_id)
        for negative in record.negatives:
            texts.append(negative.text)
            labels.append(0)
            groups.append(record.pair_id)
    if len(set(groups)) < 5 or len(set(labels)) < 2:
        raise ValueError("not enough grouped positive/negative samples for artifact baseline")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_indices, test_indices = next(splitter.split(texts, labels, groups))
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    model.fit([texts[i] for i in train_indices], [labels[i] for i in train_indices])
    predictions = model.predict([texts[i] for i in test_indices])
    accuracy = float(accuracy_score([labels[i] for i in test_indices], predictions))
    return {
        "accuracy": accuracy,
        "flag_above_60_percent": accuracy > 0.60,
        "train_examples": len(train_indices),
        "test_examples": len(test_indices),
        "train_pairs": len({groups[i] for i in train_indices}),
        "test_pairs": len({groups[i] for i in test_indices}),
        "class_counts": dict(Counter(labels)),
        "split_grouped_by_pair_id": True,
    }
