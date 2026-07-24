#!/usr/bin/env python3
"""Sanity checks for the generated benchmark and result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    events = load_jsonl(ROOT / "data" / "canonical_events.jsonl")
    queries = load_jsonl(ROOT / "data" / "benchmark_queries.jsonl")
    mem0 = load_jsonl(ROOT / "data" / "mem0_style_memories.jsonl")
    mem9 = load_jsonl(ROOT / "data" / "mem9_style_memories.jsonl")
    metrics = pd.read_csv(ROOT / "results" / "metrics.csv")
    details = pd.read_csv(ROOT / "results" / "query_results.csv")

    assert len(events) >= 220, f"unexpected event count: {len(events)}"
    assert len(queries) == 40
    assert len({q["query_id"] for q in queries}) == len(queries)
    assert len(metrics) == 6
    assert len(details) == 40 * 6
    assert set(metrics["memory_system"]) == {"mem0_style", "mem9_style"}
    assert set(metrics["retriever"]) == {"current_rag", "graph_rag", "sag"}

    for column in [
        "memory_precision",
        "memory_recall",
        "recall_at_1",
        "recall_at_3",
        "mrr",
        "evidence_complete_accuracy",
        "stale_hit_rate",
        "scope_leakage_rate",
    ]:
        assert metrics[column].between(0, 1).all(), column

    current_gold: dict[str, str] = {}
    for event in sorted(events, key=lambda x: (x["timestamp"], x["event_id"])):
        if (
            event["should_store"]
            and event["memory_kind"] == "memory"
            and event["slot"]
            and event["value"]
        ):
            current_gold[event["slot"]] = event["value"]

    for corpus_name, corpus in [("mem0", mem0), ("mem9", mem9)]:
        pairs = {(m["slot"], m["value"]) for m in corpus}
        missing = [
            (slot, value)
            for slot, value in current_gold.items()
            if (slot, value) not in pairs
        ]
        assert not missing, f"{corpus_name} missing current gold memories: {missing}"

    # The selective corpus should be meaningfully smaller than the event-preserving one.
    assert len(mem0) < len(mem9) / 5

    # Every result payload must be valid JSON and contain at most top-k=3 records.
    for payload in details["retrieved"]:
        decoded = json.loads(payload)
        assert len(decoded) <= 3

    print(
        "Validation passed:",
        f"{len(events)} events, {len(queries)} queries,",
        f"{len(mem0)} mem0-style records, {len(mem9)} mem9-style records,",
        "6 evaluated configurations.",
    )


if __name__ == "__main__":
    main()
