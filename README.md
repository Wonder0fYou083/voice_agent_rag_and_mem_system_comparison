# Reproducible Voice-Agent Memory Benchmark

This directory contains the complete local pilot derived from
`test_agent.service_78.log`.

## Run

From the workspace root:

```bash
MPLCONFIGDIR=/tmp/mpl-benchmark python3 experiment/run_experiment.py
python3 experiment/validate_experiment.py
```

The run is deterministic except for small local timing variation.

## Files

```text
experiment/
├── run_experiment.py
├── validate_experiment.py
├── REPORT.md
├── data/
│   ├── canonical_events.jsonl
│   ├── benchmark_queries.jsonl
│   ├── mem0_style_memories.jsonl
│   └── mem9_style_memories.jsonl
└── results/
    ├── metrics.csv
    ├── category_metrics.csv
    ├── paired_comparisons.csv
    ├── query_results.csv
    ├── failures.csv
    ├── log_stats.json
    └── benchmark_summary.png
```

## Important interpretation rule

`mem0_style` and `mem9_style` are controlled local architectural proxies.
They are not calls to the hosted Mem0 or mem9 products. The names indicate
selective/consolidated fact memory versus event-preserving shared memory.

To run a real comparison, replace `build_memory_corpus()` with API adapters and
leave the benchmark queries, retrieval implementations, and metrics unchanged.

