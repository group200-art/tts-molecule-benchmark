# Frozen K20 main comparison

This component reproduces the original **two-backbone, nine-method** main results
from saved per-input readouts. Each configuration has 3,905 task-input pairs and
20 fixed output positions: 70,290 metric records representing 1,405,800 positions
across all 18 configurations. The separate 500-input, seed-1729 GoT control is
outside this component.

The main reference is
`BenchmarkV2/recomb_main_task_table_20261105_v1/paper/data` in the source workspace.
The complete main readouts were frozen on 2026-10-27 and their final statistics
were computed on 2026-10-28. These are the current references, rather than the
earlier incomplete thirteen-configuration snapshot retained only for checks.

## Reproduce

Use Python 3.10 or later and NumPy 2.2.6 for the validated full calculation:

```bash
python -m pip install -r requirements.txt
python reproduce.py --output reproduced
```

Run from this directory, or invoke `reproduce.py` by its path from any directory.
All inputs are bundled. No scheduler, GPU, model weights, network access, RDKit,
generation or property prediction is required. On a managed cluster, run the
full command in an ordinary CPU allocation. The recorded check used one CPU and
4 GB requested memory, completed in seven seconds, and used about 545 MiB peak
RSS. Results may depend on the machine and BLAS implementation.

A standard-library check, including full quality/prefix aggregation and workload
checks but excluding bootstrap intervals, is also available:

```bash
python reproduce.py --skip-bootstrap --output quick_check
```

To render the current paper's composite figure (prefix curves and workload):

```bash
python -m pip install -r requirements-plot.txt
python reproduce.py --plot --output reproduced
```

`checks/reproduction_report.json` records the executed full check:
all three saved aggregation CSVs, nine published statistics CSVs, and the workload
CSV were byte-identical to their references; maximum numeric discrepancy was
zero across 11,034 compared numeric values. The current composite figure was
separately rendered with Matplotlib 3.10.8 and matched the reference PNG pixel for
pixel; `checks/figure_check.json` records that check. PNG/PDF metadata can differ.

## Coverage

| Quantity | Bundled input and reproduction |
| --- | --- |
| Per-task SR@20, IND/OOD means, equal ten-task macro | All 70,290 saved input outcomes |
| Fixed-prefix success curves | Saved flags at K = 1, 5, 10, 20; same output order and denominator |
| Valid output-position percentage | Saved valid-slot count divided by 20 per input, then equal task means |
| Threshold success SR-delta@20 | Saved per-input threshold-success flag, including failures in denominators |
| RI_0.1, Sim and SAS | Saved successful representative per input; within-task successful-subset means, then equal ten-task means |
| Macro and paired source-bootstrap intervals | All 3,905 input identities and 2,860 source-molecule clusters; original 2,000 replicates and seed 20261014 |
| Workload, ToT and Reflexion on both backbones | Reaggregated from 15,620 saved per-input counts |
| Workload, other fourteen configurations | Audited original totals retained; arithmetic means verified |

The prefixes are observed points at four K values; the compact archive cannot
recover outcomes at other K values. The archive contains derived readouts, not
all candidate SMILES, candidate properties, model responses or generation traces.
Thus this is an aggregation/statistics reproduction, not an independent
reassessment of molecular validity, eligibility, thresholds or representative
selection from raw candidates. Full-precision similarity-threshold analyses and
the separate fixed-pool/GoT-control experiments belong to other release components.

The workload table counts proposals, response sequences and returned output
tokens, including original reused-generation costs once. It excludes prompt
processing, property computation, unreturned beam work, loading, discarded
attempts and branches outside the evaluated configuration. It is not total
compute, latency, physical batched calls or oracle usage. Trace hashes are kept
in `provenance/workload*_source_index.json.gz`; those traces are not bundled here.

Successful-subset descriptor means need not concern the same inputs for different
methods. Sim and SAS retain the original comparison-round2 values. Bootstrap
intervals condition on these fixed saved generations; they are paired by source
molecule and do not measure generation-seed variance or adjust for multiplicity.
The twenty evaluated output positions do not equalize generation workload.

## Files and provenance

- `inputs/final/per_input_quality_and_prefix.jsonl.gz`: unchanged complete metric
  archive, SHA-256 `404d2e7e12d37ca71f32e41c543162f31c34d2e375879f1f434c8e1b021aa215`.
- `inputs/manifests/`: the ten unchanged input manifests, including task/input
  identifiers, source SMILES, recorded source properties and source-cluster IDs.
- `expected/`: unchanged current main numeric reference tables.
- `schema.json`: field definitions, aggregation contract and scope limitations.
- `provenance/source_manifest.json`: original workspace-relative source path,
  original hash, packaged hash, size and transformation for each source artifact.
- `provenance/original/`: unchanged original statistics, aggregation, workload and
  plot code. Historical absolute paths inside these files and audit receipts are
  provenance; run the portable entry point rather than these archived scripts.
- `scripts/aggregate_saved_metrics.py`: the original finalizer's mean function
  and aggregation block copied without changing numerical logic.
- `scripts/build_statistics_portable.py`: original statistical implementation
  with path adaptation and scheduler-only guard removal. The complete diff is
  `provenance/build_statistics_portability.patch`; no bootstrap, weighting,
  percentile, success, or paired-difference formula was changed.
- `manifest.json`: checksums for the portable component's source files; generated
  check outputs are recorded separately in the reproduction report.

The task order is BDP, BDQ, BPQ, DPQ, BDPQ (IND), then MPQ, BDMQ, BHMQ, BMPQ,
HMPQ (OOD). The first seven tasks each have 500 inputs; BHMQ has 118, BMPQ 191,
and HMPQ 96. Global input-pooled rates are not substituted for equal-task means.
