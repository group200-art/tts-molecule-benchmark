# Completed supplementary analyses

This directory contains the completed supplement frozen on November 4, 2026. It provides CPU reproduction from saved per-input outcomes and a scientific source snapshot of the code that produced them.

## Analyses and data

1. Strict Tanimoto similarity >0.6 on the original returned sequence, SR@1 and SR@20, all 3,905 task–input pairs and eighteen configurations.
2. Reordering the same twenty selected GoT candidates. This is distinct from regenerating or changing the selected set.
3. Vote selection from the same saved proposal groups, retaining the fixed pool and applying the original success definition.
4. A new paired GoT versus independent Sampling control with the same GoT scorer: 500 fixed inputs, two backbones and generation seed 1729. The arms actually share the initial twelve candidates, then each constructs sixteen additional candidates, ranks its pool of twenty-eight and returns twenty. This is not the original 3,905-input comparison.

The predefined primary endpoint is the paired difference in mean SR@1. Its confidence intervals include zero on both backbones; these results do not establish GoT superiority, equivalence, or that all differences arise from sorting. No Reflexion ablation or three-seed replication is included.

## Portable reproduction

Run the two `reproduce_metrics.py` commands in the repository root README from this directory. Python 3.10 and NumPy 2.2.6 suffice. The program checks all fixed inputs, source identities, point estimates and conditional paired intervals. Generated check reports are written under `reproduced/`.

`data/offline_per_input_metrics.csv.gz` supports 538 summaries and 58 paired contrasts. `data/got_control_per_input_metrics.csv.gz` supports 44 summaries and 14 paired contrasts. `results/` contains frozen per-task summaries, contrasts, workload accounting and candidate availability. `evidence/input_manifest_3905.csv` and `evidence/input_manifest_500.csv` independently specify the denominators.

Similarity uses full-precision Tanimoto on original RDKit 2020.03.3 Morgan fingerprints (radius 2, 2,048 bits, no chirality), with the strict `>` operator. Original success labels retain the original numerical precision. Invalid, empty and duplicate positions are not removed from output denominators. Missing records are never silently converted to failed generations.

## Source and external archives

`execution_snapshot/experiments/supplementary/` preserves the generation and analysis source. `execution_snapshot/scientific_sources/` preserves reused success, precision, property, scoring and selector implementations. These are scientific execution sources: their local paths and earlier planning constants require configuration before fresh generation. Internal user-planning documents, scheduler wrappers, troubleshooting and delivery scripts are omitted from the public copy. The portable statistics entry points do not depend on those omitted materials. The completed scope is specified in `evidence/scope.json`; completion is recorded by `results/got_control_status.json`. A legacy `subset_100` storage name does not mean the 996-input plan was completed.

Full new-control archives `paired_records.jsonl.gz` (raw prompts, responses, random streams and complete pools) and `candidate_records.jsonl.gz` are retained separately by the authors and are not part of this compact repository. Their hashes are in `evidence/full_candidate_archive.json`. If obtained separately, the portable checker can be run with:

```bash
python reproduce_got_candidates.py --pairs /path/to/paired_records.jsonl.gz \
  --metrics data/got_control_per_input_metrics.csv.gz \
  --manifest evidence/input_manifest_500.csv --expected-count 500 \
  --output reproduced/candidate_verification.json
```

This optional command does not turn the compact outcomes package into a complete raw-candidate archive. The original study's full response logs are also not all included. The repository root README and `results/candidate_availability.csv` describe the available evidence and omissions.

The original scientific sources and saved readouts are unchanged. Publication packaging does not run new molecular generation, training or property prediction.

`results/got_control_physical_property_workload.csv.gz` is a losslessly gzip-compressed CSV. Read it with Python `gzip.open(..., "rt")` or decompress with `gzip -dk`; its rows are unchanged.

`results/got_control_workload_public.csv` is a field projection of the per-arm workload table. It retains all 2,000 rows, their order, backbone/seed/task/input/arm identities, scientific counters and timings, and the short property-access and cost-basis descriptions. Every retained field is unchanged. The public projection omits `hardware_source` (internal filesystem paths), `measured_evaluator_calls` (verbose evaluator traces), and `shared_initial_record_ids` (raw trace identifiers). The complete original table remains in the authors' local archive and is not distributed here. This is a reduced public table, not a lossless copy of the full internal run log.
