#!/usr/bin/env python3
"""Reproduce the original K20 main comparison from frozen saved readouts.

No generation, property prediction, RDKit, model weights or external data access.
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
EXPECTED = ROOT / "expected"
TASKS = ["BDP", "BDQ", "BPQ", "DPQ", "BDPQ", "MPQ", "BDMQ", "BHMQ", "BMPQ", "HMPQ"]
METHODS = ["beam", "sampling", "self_refine", "tot", "got", "s1", "voting", "icl", "reflexion"]
MODELS = ["Mistral", "Llama"]
CONFIGS = [(model, method) for model in MODELS for method in METHODS]
STATS_FILES = ["macro_uncertainty.csv", "paired_vs_beam.csv", "table3_k20_consistency.csv",
               "per_task.csv", "k20_prefix_macro.csv", "k20_prefix_coverage.csv",
               "table3_k20_quality.csv", "table3_k20_quality_by_task.csv", "table1_k20.csv"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def csvread(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def csvwrite(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_csv(actual, expected):
    a, e = csvread(actual), csvread(expected)
    assert len(a) == len(e), (actual.name, len(a), len(e))
    assert list(a[0]) == list(e[0]), (actual.name, "column mismatch")
    numeric_checks, max_error = 0, 0.0
    for i, (left, right) in enumerate(zip(a, e)):
        for key in left:
            lv, rv = left[key], right[key]
            try:
                ln, rn = float(lv), float(rv)
            except (ValueError, TypeError):
                assert lv == rv, (actual.name, i, key, lv, rv)
            else:
                assert math.isfinite(ln) and math.isfinite(rn), (actual.name, i, key)
                error = abs(ln - rn)
                assert math.isclose(ln, rn, rel_tol=1e-12, abs_tol=1e-10), (actual.name, i, key, lv, rv)
                max_error = max(max_error, error)
                numeric_checks += 1
    return dict(rows=len(a), numeric_values_checked=numeric_checks, maximum_absolute_error=max_error,
                actual_sha256=sha(actual), reference_sha256=sha(expected), byte_identical=sha(actual) == sha(expected))


def verify_sources():
    manifest = json.loads((ROOT / "provenance/source_manifest.json").read_text())
    for record in manifest["records"]:
        path = ROOT / record["path"]
        assert path.is_file(), str(path)
        assert sha(path) == record["sha256"], "Changed release source: " + str(path)
        assert path.stat().st_size == record["bytes"], str(path)
        if record["transformation"].startswith("gzip encoding"):
            assert hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == record["source_sha256"]
        else:
            assert record["source_sha256"] == record["sha256"]
    # The package manifest additionally covers code and documentation, when present.
    package = ROOT / "manifest.json"
    if package.exists():
        for record in json.loads(package.read_text())["files"]:
            assert sha(ROOT / record["path"]) == record["sha256"], record["path"]
    return len(manifest["records"])


def load_metrics():
    expected = {}
    task_sizes = {}
    for task in TASKS:
        records = jsonl(INPUTS / "manifests" / (task + ".test.seen.jsonl"))
        task_sizes[task] = len(records)
        for record in records:
            key = task, record["input_uid"]
            assert record["task_id"] == task and key not in expected
            assert record["domain"] == ("IND" if TASKS.index(task) < 5 else "OOD")
            expected[key] = record
    assert len(expected) == 3905
    assert len({r["source_molecule_uid"] for r in expected.values()}) == 2860
    groups = defaultdict(dict)
    for record in jsonl(INPUTS / "final/per_input_quality_and_prefix.jsonl.gz"):
        config = record["model"], record["method_id"]
        key = record["task_id"], record["input_uid"]
        assert config in CONFIGS and key in expected and key not in groups[config]
        assert type(record["success"]) is bool and type(record["delta"]) is bool
        assert type(record["n_valid_slots"]) is int and 0 <= record["n_valid_slots"] <= 20
        assert not record["delta"] or record["success"]
        prefix = [record["SR_K" + str(k)] for k in (1, 5, 10, 20)]
        assert all(type(x) is bool for x in prefix) and prefix == sorted(prefix)
        assert record["success"] == prefix[-1]
        if record["success"]:
            assert record["n_valid_slots"] > 0 and type(record["representative_slot"]) is int
            assert 0 <= record["representative_slot"] < 20
            assert all(isinstance(record[k], (int, float)) and math.isfinite(record[k])
                       for k in ("RI_01", "Sim", "SAS"))
        else:
            assert all(record[k] is None for k in ("representative_slot", "RI_01", "Sim", "SAS"))
        groups[config][key] = {k: v for k, v in record.items() if k not in ("model", "method_id")}
    assert set(groups) == set(CONFIGS) and all(set(group) == set(expected) for group in groups.values())
    return groups, expected, task_sizes


def aggregate_quality(output, groups, expected):
    spec = importlib.util.spec_from_file_location("saved_aggregation", ROOT / "scripts/aggregate_saved_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    task, macro, prefix = module.aggregate(groups, expected)
    checks = {}
    for filename, rows in [("per_task.csv", task), ("macro.csv", macro), ("prefix_coverage.csv", prefix)]:
        path = output / ("aggregated_" + filename)
        csvwrite(path, rows)
        checks[path.name] = compare_csv(path, INPUTS / "final" / filename)
    return checks


def aggregate_workload(output, expected):
    rows = csvread(INPUTS / "workload/baseline_table2_k20_workload.csv")
    assert len(rows) == 18
    groups = defaultdict(dict)
    for record in jsonl(INPUTS / "workload/new_workload_per_input.jsonl.gz"):
        config = record["model"], record["method_id"]
        key = record["task_id"], record["input_uid"]
        assert config in [(m, method) for m in MODELS for method in ("tot", "reflexion")]
        assert key in expected and key not in groups[config]
        assert all(type(record[k]) is int and record[k] > 0 for k in ("proposals", "responses", "tokens"))
        groups[config][key] = record
    assert len(groups) == 4 and all(set(group) == set(expected) for group in groups.values())
    audit = json.loads((ROOT / "provenance/k20_workload_audit.json").read_text())
    for row in rows:
        config = row["model"], row["method_id"]
        assert int(row["n_inputs"]) == 3905 and int(row["output_slots_per_input"]) == 20
        if config in groups:
            assert row["status"] == "full_run_pending"
            detail = audit["new_details"]["/".join(config)]
            for field, key in [("molecular_proposals", "proposals"), ("response_sequences", "responses"), ("output_tokens", "tokens")]:
                count = sum(r[key] for r in groups[config].values())
                assert count == detail[key]
                row[field + "_total"] = count
                row[field + "_mean"] = count / 3905
            row["status"] = "complete_generation"
            row["basis"] = detail["basis"]
        else:
            assert row["status"] == "complete_generation"
            for field in ("molecular_proposals", "response_sequences", "output_tokens"):
                assert math.isclose(float(row[field + "_mean"]), int(row[field + "_total"]) / 3905,
                                    rel_tol=1e-12, abs_tol=1e-10)
    path = output / "table2_k20_workload.csv"
    csvwrite(path, rows)
    return dict(comparison=compare_csv(path, EXPECTED / path.name),
                configurations_reaggregated_from_per_input=4,
                configurations_retained_from_audited_totals=14,
                per_input_records=15620,
                limitation="The other 14 totals are retained from audited aggregate records; their underlying traces are not bundled.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Check and aggregate readouts/workload without NumPy or bootstrap.")
    parser.add_argument("--plot", action="store_true", help="Also render fixed-prefix figure; requires Matplotlib.")
    args = parser.parse_args()
    output = args.output.resolve()
    forbidden = [INPUTS, EXPECTED, ROOT / "provenance", ROOT / "scripts", ROOT / "figures"]
    assert all(output != p and p not in output.parents for p in forbidden), "Choose a separate output directory."
    assert output != ROOT, "Choose a separate output directory."
    assert not (args.plot and args.skip_bootstrap), "--plot requires the full reproduction."
    output.mkdir(parents=True, exist_ok=True)
    sources = verify_sources()
    groups, expected, task_sizes = load_metrics()
    checks = aggregate_quality(output, groups, expected)
    workload = aggregate_workload(output, expected)
    env = dict(os.environ, MAIN_RESULTS_OUTPUT=str(output), PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    stats_checks, bootstrap = {}, None
    if not args.skip_bootstrap:
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/build_statistics_portable.py")],
                                env=env, capture_output=True, text=True)
        (output / "statistics.stdout.txt").write_text(result.stdout)
        (output / "statistics.stderr.txt").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError("Statistics failed; inspect " + str(output / "statistics.stderr.txt"))
        for name in STATS_FILES:
            stats_checks[name] = compare_csv(output / name, EXPECTED / name)
        bootstrap = json.loads((output / "statistics_audit.json").read_text())
        for field in ("bootstrap_replicates", "bootstrap_seed", "source_clusters", "cluster_weights_sha256_int64"):
            original = json.loads((ROOT / "provenance/statistics_audit.json").read_text())
            assert bootstrap[field] == original[field], field
        if args.plot:
            result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/render_prefix_portable.py")],
                                    env=env, capture_output=True, text=True)
            (output / "figure.stdout.txt").write_text(result.stdout)
            (output / "figure.stderr.txt").write_text(result.stderr)
            if result.returncode:
                raise RuntimeError("Figure rendering failed; inspect " + str(output / "figure.stderr.txt"))
    report = dict(status="PASS_SAVED_READOUTS_ONLY" if args.skip_bootstrap else "PASS_MAIN_RESULTS_REPRODUCTION",
                  scope="Original two-backbone nine-method K20 comparison; excludes separate 500-input seed-1729 control",
                  input_source_artifacts_verified=sources, configurations=18, task_inputs_per_configuration=3905,
                  saved_metric_rows=70290, fixed_output_slots=1405800, task_sizes=task_sizes, source_clusters=2860,
                  aggregate_checks=checks, published_statistics_checks=stats_checks, workload=workload,
                  bootstrap=None if bootstrap is None else {key: bootstrap[key] for key in
                      ("bootstrap_replicates", "bootstrap_seed", "numpy_version", "cluster_weights_sha256_int64",
                       "unchanged_old_statistics_checked", "unchanged_old_statistics_max_error", "uncertainty_scope")},
                  generation_calls=0, property_prediction_calls=0, chemistry_calls=0,
                  plotting="rendered from reproduced prefix table; PDF metadata need not be byte-identical" if args.plot else "not requested",
                  python_version=sys.version.split()[0])
    (output / "reproduction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
