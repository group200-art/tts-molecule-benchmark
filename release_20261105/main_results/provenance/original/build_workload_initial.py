"""Count complete recorded K20 trajectories, including reused prefixes once."""
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
B = OUT.parent
SOURCES = {}
TASKS = ["BDP", "BDQ", "BPQ", "DPQ", "BDPQ", "MPQ", "BDMQ", "BHMQ", "BMPQ", "HMPQ"]
EXPECTED = set()


def read(path):
    p = Path(path)
    raw = p.read_bytes()
    SOURCES[str(p.resolve())] = hashlib.sha256(raw).hexdigest()
    return json.loads(gzip.decompress(raw) if p.suffix == ".gz" else raw)


for task in TASKS:
    for line in (B / ("manifests/" + task + ".test.seen.jsonl")).read_text().splitlines():
        r = json.loads(line)
        EXPECTED.add((task, r["input_uid"]))

costs = {(r["model"], r["method_id"]): r for r in read(OUT / "workload_verified_pools.json")}


def finish(model, method, seen, proposals, responses, tokens, basis):
    assert seen == {(t, u, i) for t, u in EXPECTED for i in range(20)}, (model, method, len(seen))
    assert all(isinstance(v, int) and v > 0 for v in [proposals, responses, tokens])
    costs[model, method] = dict(model=model, method_id=method, proposals=proposals,
                               responses=responses, tokens=tokens, basis=basis)
    print("WORKLOAD", model, method, proposals, responses, tokens, flush=True)


for model, root, reuse_seeds in [
    ("Mistral", B / "self_refine_r1_k20_v1", [51001, 51002, 51003]),
    ("Llama", B / "benchmark_paired_k1_k20_20261014_v1/llama_self_refine", [51001]),
]:
    seen, proposals, responses, tokens = set(), 0, 0, 0
    for seed in reuse_seeds:
        for task in TASKS:
            p = root / ("reuse/" + str(seed) + "/" + task + ".json.gz") if model == "Mistral" else root / ("reuse/" + task + ".json.gz")
            d = read(p)
            rows = d["rows"] if model == "Mistral" else d
            for r in rows:
                record = r if model == "Mistral" else r["record"]
                key = record["task_id"], record["input_uid"], seed - 51001
                assert key not in seen
                seen.add(key)
                c = r["prefix_generation_cost"] if model == "Mistral" else r["provenance"]["prefix_generation_cost"]
                assert c["r"] == 1 and c["model_calls"] == 4 and c["generated_slots"] == 2
                proposals += c["generated_slots"]
                responses += c["model_calls"]
                tokens += c["output_tokens"]
    paths = sorted((root / "generation").glob("*/batches/*.json.gz")) if model == "Mistral" else sorted((root / "generation/units").glob("*/*.json.gz"))
    for p in paths:
        d = read(p)
        assert d["status"] in ["complete", "GENERATION_COMPLETE_PROPERTIES_PENDING"]
        batch_tokens = 0
        for lane in d["lanes"]:
            r = lane["record"]
            key = r["task_id"], r["input_uid"], (d["rep_seed"] if model == "Mistral" else lane["rep_seed"]) - 51001
            assert key not in seen
            seen.add(key)
            assert len(lane["answers"]) == len(lane["feedbacks"]) == 2
            sequences = lane["answers"] + lane["feedbacks"]
            batch_tokens += sum(x["output_tokens"] for x in sequences)
            proposals += 2
            responses += 4
        assert batch_tokens == sum(m["output_tokens"] for m in d["measurements"])
        tokens += batch_tokens
    finish(model, "self_refine", seen, proposals, responses, tokens,
           "All 20 one-revision trajectories, including two feedback responses and original reused-prefix costs")


for model, roots in [
    ("Mistral", [B / "k20_remaining_three_v1/runs",
                 B / "p10_n20_r20_two_models_20261013_v1/next_k20/mistral_s1_resume/runs"]),
    ("Llama", [B / "benchmark_paired_k1_k20_20261014_v1/llama_s1/runs"]),
]:
    seen, proposals, responses, tokens = set(), 0, 0, 0
    # The first root owns original completed batches; continuation owns only missing batches.
    for root in roots:
        pattern = "s1_*/*/generation.json.gz" if model == "Mistral" else "*/*/generation.json.gz"
        for p in sorted(root.glob(pattern)):
            d = read(p)
            assert d["status"] == "complete"
            for lane in d["lanes"]:
                r = lane["record"]
                key = r["task_id"], r["input_uid"], lane["repeat"]
                assert key not in seen, ("duplicate generation lane", key, str(p))
                seen.add(key)
                tr = lane["trajectory"]
                stages = [v for v in tr["stage_summary"].values() if isinstance(v, dict) and v.get("call_index") is not None]
                assert len(stages) in [2, 3] and len(stages) == len(tr["call_indices"])
                assert sum(s["output_tokens"] for s in stages) == tr["total_output_tokens"]
                proposals += 1
                responses += len(stages)
                tokens += tr["total_output_tokens"]
    finish(model, "s1", seen, proposals, responses, tokens,
           "All initial reasoning, actual Wait continuations, and final-answer responses; successful retries counted once")

root = B / "p10_n20_r20_two_models_20261013_v1/next_k20/llama_icl_sampled/runs"
seen, tokens = set(), 0
for p in sorted(root.glob("*/*.json.gz")):
    d = read(p)
    assert d["status"] == "complete"
    for lane in d["outputs"]:
        r = lane["record"]
        key = r["task_id"], r["input_uid"], lane["repeat"]
        assert key not in seen
        seen.add(key)
        tokens += lane["output_tokens"]
finish("Llama", "icl", seen, len(seen), len(seen), tokens, "All 20 sampled final-answer sequences per input")

tools = dict(got="Search and selection", reflexion="Feedback and stopping")
rows = []
for model in ["Mistral", "Llama"]:
    for method in ["beam", "sampling", "self_refine", "tot", "got", "s1", "voting", "icl", "reflexion"]:
        c = costs.get((model, method))
        rows.append(dict(model=model, method_id=method, n_inputs=3905, output_slots_per_input=20,
                         molecular_proposals_total=c["proposals"] if c else None,
                         response_sequences_total=c["responses"] if c else None,
                         output_tokens_total=c["tokens"] if c else None,
                         molecular_proposals_mean=c["proposals"] / 3905 if c else None,
                         response_sequences_mean=c["responses"] / 3905 if c else None,
                         output_tokens_mean=c["tokens"] / 3905 if c else None,
                         inference_property_tools=tools.get(method, "None"),
                         status="complete_generation" if c else "full_run_pending",
                         basis=c["basis"] if c else "No full-run count available"))
assert sum(r["status"] == "complete_generation" for r in rows) == 14
with (OUT / "paper/data/table2_k20_workload.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(OUT / "workload_source_index.json").write_text(json.dumps(SOURCES, indent=2) + "\n")
(OUT / "workload_audit.json").write_text(json.dumps(dict(status="PASS", complete_configurations=14,
    reused_prefixes_included=True, physical_calls_not_conflated_with_sequences=True,
    excludes="prompt processing, unreturned beam work, retries, model loading and property computation",
    missing=["Mistral tot", "Llama tot", "Mistral reflexion", "Llama reflexion"]), indent=2) + "\n")
