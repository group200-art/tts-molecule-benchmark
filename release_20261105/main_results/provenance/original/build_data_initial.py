"""Reconcile the three K20 tables from frozen records, without model inference."""
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

OUT = Path(__file__).resolve().parent
B = OUT.parent
PAPER = OUT / "paper"
DATA = PAPER / "data"
DATA.mkdir(exist_ok=True)
TASKS = ["BDP", "BDQ", "BPQ", "DPQ", "BDPQ", "MPQ", "BDMQ", "BHMQ", "BMPQ", "HMPQ"]
MODELS = ["Mistral", "Llama"]
METHODS = ["beam", "sampling", "self_refine", "tot", "got", "s1", "voting", "icl", "reflexion"]
KS = [1, 5, 10, 20]
DELTAS = dict(bbbp=.2, drd2=.2, plogp=1., qed=.1, hia=.1, mutagenicity=.1)
SOURCES = {}
EXPECTED = {}
GROUPS = defaultdict(dict)
WORKLOAD = {}


def track(path):
    p = Path(path).resolve()
    if str(p) not in SOURCES:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1048576), b""):
                h.update(block)
        SOURCES[str(p)] = dict(sha256=h.hexdigest(), bytes=p.stat().st_size)
    return p


def read(path):
    p = track(path)
    with (gzip.open(p, "rt") if p.suffix == ".gz" else p.open()) as f:
        return json.load(f)


def lines(path):
    p = track(path)
    with (gzip.open(p, "rt") if p.suffix == ".gz" else p.open()) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csvout(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def mean(xs):
    xs = list(xs)
    assert xs
    return math.fsum(xs) / len(xs)


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-9, rel_tol=1e-11), (a, b)


for task in TASKS:
    for r in lines(B / ("manifests/" + task + ".test.seen.jsonl")):
        EXPECTED[task, r["input_uid"]] = r
assert len(EXPECTED) == 3905

draft = B / "recomb_k20_table_draft_20261018_v1"
published = list(csv.DictReader(track(draft / "per_task.csv").open()))
summary = list(csv.DictReader(track(draft / "table1_k20.csv").open()))


def quality(source, props):
    ri = mean((-1 if k == "mutagenicity" else 1) * (props[k] - x) / max(abs(x), .1)
              for k, x in source.items())
    assert finite(props["sim"]) and finite(props["sas"])
    # Use retained comparison-round2 values consistently for every configuration.
    assert all(abs(props[k] - round(props[k], 2)) < 1e-9 for k in ["sim", "sas"])
    return dict(RI_01=ri, Sim=props["sim"], SAS=props["sas"])


def add(model, method, record, candidates, archived=None):
    task, uid = record["task_id"], record["input_uid"]
    key = task, uid
    assert key in EXPECTED and key not in GROUPS[model, method]
    source = record["source_properties"]
    assert source == EXPECTED[key]["source_properties"]
    assert len(candidates) == 20
    assert [c["original_slot_id"] for c in candidates] == list(range(20))
    valid, success, delta = [], [], []
    for c in candidates:
        chem = bool(c["rdkit_valid"] and c["raw_smiles"].strip())
        p = c.get("properties") or {}
        eligible = chem and all(finite(p.get(k)) for k in source)
        ok = eligible and all(p[k] < x if k == "mutagenicity" else p[k] > x
                              for k, x in source.items())
        strong = ok and all(Decimal(-1 if k == "mutagenicity" else 1) *
                            (Decimal(str(p[k])) - Decimal(str(x))) >= Decimal(str(DELTAS[k]))
                            for k, x in source.items())
        valid.append(chem)
        success.append(bool(ok))
        delta.append(bool(strong))
    pool = [i for i in range(20) if success[i]]

    def score(i):
        p = candidates[i]["properties"]
        return sum(p[k] if x == 0 else
                   (-1 if k == "mutagenicity" else 1) * (p[k] - x) / abs(x)
                   for k, x in source.items())

    selected = max(pool, key=lambda i: (score(i), i)) if pool else None
    row = dict(task_id=task, input_uid=uid, success=bool(pool), delta=any(delta),
               n_valid_slots=sum(valid), representative_slot=selected,
               RI_01=None, Sim=None, SAS=None,
               **{"SR_K" + str(k): any(success[:k]) for k in KS})
    if selected is not None:
        row.update(quality(source, candidates[selected]["properties"]))
    if archived is not None:
        assert row["success"] == archived.get("SR_dir_success", archived.get("success"))
        if "SR_delta_success" in archived:
            assert row["delta"] == archived["SR_delta_success"]
            assert row["n_valid_slots"] == archived["n_valid_slots"]
            assert selected == archived["selected_slot_id_dir"]
            if selected is not None:
                close(row["RI_01"], archived["RI_abs_eps01"])
        elif selected is not None:
            assert candidates[selected]["candidate_id"] == archived["representative_candidate_id"]
            assert all(candidates[selected]["properties"][k] == x
                       for k, x in archived["representative_properties"].items())
    GROUPS[model, method][key] = row


# The native candidate pools are unchanged for Beam, Sampling and GoT.
for model in MODELS:
    for method, old_id in [("beam", "beam20"), ("sampling", "sampling40_prefix20"), ("got", "got")]:
        if (model, method) == ("Mistral", "got"):
            continue
        for task in TASKS:
            if model == "Mistral":
                path = B / ("next3_ten_tasks_run01_v1/results/tasks/" + old_id + "__" + task + ".json")
            else:
                path = B / ("p6_llama_failure_recovery_20261003_v1/results/by_task/" + old_id + "__" + task + ".json")
            for r in read(path)["inputs"]:
                add(model, method, r, r["candidates"], r)
        print("AUDITED", model, method, flush=True)

# This accepted GoT export has compact readouts rather than full candidate arrays.
got_path = B / "method_review_P8_v1/design_after_p9_v1/execution_v1/recovery_execution_v1/results/per_input_topK.jsonl.gz"
got = defaultdict(dict)
for r in lines(got_path):
    if r["G"] == 12 and r["K"] in KS:
        key = r["task_id"], r["input_uid"]
        assert r["K"] not in got[key]
        got[key][r["K"]] = r
assert set(got) == set(EXPECTED)
for key, views in got.items():
    assert set(views) == set(KS)
    r = views[20]
    assert r["source_properties"] == EXPECTED[key]["source_properties"] and r["n_slots"] == 20
    row = dict(task_id=key[0], input_uid=key[1], success=r["SR_dir_success"],
               delta=r["SR_delta_success"], n_valid_slots=r["n_valid_slots"],
               representative_slot=r["selected_slot_id_dir"], RI_01=None, Sim=None, SAS=None,
               **{"SR_K" + str(k): views[k]["SR_dir_success"] for k in KS})
    if row["success"]:
        row.update(quality(r["source_properties"], r["selected_property_values"]))
        close(row["RI_01"], r["RI_abs_eps01"])
    GROUPS["Mistral", "got"][key] = row
print("AUDITED Mistral got (accepted compact readouts)", flush=True)

for task in TASKS:
    for r in read(B / ("self_refine_r1_k20_v1/results/" + task + "_K20.json.gz"))["rows"]:
        add("Mistral", "self_refine", r, r["candidates"], r)
print("AUDITED Mistral self_refine", flush=True)

icl_tokens = 0
for path in sorted((B / "k20_remaining_three_v1/runs").glob("fewshot_*/result.json.gz")):
    d = read(path)
    assert d["status"] == "complete" and not d["probe"]
    for r in d["rows"]:
        add("Mistral", "icl", r, r["candidates"], r)
        for c in r["candidates"]:
            answer = c["trace"]["answer"]
            assert isinstance(answer["output_tokens"], int)
            icl_tokens += answer["output_tokens"]
WORKLOAD["Mistral", "icl"] = dict(proposals=78100, responses=78100, tokens=icl_tokens,
                                basis="Complete 20-terminal traces, one answer per repeat")
print("AUDITED Mistral icl", flush=True)

policy_roots = [
    ("Mistral", "s1", B / "p10_n20_r20_two_models_20261013_v1/next_k20/mistral_s1_properties"),
    ("Llama", "s1", B / "benchmark_paired_k1_k20_20261014_v1/llama_s1/properties"),
    ("Llama", "icl", B / "p10_n20_r20_two_models_20261013_v1/next_k20/llama_icl_properties"),
]
for model, method, root in policy_roots:
    old = {(r["task_id"], r["input_uid"]): r for r in lines(root / "metrics/per_input.jsonl.gz")
           if r["readout"] == "R20_K20"}
    for r in lines(root / "results/terminal_inputs.jsonl.gz"):
        add(model, method, r, r["candidates"], old[r["task_id"], r["input_uid"]])
    print("AUDITED", model, method, flush=True)

vroot = B / "p10_n20_r20_two_models_20261013_v1"
properties = {(r["model_id"], r["task_id"], r["input_uid"], r["raw_smiles"]): r
              for r in lines(vroot / "properties/results/winner_properties.jsonl.gz")}
old_votes = {(r["model_id"], r["task_id"], r["input_uid"]): r
             for r in lines(vroot / "results/per_input.jsonl.gz") if r["readout"] == "N20_R20_K20"}
for model in MODELS:
    for r in lines(vroot / ("voting/grouped_inputs_" + model.lower() + ".jsonl.gz")):
        assert len(r["winners"]) == 20 and r["N"] == r["R"] == r["K_out"] == 20
        candidates = []
        for i, w in enumerate(r["winners"]):
            assert w["round_idx"] == i
            p = properties[r["model_id"], r["task_id"], r["input_uid"], w["raw_smiles"]]
            candidates.append(dict(p, candidate_id=w["candidate_id"], original_slot_id=i))
        add(model, "voting", r, candidates, old_votes[r["model_id"], r["task_id"], r["input_uid"]])
    print("AUDITED", model, "voting", flush=True)

# Eight rows reuse exactly the same generated pools as the audited K1 workload.
old_cost_path = B / "naacl_long_paper_20261012_v1/final_sources/data/table2_k1_workload.csv"
for r in csv.DictReader(track(old_cost_path).open()):
    method = dict(beam20="beam", sampling_verifier20="sampling", got="got", ev="voting").get(r["method"])
    if method:
        WORKLOAD[r["model"].capitalize(), method] = dict(
            proposals=int(r["molecular_proposals_total"]),
            responses=int(r["logical_response_sequences_total"]),
            tokens=int(r["returned_output_tokens_total"]),
            basis="Unchanged generated pool; audited original full-pool token accounting")

assert len(GROUPS) == 13
task_rows, macro_rows, curves, compact = [], [], [], []
for (model, method), inputs in GROUPS.items():
    assert set(inputs) == set(EXPECTED)
    per_task = []
    for task in TASKS:
        rows = [r for (t, u), r in inputs.items() if t == task]
        good = [r for r in rows if r["success"]]
        assert good and all(all(finite(r[k]) for k in ["RI_01", "Sim", "SAS"]) for r in good)
        expected = next(r for r in published if (r["model"], r["method_id"], r["task_id"]) == (model, method, task))
        assert len(rows) == int(expected["n_inputs"]) and len(good) == int(expected["n_success"])
        r = dict(model=model, method_id=method, task_id=task, n_inputs=len(rows),
                 n_success=len(good), n_delta=sum(r["delta"] for r in rows),
                 n_valid_slots=sum(r["n_valid_slots"] for r in rows),
                 SR20=100 * len(good) / len(rows),
                 Valid_slots=100 * sum(r["n_valid_slots"] for r in rows) / (20 * len(rows)),
                 SR_delta20=100 * sum(r["delta"] for r in rows) / len(rows),
                 **{k: mean(x[k] for x in good) for k in ["RI_01", "Sim", "SAS"]})
        per_task.append(r)
        task_rows.append(r)
        for k in KS:
            curves.append(dict(model=model, method_id=method, task_id=task, K=k,
                               n_inputs=len(rows), n_success=sum(r["SR_K" + str(k)] for r in rows),
                               SR=100 * mean(r["SR_K" + str(k)] for r in rows)))
    macro_rows.append(dict(model=model, method_id=method, n_inputs=3905, n_slots=78100,
                           n_success=sum(r["n_success"] for r in per_task),
                           **{k: mean(r[k] for r in per_task)
                              for k in ["SR20", "Valid_slots", "SR_delta20", "RI_01", "Sim", "SAS"]}))
    for r in inputs.values():
        assert all(not r["SR_K" + str(a)] or r["SR_K" + str(b)] for a, b in zip(KS, KS[1:]))
        compact.append(dict(model=model, method_id=method, **r))

csvout(DATA / "table3_k20_quality.csv", macro_rows)
csvout(DATA / "table3_k20_quality_by_task.csv", task_rows)
csvout(DATA / "k20_prefix_coverage.csv", curves)
with gzip.open(OUT / "per_input_quality_and_prefix.jsonl.gz", "wt") as f:
    for r in compact:
        f.write(json.dumps(r, allow_nan=False) + "\n")
save("workload_verified_pools.json", [
    dict(model=m, method_id=p, **r) for (m, p), r in WORKLOAD.items()])
save("source_index.json", SOURCES)
save("data_audit.json", dict(status="PASS", checked_utc=datetime.now(timezone.utc).isoformat(),
     configurations=13, task_groups=len(task_rows), task_inputs=len(compact), output_slots=20 * len(compact),
     agreement_with_existing_table1="All 130 task success counts match exactly",
     chemistry_or_model_inference=False, quality_selection="Frozen guarded legacy score; later slot wins exact ties",
     quality_precision="Retained comparison-round2 Sim and SAS for every configuration",
     quality_aggregation="Within-task successful representative means, then equal ten-task macro",
     validity_denominator="20 fixed slots per input, invalid/empty/duplicate slots retained",
     delta_rule="At least one eligible slot meets all registered Decimal improvement thresholds",
     RI="Directional gain divided by max(abs(source), 0.1), mean across target properties",
     compact_readout_exception="Mistral GoT uses accepted archived per-input K1/K5/K10/K20 readouts",
     missing_configurations=["Mistral tot", "Llama tot", "Mistral reflexion", "Llama reflexion", "Llama self_refine"]))
print("PASS: 13 configurations, 50,765 input rows, 1,015,300 output slots", flush=True)
