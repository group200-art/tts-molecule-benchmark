"""Rebuild long-paper Tables 2--4 using audited, unchanged final K=1 outputs.

Run with the standard Python library; no generation, chemistry, or prediction.
The frozen source snapshot supports table reproduction, while endpoint hashes,
membership and numerical checks establish that the readout has not changed.
"""
import csv
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

OUT = Path(__file__).resolve().parent
SRC = OUT / "sources"
TASKS = ["BDP", "BDQ", "BPQ", "DPQ", "BDPQ", "MPQ", "BDMQ", "BHMQ", "BMPQ", "HMPQ"]
COUNTS = dict(zip(TASKS, [500] * 7 + [118, 191, 96]))
MODELS = ["gellmo_p6_mistral", "gellmo_p6_llama"]
METHODS = ["beam20", "sampling_verifier20", "self_refine_r1", "tot_lm_vote", "got", "s1", "ev", "fewshot", "reflexion"]
LABELS = [r"Beam ($w=20$)", "Sampling + task verifier", r"Self-Refine ($r=1$)",
          "ToT (LM voting)", "GoT (G12)", r"s1-style ($B=512$)",
          r"Exact Voting ($N=400$)", "4-shot ICL", "Reflexion (max 3 trials)"]
MODEL_LABELS = ["GeLLMO-P6-Mistral (7B)", "GeLLMO-P6-Llama (8B)"]
METRICS = {"final_validity": "Val_slot", "SR_delta": "SR_delta",
           "RI_eps01": "RI_abs_eps01", "Sim": "Sim_cached", "SAS": "SAS_cached"}
DELTAS = {"bbbp": .2, "drd2": .2, "plogp": 1., "qed": .1, "hia": .1, "mutagenicity": .1}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(name):
    return json.loads((SRC / name).read_text())


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10), (a, b)


def quality_table(rows):
    lines = [r"% Reproducible ten-task macro means of unchanged final K=1 endpoints.",
             r"\begin{table*}[t]", r"\centering\EvalTableFont",
             r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.10}",
             r"\caption{Final-output validity, magnitude-sensitive success, and molecular quality. All five columns are equal macro averages over all ten tasks, using the same 3,905 inputs per model--method configuration and single final outputs ($K=1$) as Table~\ref{tab:effectiveness}.}",
             r"\label{tab:quality}\label{tab:generalization}\label{tab:input_regime_reliability}",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l r r r r r@{}}", r"\toprule",
             r"Method & \shortstack{Final validity\\(\%)} & \shortstack{SR$_\Delta$\\(\%)} & RI$_{0.1}$ & Sim & SAS \\"]
    for model, model_label in zip(MODELS, MODEL_LABELS):
        lines += [r"\midrule", r"\multicolumn{6}{l}{\textbf{" + model_label + r"}} \\"]
        for method, label in zip(METHODS, LABELS):
            r = next(r for r in rows if (r["model_id"], r["method_id"]) == (model, method))
            cells = [f"{r[k]:.{2 if k in ['final_validity', 'SR_delta'] else 3}f}" for k in METRICS]
            lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule\end{tabular*}", r"\par\vspace{5pt}\noindent\begin{minipage}{\linewidth}\TableNoteFont",
              r"\textbf{Rates.} Invalid and unsuccessful final outputs remain in the denominators. Validity means RDKit-valid, not successful property evaluation. SR$_\Delta$ requires all target improvements to meet the stated magnitude thresholds; it supplements strict-direction SR without changing the method's selected output.",
              r"\par\smallskip",
              r"\textbf{Conditional quality.} RI$_{0.1}$, Sim, and SAS first average directionally successful final outputs within each task, then average all ten tasks equally. RI$_{0.1}$ uses the stabilized denominator $\max(|x_j|,0.1)$; Sim is source--output Morgan-fingerprint similarity; lower SAS indicates easier synthesis. Successful subsets differ across methods, so these means are descriptive, not paired quality improvements. No values are bolded. Sim/SAS retain the recorded two-decimal conversion before averaging; formulas and thresholds are given in the text.",
              r"\end{minipage}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def workload_table():
    old = (SRC / "table2_workload.tex").read_text()
    tab = re.search(r"\\begin\{tabular\*\}.*?\\end\{tabular\*\}", old, re.S).group()
    transformed, original_rows, new_rows = [], [], []
    for line in tab.splitlines():
        if " & " in line:
            cells = line.removesuffix(r" \\").split(" & ")
            assert len(cells) == 6
            if not line.startswith("Method &"):
                original_rows.append(cells)
                new_rows.append(cells[:5])
            line = " & ".join(cells[:5]) + r" \\"
        line = line.replace(r"l r r r l r@{}", r"l r r r l@{}").replace(r"\multicolumn{6}", r"\multicolumn{5}")
        transformed.append(line)
    assert len(original_rows) == len(new_rows) == 18
    assert all(old[:5] == new for old, new in zip(original_rows, new_rows))
    text = "\n".join([
        r"% Existing workload values preserved; Table 1 alone reports main success rates.",
        r"\begin{table*}[t]", r"\centering\EvalTableFont",
        r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.10}",
        r"\caption{Recorded generation workload and inference-time property-tool access. Means include all 3,905 inputs per configuration, including failures, for the same final-output procedures ($K=1$) as Table~\ref{tab:effectiveness}.}",
        r"\label{tab:workload}\label{tab:tts_efficiency}",
        "\n".join(transformed),
        r"\par\vspace{5pt}\noindent\begin{minipage}{\linewidth}\TableNoteFont",
        r"Proposals count returned molecular candidates, including unselected ones; LM sequences additionally count reasoning, feedback, reflection, and judging responses. Reused generation prefixes are charged once. Sampling counts all 40 generated proposals although selection scans only the first 20; Self-Refine includes terminal feedback; ToT includes 20 proposals and ten judge responses. Output tokens use each backbone's tokenizer and include the first EOS, excluding prompts and later padding. These are recorded generation statistics, not physical calls, latency, FLOPs, or total compute: unreturned beam-search work and property-tool costs are not included. Property-tool categories describe inference access, not query counts; \emph{None} still permits final evaluation. Internal budgets and tool access are disclosed rather than equalized.",
        r"\end{minipage}", r"\end{table*}", ""])
    return text, original_rows, new_rows


def difficulty_table():
    old = (SRC / "table3_generalization.tex").read_text()
    blocks = re.findall(r"\\begin\{tabular\*\}.*?\\end\{tabular\*\}", old, re.S)
    assert len(blocks) == 2
    original = blocks[1]
    tab = original.replace("B. Shared Mistral-reference difficulty groups", "Shared Mistral-reference difficulty groups")
    assert re.findall(r"\d+(?:\.\d+)?", original) == re.findall(r"\d+(?:\.\d+)?", tab)
    return "\n".join([
        r"% Former Table 3 Panel B, with every numeric cell and highlight unchanged.",
        r"\begin{table*}[t]", r"\centering\EvalTableFont",
        r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.10}",
        r"\caption{Success by shared Mistral-reference difficulty. Both models and all nine methods use the same fixed input groups; every displayed method result evaluates a single final output ($K=1$). Rates pool inputs within each group.}",
        r"\label{tab:difficulty}", tab,
        r"\par\vspace{5pt}\noindent\begin{minipage}{\linewidth}\TableNoteFont",
        r"The retrospective reference partition uses Mistral candidate-pool coverage: Easy if Beam20 succeeds; Medium if Beam20 fails but Beam100 or Sample160 succeeds; Hard otherwise. These historical pools define labels only and are not the displayed evaluation setting. Failures remain in all rate denominators. Bold marks the highest displayed success rate per column within each model.",
        r"\par\smallskip",
        r"$\dagger$ Mistral Beam has zero Medium/Hard success and Mistral Sampling zero Hard success by construction; Llama has no corresponding constraint. These are reference-relative groups, not model-independent difficulty, and cannot establish causal cross-model superiority. The 27 OOD Hard inputs comprise 2 MPQ, 24 BDMQ, and 1 BMPQ cases; one success changes the pooled rate by 3.70 percentage points.",
        r"\end{minipage}", r"\end{table*}", ""])


def main():
    snapshots = read("snapshot_manifest.json")
    for name, entry in snapshots.items():
        assert sha(SRC / name) == entry["sha256"], name
    old_audit = read("table3_prior_audit.json")
    assert old_audit["status"] == "PASS_TABLE3_K1_FULL_PAIRED_DATA"
    for name in ["task_metrics.json", "endpoint_audit.json", "methods.json", "manifests.json"]:
        source = snapshots[name]["source_path"]
        assert snapshots[name]["sha256"] == old_audit["sources"][source]
    config, manifests, endpoint_audit = read("methods.json"), read("manifests.json"), read("endpoint_audit.json")
    assert config["models"] == MODELS and config["methods"] == METHODS and config["tasks"] == TASKS
    assert endpoint_audit["status"] == "COMPLETE_FULL3905_TWO_MODELS_NINE_METHODS_K1"
    assert endpoint_audit["final_records"] == 70290 and endpoint_audit["unknown_property_failures"] == 0
    metrics_list = read("task_metrics.json")
    metrics = {(r["model_id"], r["method_id"], r["task_id"]): r["metrics"] for r in metrics_list}
    expected = {(m, p, t) for m in MODELS for p in METHODS for t in TASKS}
    assert len(metrics_list) == len(metrics) == 180 and set(metrics) == expected
    membership, source_props = {}, {}
    for task in TASKS:
        spec = manifests[task]
        assert sha(spec["path"]) == spec["sha256"]
        rr = [json.loads(line) for line in Path(spec["path"]).read_text().splitlines() if line.strip()]
        membership[task] = {r["input_uid"] for r in rr}
        assert len(rr) == len(membership[task]) == COUNTS[task]
        source_props.update({(task, r["input_uid"]): r["source_properties"] for r in rr})
    checked_groups = defaultdict(list)
    seen = set()
    for path, digest in endpoint_audit["source_files"].items():
        assert digest == old_audit["sources"][path] == sha(path)
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                model, method, task, uid = (r[k] for k in ["model_id", "method_id", "task_id", "input_uid"])
                key = model, method, task, uid
                assert key not in seen and (model, method, task) in expected
                assert uid in membership[task] and r["source_properties"] == source_props[task, uid]
                seen.add(key)
                c, e = r["selected_candidate"], r["evaluation"]
                assert e["n_slots"] == 1 and e["candidates"] == [c] and c["original_slot_id"] == 0
                valid = bool(c["rdkit_valid"] and c["raw_smiles"].strip())
                assert e["input_valid"] == valid and e["n_valid_slots"] == int(valid)
                src, props = r["source_properties"], c["properties"]
                finite = isinstance(props, dict) and all(isinstance(props.get(p), (int, float)) and math.isfinite(props[p]) for p in src)
                success = bool(valid and finite and all((props[p] < x) if p == "mutagenicity" else (props[p] > x) for p, x in src.items()))
                assert success == e["SR_dir_success"]
                delta = bool(success and all(Decimal(-1 if p == "mutagenicity" else 1) * (Decimal(str(props[p])) - Decimal(str(x))) >= Decimal(str(DELTAS[p])) for p, x in src.items()))
                assert delta == e["SR_delta_success"]
                quality = {}
                if success:
                    assert e["selected_property_values"] == props
                    ri = sum((-1 if p == "mutagenicity" else 1) * (props[p] - x) / max(abs(x), .1) for p, x in src.items()) / len(src)
                    close(ri, e["RI_abs_eps01"])
                    quality = {"RI_abs_eps01": ri, "Sim_cached": props["sim"], "SAS_cached": props["sas"]}
                    assert all(math.isfinite(v) for v in quality.values())
                else:
                    assert e["RI_abs_eps01"] is None and e["selected_property_values"] is None
                checked_groups[model, method, task].append(dict(valid=valid, success=success, delta=delta, **quality))
    assert len(seen) == 70290 and set(checked_groups) == expected
    flat = []
    for key in sorted(expected):
        rows, m = checked_groups[key], metrics[key]
        assert len(rows) == m["n_inputs"] == m["n_slots"] == COUNTS[key[2]]
        good = [r for r in rows if r["success"]]
        assert len(good) == m["n_success_dir"] == m["Sim_cached_available_count"] == m["SAS_cached_available_count"] > 0
        counts = dict(n_success_dir=len(good), n_success_delta=sum(r["delta"] for r in rows), n_valid_final=sum(r["valid"] for r in rows))
        assert counts["n_success_delta"] == m["n_success_delta"]
        independent = {"Val_slot": 100 * counts["n_valid_final"] / len(rows), "SR_delta": 100 * counts["n_success_delta"] / len(rows),
                       **{k: sum(r[k] for r in good) / len(good) for k in ["RI_abs_eps01", "Sim_cached", "SAS_cached"]}}
        for metric, value in independent.items():
            close(value, m[metric])
        flat.append(dict(model_id=key[0], method_id=key[1], task_id=key[2], n_inputs=len(rows), **counts, **independent))
    with (SRC / "task_metrics_prior.csv").open() as f:
        prior = {(r["model_id"], r["method_id"], r["task_id"]): r for r in csv.DictReader(f)}
    for row in flat:
        old = prior[row["model_id"], row["method_id"], row["task_id"]]
        for k in ["n_inputs", "n_success_dir", "n_success_delta", "n_valid_final"] + list(METRICS.values()):
            close(row[k], float(old[k]))
    all_task = []
    for model in MODELS:
        for method in METHODS:
            mm = [metrics[model, method, task] for task in TASKS]
            row = dict(model_id=model, method_id=method, n_tasks=10, n_inputs=sum(m["n_inputs"] for m in mm),
                       n_success_dir=sum(m["n_success_dir"] for m in mm), n_success_delta=sum(m["n_success_delta"] for m in mm),
                       n_valid_final=sum(round(m["Val_slot"] * m["n_inputs"] / 100) for m in mm))
            assert row["n_inputs"] == 3905
            row.update({label: math.fsum(m[k] for m in mm) / 10 for label, k in METRICS.items()})
            all_task.append(row)
    write_csv("table3_quality.csv", all_task)
    write_csv("task_metrics.csv", flat)
    save("table3_quality.json", dict(final_k=1, tasks=TASKS, aggregation="equal ten-task macro; RI/Sim/SAS condition on success within each task", rows=all_task))
    (OUT / "table3_quality.tex").write_text(quality_table(all_task))
    workload, old_cost_rows, new_cost_rows = workload_table()
    (OUT / "table2_workload.tex").write_text(workload)
    (OUT / "table4_difficulty.tex").write_text(difficulty_table())
    (OUT / "table4_difficulty.csv").write_bytes((SRC / "difficulty_prior.csv").read_bytes())
    generated = ["table3_quality.csv", "task_metrics.csv", "table3_quality.json", "table3_quality.tex", "table2_workload.tex", "table4_difficulty.tex", "table4_difficulty.csv"]
    save("audit.json", dict(status="PASS", checked_utc=datetime.now(timezone.utc).isoformat(), configurations=18, task_groups=180,
        final_endpoints=len(seen), final_k=1, per_configuration_inputs=3905, task_counts=COUNTS,
        success_endpoints_with_complete_quality=sum(r["n_success_dir"] for r in flat), missing_quality=0,
        unknown_property_failures=0, endpoint_source_hashes=endpoint_audit["source_files"],
        endpoint_and_auxiliary_selectors_unchanged=True, task_metrics_independently_recomputed=True,
        ten_task_macro_uses_unrounded_task_means=True, table2_18_rows_cost_and_tool_cells_unchanged=True,
        table2_removed_only_success_column=True, table4_data_and_bold_unchanged=True,
        new_generation=0, new_property_predictions=0, original_table2_rows=old_cost_rows, updated_table2_rows=new_cost_rows,
        sources=snapshots, output_sha256={name: sha(OUT / name) for name in generated},
        limitations=["Conditional quality uses differing successful subsets; it is not a paired quality comparison.",
          "SR_delta and RI_eps01 are supplementary benchmark metrics, not unmodified original-paper metrics.",
          "Difficulty partition is retrospective and Mistral-reference dependent; both models nevertheless use exactly the same input subsets.",
          "Equal final K does not equalize internal generation budgets, public property tools, or total compute."]))
    print(json.dumps({"status": "PASS", "endpoints": len(seen), "task_groups": len(flat), "rows": len(all_task), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
