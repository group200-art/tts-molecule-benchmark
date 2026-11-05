"""Small, property-record-only metric kernels. No model/RDKit imports or file writes.

legacy_numeric_v1 reproduces the finite numerical legacy contract for regression.
v2_dir_guarded_legacy_rank_v1 adds explicit eligibility/validity and NA reporting.
SR_delta_decimal_v1 and RI_abs_eps01_v1 are separately named supplements.
"""
import math
from decimal import Decimal, InvalidOperation


def direction(prop):
    return -1 if prop == "mutagenicity" else 1


def finite_values(props, names):
    if not isinstance(props, dict):
        return False
    try:
        return all(not isinstance(props[p], bool) and math.isfinite(float(props[p])) for p in names)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def require_source(source, names):
    if not names or len(names) != len(set(names)) or not finite_values(source, names):
        raise ValueError("Task needs unique objectives and finite complete source properties")


def strictly_improves(source, candidate, names):
    return all(float(candidate[p]) < float(source[p]) if direction(p) < 0
               else float(candidate[p]) > float(source[p]) for p in names)


def legacy_selection_score(source, candidate, names):
    value = 0.0
    for p in names:
        x, y = float(source[p]), float(candidate[p])
        value += y if x == 0 else direction(p) * (y - x) / abs(x)
    return value


def legacy_report_ri(source, candidate, names):
    values = []
    for p in names:
        x, y = float(source[p]), float(candidate[p])
        values.append((y - 0.1) / 0.1 if x == 0 else abs(x - y) / abs(x))
    return sum(values) / len(values)


def ri_abs_eps01(source, candidate, names):
    require_source(source, names)
    if not finite_values(candidate, names):
        raise ValueError("Non-finite candidate")
    return sum(direction(p) * (float(candidate[p]) - float(source[p])) /
               max(abs(float(source[p])), 0.1) for p in names) / len(names)


def meets_delta(source, candidate, names, thresholds):
    require_source(source, names)
    if not finite_values(candidate, names):
        return False
    if set(thresholds) != set(names):
        raise ValueError("Thresholds must match the task's properties exactly")
    for p in names:
        try:
            threshold = Decimal(str(thresholds[p]))
        except (InvalidOperation, ValueError):
            raise ValueError("Invalid delta threshold")
        if not threshold.is_finite() or threshold <= 0:
            raise ValueError("Delta thresholds must be positive and finite")
        delta = Decimal(direction(p)) * (Decimal(str(candidate[p])) - Decimal(str(source[p])))
        if delta < threshold:
            return False
    return True


def legacy_best_finite(source, candidates, names):
    """Same rank and later-candidate tie rule as legacy, on the finite domain only.

    None denotes absent props. Non-finite/partial existing rows are rejected rather
    than silently asserted compatible with legacy. The guarded V2 kernel handles them.
    """
    require_source(source, names)
    best, score = None, -1.0
    for i, candidate in enumerate(candidates):
        if candidate is None:
            continue
        if not finite_values(candidate, names):
            raise ValueError("Outside the finite legacy compatibility domain")
        if strictly_improves(source, candidate, names):
            value = legacy_selection_score(source, candidate, names)
            if value >= score:
                best, score = i, value
    return best


def evaluate_input(input_uid, source, slots, names, thresholds, expected_slots=None):
    """Preserve every slot and return separate chemistry and property coverage.

    rdkit_valid must be an externally supplied bool from a pinned parser (not
    inferred from presence in a property table). This function never parses SMILES.
    """
    require_source(source, names)
    if not isinstance(input_uid, str) or not input_uid:
        raise ValueError("Stable input_uid is required")
    if expected_slots is not None and len(slots) != expected_slots:
        raise ValueError("Candidate-slot count mismatch; do not drop or refill slots")
    ids = [s["slot_id"] for s in slots]
    if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids) or ids != sorted(ids):
        raise ValueError("Unique nonnegative slot IDs must be in original ascending order")
    eligible, slot_status = [], []
    n_valid = 0
    for slot in slots:
        if type(slot.get("rdkit_valid")) is not bool:
            raise ValueError("Explicit pinned RDKit validity flag is required for every slot")
        chemical_valid = slot["rdkit_valid"] and bool(str(slot.get("smiles") or "").strip())
        n_valid += int(chemical_valid)
        props = slot.get("properties")
        has_properties = finite_values(props, names)
        eligible.append(props if chemical_valid and has_properties else None)
        slot_status.append({"slot_id": slot["slot_id"], "chemical_valid": chemical_valid,
                            "finite_complete_properties": has_properties,
                            "evaluation_eligible": chemical_valid and has_properties})
    best = legacy_best_finite(source, eligible, names)
    delta_candidates = [i for i, props in enumerate(eligible) if props is not None and meets_delta(source, props, names, thresholds)]
    # Independently select within the delta-successful set; do not reuse the dir winner.
    delta_pool = [props if i in delta_candidates else None for i, props in enumerate(eligible)]
    delta_best = legacy_best_finite(source, delta_pool, names)
    return {"input_uid": input_uid, "metric_version": "v2_dir_guarded_legacy_rank_v1",
            "SR_dir_success": best is not None, "SR_delta_success": delta_best is not None,
            "selected_slot_id_dir": ids[best] if best is not None else None,
            "selected_slot_id_delta": ids[delta_best] if delta_best is not None else None,
            "RI_legacy": legacy_report_ri(source, eligible[best], names) if best is not None else None,
            "RI_abs_eps01": ri_abs_eps01(source, eligible[best], names) if best is not None else None,
            "input_valid": n_valid > 0, "n_slots": len(slots), "n_valid_slots": n_valid,
            "n_eligible_slots": sum(p is not None for p in eligible), "slots": slot_status}


def aggregate(inputs):
    uids = [row["input_uid"] for row in inputs]
    if len(set(uids)) != len(uids):
        raise ValueError("Repeated input_uid would double-count inputs")
    n = len(inputs)
    total_slots = sum(r["n_slots"] for r in inputs)
    successes = [r for r in inputs if r["SR_dir_success"]]
    return {"n_inputs": n, "n_success_dir": len(successes),
            "SR_dir": 100 * len(successes) / n if n else None,
            "SR_delta": 100 * sum(r["SR_delta_success"] for r in inputs) / n if n else None,
            "Val_input": 100 * sum(r["input_valid"] for r in inputs) / n if n else None,
            "Val_slot": 100 * sum(r["n_valid_slots"] for r in inputs) / total_slots if total_slots else None,
            "property_eligible_slot_pct": 100 * sum(r["n_eligible_slots"] for r in inputs) / total_slots if total_slots else None,
            "RI_legacy": sum(r["RI_legacy"] for r in successes) / len(successes) if successes else None,
            "RI_abs_eps01": sum(r["RI_abs_eps01"] for r in successes) / len(successes) if successes else None}
