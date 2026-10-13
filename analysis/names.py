"""Readable labels for exported results; archived experiment records stay unchanged."""

METHOD_IDS = {
    'beam20': 'beam_search',
    'sampling_verifier20': 'sampling_with_verification',
    'self_refine_r1': 'self_refine',
    'tot_lm_vote': 'tree_of_thoughts',
    'got': 'graph_of_thoughts',
    's1': 's1_budget_forcing',
    'ev': 'vote',
    'fewshot': 'few_shot_icl',
    'reflexion': 'reflexion',
    's1-B512-sampled-R20': 's1_budget_forcing_sampled',
}

# Keep GeLLMO's metric names. Suffixes distinguish evaluator variants.
COLUMN_NAMES = {
    'Macro10': 'Macro_avg',
    'SR_delta': 'SR_threshold',
    'RI_abs_eps01': 'RI_stabilized',
    'RI_eps01': 'RI_stabilized',
    'RI_eps01_current': 'RI_stabilized',
    'RI_official_code': 'RI_original_code',
    'RI_legacy_cached': 'RI_original_cached',
    'Val_legacy_rdkit': 'Validity_original',
    'Validity_legacy': 'Validity_original',
    'Val_current': 'Validity_current',
    'Val_property_row_diagnostic': 'property_evaluation_coverage_pct',
    'Val_input': 'input_validity_pct',
    'Val_slot': 'candidate_validity_pct',
    'final_validity': 'Validity',
    'Sim_raw': 'Sim',
    'SAS_raw': 'SAS',
    'Sim_current_cached': 'Sim_cached',
    'SAS_current_cached': 'SAS_cached',
    'Nov_official': 'Novelty',
    'Novelty_official': 'Novelty',
    'Nov_raw_literal': 'Novelty_raw_smiles',
    'Nov_canonical_reference': 'Novelty_canonical_smiles',
    'APS_equal_target_task_mean': 'APS_macro_avg',
    'n_valid_legacy_rdkit': 'n_valid_original',
    'n_success_dir': 'n_success',
    'n_success_delta': 'n_threshold_success',
    'readout': 'output_selection',
    'r': 'num_revisions',
}

REFERENCE_GROUPS = {
    'Easy': 'reference_beam_solved',
    'Medium': 'reference_large_pool_only',
    'Hard': 'reference_unsolved',
    'OOD_Hard': 'OOD_reference_unsolved',
}

OUTPUT_SELECTION = {
    'N20_R1_K1': '20_votes_first_run',
    'N20_R20_K20': '20_votes_all_20_runs',
    'N400_R1_K1_reference': '400_votes_single_run',
    'R1_K1_first_repeat': 'first_run',
    'R20_K20': 'all_20_runs',
}

RESULT_DIRS = {
    'k1': 'k1',
    'original_metrics_k1': 'original_metrics_k1',
    'paper_tables': 'paper_tables',
    'p10_n20_r20': 'vote_20_votes_20_runs',
    'mistral_s1_properties': 'mistral_s1_sampled_20_runs',
    'mistral_self_refine_r1_r20': 'mistral_self_refine_1_revision_20_runs',
}


def column_name(name):
    if name in COLUMN_NAMES:
        return COLUMN_NAMES[name]
    if name.endswith('_Macro10'):
        return name[:-len('Macro10')] + 'Macro_avg'
    for old, new in REFERENCE_GROUPS.items():
        if name.startswith(old + '_'):
            return new + name[len(old):]
    return name


def renamed_row(row):
    result = {}
    for key, value in row.items():
        name = column_name(key)
        if name in result:
            raise ValueError('Duplicate column after renaming: ' + name)
        if key == 'method_id':
            value = METHOD_IDS[value]
        elif key == 'scope' and value == 'Macro10':
            value = 'Macro_avg'
        elif key == 'readout':
            value = OUTPUT_SELECTION[value]
        result[name] = value
    return result
