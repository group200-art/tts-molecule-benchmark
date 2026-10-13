import json
from typing import List
PROPERTY_FULL_NAMES = {
    "plogp": ["Penalized octanol-water partition coefficient (penalized logP)", "Penalized logP", 
              "Penalized logP which is logP penalized by synthetic accessibility score and number of large rings"],
    "qed": ["QED", "Quantitative Estimate of Drug-likeness (QED)", "drug-likeness quantified by QED score"],
    "drd2": ["DRD2 inhibition", "Dopamine receptor D2 inhibition probability", "inhibition probability of Dopamine receptor D2"],
    "bbbp": ["BBB permeability", "BBBP", "Blood-brain barrier permeability (BBBP)"],
    "mutagenicity": ["Mutagenicity", "Mutagenicity predicted by Ames test", "probability to induce genetic alterations (mutagenicity)"],
    "hia": ["Intestinal adsorption", "probability to be absorbed in the intestine", "human intestinal adsorption ability"],
}
PROPERTY_IMPV_THRESHOLDS = {
    "bbbp": 0.2,
    "drd2": 0.2,
    "hia": 0.1,
    "mutagenicity": 0.1,
    "plogp": 1.0,
    "qed": 0.1,
}

class Prompter:

    def __init__(self, opt_type: str, response_split: str='[/INST] %%% Response:', template_path: str='templates/'):
        with open(f'{template_path}/{opt_type}.json', 'r') as f:
            template_data = json.load(f)
        self.template_icl = template_data['template_icl']
        self.template_zero = template_data['template']
        self.system_prompt = template_data['system_prompt']
        self.task_prompt = template_data['task_prompt']
        self.task_prompt_llasmol = template_data['task_prompt_llasmol']
        self.task_prompt_noexplain = template_data['task_prompt_no_explain']
        self.instructions = template_data['instructions']
        self.instructions.append(template_data['base_inst'])
        self.property_change = PROPERTY_IMPV_THRESHOLDS
        self.response_split = response_split

    def format_prompt(self, input_smiles: str, change_properties: List[dict], prompt_type: str, output_smiles: str, add_response: bool) -> str:
        """
        Generates a prompt for the given smiles and list of properties, and template type
        Each entry in `properties` is a dictionary with keys property, change, and value
        `prompt_type` can be 'simple' or 'thresh'
        """
        if prompt_type == 'icl' or prompt_type == 'ins':
            templates = self.template_icl
        elif prompt_type == 'pt0':
            templates = self.template_zero
        else:
            raise ValueError(f'Invalid task prompt type: {prompt_type}')
        num_properties = len(change_properties)
        template_key = f'prop{num_properties}'
        template = templates[template_key]
        format_dict = {'smiles': input_smiles}
        for (i, prop) in enumerate(change_properties):
            format_dict[f'property{i + 1}'] = prop['property']
            format_dict[f'change{i + 1}'] = prop['change']
            format_dict[f'value{i + 1}'] = prop['value']
        if prompt_type == 'icl' or prompt_type == 'ins':
            prompt = f'%%% Input : <SMILES> {input_smiles} </SMILES>\n' + f'%%% Adjust: {template.format(**format_dict)}\n'
        else:
            prompt = f'{template.format(**format_dict)}\n'
        if add_response:
            if not output_smiles:
                raise ValueError('Output SMILES must be provided for prompts with response')
            prompt += f'%%% Response: <SMILES> {output_smiles} </SMILES>\n\n'
        return prompt

    def generate_prompt(self, sample: dict, sample_is_pair: bool, prompt_type: str, opt_type: str, add_response: bool, instr_setting: str='seen') -> str:
        """
        Generates a prompt for the given pair of smiles and list of properties with or without a response
        Args:
            sample: A dictionary containing the input smiles, target smiles, and property values
            properties: A list of properties to be changed
            prompt_type: The type of prompt to be generated
            opt_type: The type of optimization (simple or threshold)
            add_response: Whether to add the target smiles to the prompt
        """
        input_smiles = sample['source_smiles']
        output_smiles = sample['target_smiles'] if sample_is_pair else None
        prop_values = sample['properties']
        task = sample['task']
        change_properties = []
        for (i, prop) in enumerate(task.split('+')):
            change = {}
            change['property'] = PROPERTY_FULL_NAMES[prop][-1] if instr_setting == 'unseen' else PROPERTY_FULL_NAMES[prop][0]
            change['change'] = 'increase' if prop != 'mutagenicity' else 'decrease'
            change['value'] = None if opt_type == 'simple' else max(abs(prop_values[prop]['change']), self.property_change[prop]).__round__(2)
            change_properties.append(change)
        prompt = ''
        if prompt_type == 'ins':
            prompt = self.instructions[sample['instr_idx']] + '\n'
        prompt += self.format_prompt(input_smiles, change_properties, prompt_type, output_smiles, add_response)
        return prompt

    def get_response(self, output: str) -> str:
        """
        Extracts the response from the output
        """
        response = output.split(self.response_split)[-1].strip()
        return response
