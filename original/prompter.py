import json
from typing import List
import random
import pandas as pd

from datasets import load_dataset, Dataset
from config import PROPERTY_FULL_NAMES, PROPERTY_IMPV_THRESHOLDS

class Prompter:

    def __init__(self, opt_type: str, 
                 response_split: str = "[/INST] %%% Response:",
                 template_path: str = 'templates/'
                 ):
        # read the prompt and property threshold from json files
        with open(f"{template_path}/{opt_type}.json", "r") as f:
            template_data = json.load(f)

        self.template_icl = template_data["template_icl"]
        self.template_zero = template_data["template"]

        self.system_prompt = template_data["system_prompt"]
        self.task_prompt = template_data["task_prompt"]
        self.task_prompt_llasmol = template_data["task_prompt_llasmol"]
        self.task_prompt_noexplain = template_data["task_prompt_no_explain"]
        self.instructions = template_data["instructions"]
        self.instructions.append(template_data["base_inst"])

        #self.property_change = template_data["property_change"]
        self.property_change = PROPERTY_IMPV_THRESHOLDS
        
        self.response_split = response_split


    def format_prompt(self,
                        input_smiles: str, 
                        change_properties: List[dict], 
                        prompt_type: str, 
                        #opt_type: str,
                        output_smiles: str,
                        add_response: bool) -> str:
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
            raise ValueError(f"Invalid task prompt type: {prompt_type}")

        num_properties = len(change_properties)
        template_key = f'prop{num_properties}'
        template = templates[template_key]

        format_dict = {"smiles": input_smiles}
        for i, prop in enumerate(change_properties):
            format_dict[f'property{i+1}'] = prop['property']
            format_dict[f'change{i+1}'] = prop['change']
            format_dict[f'value{i+1}'] = prop['value']
        
        #print(template, format_dict)
        if prompt_type == 'icl' or prompt_type == 'ins':
            prompt = f"%%% Input : <SMILES> {input_smiles} </SMILES>\n" + f"%%% Adjust: {template.format(**format_dict)}\n"
        else:
            prompt = f"{template.format(**format_dict)}\n"
        
        if add_response:
            if not output_smiles:
                raise ValueError("Output SMILES must be provided for prompts with response")
            prompt += f"%%% Response: <SMILES> {output_smiles} </SMILES>\n\n"

        return prompt


    def generate_prompt(self,
                         sample: dict,
                         sample_is_pair: bool,
                         prompt_type: str,
                         opt_type: str,
                         add_response: bool,
                         instr_setting: str = 'seen'
                         ) -> str:
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
        for i, prop in enumerate(task.split('+')):
            change = {}
            change['property'] = PROPERTY_FULL_NAMES[prop][-1] if instr_setting == 'unseen' else PROPERTY_FULL_NAMES[prop][0]
            change['change'] = 'increase' if prop !="mutagenicity" else 'decrease'
            change['value'] = None if opt_type == 'simple' else max(abs(prop_values[prop]['change']), self.property_change[prop]).__round__(2)
            change_properties.append(change)
        
        prompt = ""
        if prompt_type == 'ins':
            prompt = self.instructions[sample['instr_idx']] + "\n"
        prompt += self.format_prompt(input_smiles, change_properties, prompt_type, output_smiles, add_response)
        return prompt

    
    def generate_prompt_for_general_purpose_LLMs(self,
                                                data_path: str,
                                                task: str,
                                                prompt_type: str,
                                                model_id: str,
                                                opt_type: str,
                                                sampling: str,
                                                num_shots: int,
                                                prompt_explain: bool,
                                                test_sample_is_pair: bool = False,
                                                instr_setting: str = 'seen'
                                                ) -> List[str]:
        """
        Generates prompts for the given test sample with in-context examples for prompting general-purpose LLMs
        Args:
            data_path: The path to the data files
            task: The task for which the prompts are being generated
            prompt_type: The type of prompt to be generated
            model_id: The model ID for which the prompts are being generated
            opt_type: The type of optimization (simple or threshold)
            sampling: The type of sampling to be used for in-context examples
            num_shots: The number of in-context examples to be used
            prompt_explain: Whether to include explanations in the prompts
            test_sample_is_pair: Whether the test sample is a pair of smiles
        """
        
        test_file = f"{data_path}/test.json"
        icl_file = f"{data_path}/train.json"
        test_data = load_dataset("json", data_files=test_file)['train']
        icl_data = load_dataset("json", data_files=icl_file)['train']

        test_data = pd.DataFrame(test_data)
        icl_data = pd.DataFrame(icl_data)

        test_data = test_data[(test_data['task'] == task) & (test_data['instr_setting'] == instr_setting)]
        test_data = Dataset.from_pandas(test_data)
        icl_data = icl_data[(icl_data['split'] == 'train') & (icl_data['task'] == task)]
        icl_data = Dataset.from_pandas(icl_data)


        prompts = []
        task_prompt = self.task_prompt if prompt_explain else self.task_prompt_noexplain
        system_prompt = f"<<SYS>>\n{self.system_prompt}\n<</SYS>\n" if (model_id == 'llama' or model_id == "mistral") else ""

        if model_id == "llasmol":
            task_prompt = self.task_prompt_llasmol

        for index, sample in enumerate(test_data):
            test_prompt = self.generate_prompt(sample, 
                                                test_sample_is_pair, 
                                                prompt_type, 
                                                opt_type,
                                                add_response=False,
                                                instr_setting=instr_setting)
            
            # Generate a random sample of in-context examples from the training pairs
            in_context_examples = ""

            if num_shots:
                if sampling == 'random':
                    # randomly sample from list of dicts
                    rand_idx = random.sample(range(len(icl_data)), num_shots)
                    sampled_icl_samples = [icl_data[i] for i in rand_idx]
                elif sampling == 'directional':
                    sampled_icl_samples = []
                    for icl_sample in icl_data:
                        same_direction = True
                        for prop in sample['properties'].keys():
                            if icl_sample['properties'][prop]['change'] < 0 != sample['properties'][prop]['change'] < 0:
                                same_direction = False
                                break
                        if same_direction:
                            sampled_icl_samples.append(icl_sample)
                    
                    if len(sampled_icl_samples) < num_shots:
                        print(f"Test Pair {index}: Number of IC samples with same direction of change is less than {num_shots}")
                        continue
                
                    rand_idx = random.sample(range(len(sampled_icl_samples)), num_shots)
                    sampled_icl_samples = [sampled_icl_samples[i] for i in rand_idx]
                else:
                    raise ValueError(f"Invalid sampling method: {sampling}")
                
                for icl_sample in sampled_icl_samples:
                    icl_prompt = self.generate_prompt(icl_sample, True, 
                                                       prompt_type, 
                                                       opt_type,
                                                       add_response=True)
                    in_context_examples += icl_prompt
            
            # add [INST] tags for general purpose llms
            inst_open_tag = "[INST]\n" if model_id != "llasmol" else ""
            inst_close_tag = "\n[/INST]\n" if model_id != "llasmol" else ""
            response_tag = "%%% Response:"

            if prompt_type == 'pt0':
                prompt = f"{system_prompt}{inst_open_tag}" + in_context_examples + test_prompt + f"{inst_close_tag}{response_tag}\n"
            elif prompt_type == 'icl' and num_shots:
                prompt = f"{system_prompt}{inst_open_tag}{task_prompt}\n\n" + "Examples:\n" + in_context_examples + "Task:\n" + test_prompt + f"{inst_close_tag}{response_tag}\n"
            else:
                prompt = f"{system_prompt}{inst_open_tag}{task_prompt}\n\n" + "Task:\n" + test_prompt + f"{inst_close_tag}{response_tag}\n"
            prompts.append(prompt)
        
        return prompts
                                   

    def get_response(self, output: str) -> str:
        """
        Extracts the response from the output
        """
        response = output.split(self.response_split)[-1].strip()
        return response
