# GeLLMO: Generalizing Large Language Models for Multi-property Molecule Optimization
We introduce $\mathtt{MuMOInstruct}$, the first high-quality instruction-tuning dataset 
specifically focused on complex multi-property molecule optimization tasks. 
Leveraging $\mathtt{MuMOInstruct}$, we develop $\mathtt{GeLLM^3O}$, a series of instruction-tuned LLMs for molecule optimization.
Extensive evaluations across 5 in-domain and 5 out-of-domain
tasks demonstrate that $\mathtt{GeLLM^3O}$ consistently outperform state-of-the-art baselines. 
$\mathtt{GeLLM^3O}$ also exhibits outstanding zero-shot generalization to unseen tasks, significantly outperforming powerful closed-source LLMs.
Such strong generalizability demonstrates the tremendous potential of $\mathtt{GeLLM^3O}$ as foundational models for molecule optimization,
thereby tackling novel optimization tasks without resource-intensive retraining. 

## Requirements

Please use `pip install -r requirements.txt` to install dependies. Ensure you have `python >= 3.10.0` installed.


## Dataset

The instruction-tuning dataset $\mathtt{MuMOInstruct}$ is available on [HuggingFace](https://huggingface.co/collections/NingLab/gellmo-67b527a2d221f06d09a240ef). 

## Models

The instruction-tuned model checkpoints are available in [HuggingFace](https://huggingface.co/collections/NingLab/gellmo-67b527a2d221f06d09a240ef). 


## Training

To instruction-tune the base models, run the following:
```
bash train.sh $base_model $data_dir $expt_dir $num_epochs $tasks
```
- `$base_model` specifies the base model (either `mistralai/Mistral-7B-Instruct-v0.3` or `meta-llama/Llama-3.1-8B-Instruct`)
- `$data_dir` specifies the path to json files or HuggingFace dataset hub `NingLab/MuMOInstruct`
- `$expt_dir` specifies the path to the directory where the finetuned checkpoints and lora weights will be saved
- `$num_epochs` specifies the number of epochs
- `$tasks` specifies the list of tasks to be used for instruction-tuning
    - example 1: "['bbbp+drd2+qed']" to train a task-specific model optimizing only this task
    - example 2: "['bbbp','drd2','qed','bbbp+qed','bbbp+drd2','drd2+qed','bbbp+drd2+qed']" to train a generalist model on the superset of 3 properties: BBBP, DRD2 and QED

For faster runs, download the base models once locally in a folder and provide the local path into `$base_model`. Please download the base models: [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) and [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) following the HuggingFace instructions.  

## Inference

For model inference, run the following command:

```
bash inference.sh $base_model $data_dir $lora_weights $output_dir $task $num_seq $setting
```
- `$base_model` specifies the base model (either `mistralai/Mistral-7B-Instruct-v0.3` or `meta-llama/Llama-3.1-8B-Instruct`)
- `$data_dir` specifies the path to json files or HuggingFace dataset hub `NingLab/MuMOInstruct`
- `$lora_weights` specifies the local path or the HuggingFace model hub containing the adapter weights of the instruction-tuned model
- `$output_dir` specifies the output path where the LLM generated responses will be stored in JSON format
- `$task` specifies the task name
- `$num_seq` specifies the number of generated responses for each prompt (we set it to 20 in our experiments)
- `$setting` specifies whether to use seen or unseen instruction
    - Permitted values: `seen` or `unseen` 

Example of inference on the task BDPQ on the tuned $\mathtt{GeLLMO\text{-}P(6)_\text{Mistral}}$ in the seen instruction setting:
```
bash inference.sh mistralai/Mistral-7B-Instruct-v0.3 NingLab/MuMOInstruct NingLab/GeLLMO-P6-Mistral /tmp/ bbbp+drd2+plogp+qed 20 seen
```

## Evaluation

For the sake of easy reproducibility, we also provide processing and evaluation codes in `process-output.ipynb` and `evaluate.ipynb`, respectively.
Please note that to run `evaluate.ipynb`, you need to install a separate virtual environment
that supports running the utility functions for computing drd2 scores following the [official implementation](https://github.com/wengong-jin/hgraph2graph/tree/master).
We provide these utility functions in `data/props/` adapted from the official implementation.
We used a separate virtual env with the following python and library versions:
```
python = 3.6.13
pytorch = 1.6.0
rdkit = 2020.03.3.0
scikit-learn = 0.21.3
```

We provide a sample output in `examples/` which contains the following:
- `bbbp+drd2+qed_response.json`: file with the JSON output after running `inference.py` using the instruction tuned $\mathtt{GeLLM^{3}O\text{-}P(6)_{Mistral}}$ LoRA checkpoint available in [Huggingface](https://huggingface.co/NingLab/GeLLMO-P6-Mistral) for the task BDQ.
- `bbbp+drd2+qed-smiles.csv`: file containing comma separated list of SMILES strings that are successfully extracted from the JSON output. Use `process_output_llms()` provided in `process-output.ipynb` to generate this file.
- `bbbp+drd2+qed-admet_props.csv`: file containing the properties for each SMILES string after running ADMET-AI. Use `generate_props()` provided in `process-output.ipynb` to generate this file.
- `bbbp+drd2+qed-props.csv`: file containing the plogp, drd2, qed and sas scores along with ADMET-AI properties. Use `compute_props()` provided in `evaluate.ipynb` to generate this file.

Once all these files are generated, please use `compute_metrics()` in `evaluate.ipynb` to compute all metrics. We provided example usages of these functions in the scripts.