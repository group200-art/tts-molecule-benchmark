#!/bin/sh

# Usage: inference.sh <base_model> <data_dir> <model_dir> <lora_weights> <output_dir> <task> <num_return_sequences>

base_model=$1
data_dir=$2
lora_weights=$3
output_dir=$4
task=$5
opt="simple"
num_return_sequences=$6
setting=$7
num_shots=0
sampling="random"

output_dir="$output_dir/$opt-${num_shots}-$setting-${num_return_sequences}-beam/output/"


python inference.py \
    --data_path $data_dir \
    --output_dir $output_dir \
    --lora_weights $lora_weights \
    --base_model $base_model \
    --setting $setting \
    --opt_type $opt \
    --task $task \
    --num_return_sequences $num_return_sequences