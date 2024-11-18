#!/bin/bash
#SBATCH --job-name=multi_gpu_job
#SBATCH --output=a100_job_%j.log
#SBATCH --ntasks=3
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=scavenge_gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --array=0-2

# Load necessary modules
module load GCC/12.2.0
module load CUDA/12.1.1
module load miniconda

conda activate ml-bench

models=(
    "Qwen_Qwen1.5-110B-Chat"
    "Qwen_Qwen1.5-72B-Chat"
    "Qwen_Qwen2-72B-Instruct"
    "meta-llama_Meta-Llama-3.1-405B-Instruct-Turbo"
    "meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo"
    "meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo"
)

job_array_task1=()
job_array_task2=()
job_array_task3=()

for model in "${models[@]}"; do
    # Task 1
    if [[ $model != *"405B"* ]]; then
        input_file="output/${model}-task1.jsonl"
        log_file="./logs/${model}-task1.log"
        job_array_task1+=("python utils/exec.py --input_path \"$input_file\" --dataset_path \"cache/ml-bench-merged\" --split \"full\" > \"$log_file\" 2>&1")
    fi
    
    # Task 2 (including 405B model)
    input_file="output/${model}-task2.jsonl"
    log_file="./logs/${model}-task2.log"
    job_array_task2+=("python utils/exec.py --input_path \"$input_file\" --dataset_path \"cache/ml-bench-merged\" --split \"full\" > \"$log_file\" 2>&1")
    
    # Task 3
    if [[ $model != *"405B"* ]]; then
        input_file="output/${model}-task3.jsonl"
        log_file="./logs/${model}-task3.log"
        job_array_task3+=("python utils/exec.py --input_path \"$input_file\" --dataset_path \"cache/ml-bench-merged\" --split \"full\" > \"$log_file\" 2>&1")
    fi
done

# Execute the job arrays in parallel
echo "Executing all tasks in parallel..."

# Function to run jobs in parallel
run_parallel() {
    local jobs=("$@")
    for job in "${jobs[@]}"; do
        eval "$job" > /dev/null 2>&1 &
    done
    wait
}

# Run each task array
run_parallel "${job_array_task1[@]}"
run_parallel "${job_array_task2[@]}"
run_parallel "${job_array_task3[@]}"

echo "All tasks completed."
