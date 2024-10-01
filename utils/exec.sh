models=(
    "Qwen_Qwen1.5-110B-Chat"
    "Qwen_Qwen2-72B-Instruct"
    "meta-llama_Meta-Llama-3.1-405B-Instruct-Turbo"
    "meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo"
    "meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo"
)

tasks=(1 2 3)

for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        input_file="output/${model}-task${task}.jsonl"
        log_file="./logs/${model}-task${task}.log"
        
        # Skip the 405B model for tasks 1 and 3
        if [[ $model == *"405B"* ]] && [[ $task != 2 ]]; then
            continue
        fi
        
        nohup python utils/exec.py --input_path "$input_file" --dataset_path "cache/ml-bench-merged" --split "full" > "$log_file" 2>&1 &
    done
done

wait